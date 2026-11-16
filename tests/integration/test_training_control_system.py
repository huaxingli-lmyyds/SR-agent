import json
from pathlib import Path

import pytest

from agent.agents.communication import AgentTaskRequest
from agent.agents.hpo_agent import HPOAgent
from agent.hpo import HPOService
from agent.memory import MemoryService
from agent.runners import RUNNER_ADAPTERS
from agent.utils import ExperimentTracker
from tests.managed_fakes import ControlledProcessRunner


@pytest.mark.parametrize("managed", [False, True])
def test_early_stopped_trials_skip_evaluation_and_resume_without_retraining(
    tmp_path, minimal_config, dataset_dir, monkeypatch, managed
):
    runner = ControlledProcessRunner()
    monkeypatch.setitem(RUNNER_ADAPTERS, runner.runner, runner)
    pairs = tmp_path / "validation.txt"
    pairs.write_text("1 val1/a.wav val1/b.wav\n0 val1/a.wav val2/a.wav\n")
    experiments = tmp_path / "experiments"
    agent = HPOAgent(
        config_path=str(minimal_config),
        runner=runner.runner,
        implementation="fake",
        verbose=False,
        experiments_dir=str(experiments),
        memory_service=MemoryService(tmp_path / "memory"),
    )
    request = AgentTaskRequest(
        action=agent.action,
        objective="minimize verification EER within budget",
        context={
            "data_folder": str(dataset_dir),
            "strategy": "grid_search",
            "controller_mode": "fixed",
            "search_space": {
                "parameters": [
                    {
                        "name": "lr",
                        "parameter_type": "categorical",
                        "choices": [0.001, 0.002],
                    }
                ]
            },
            "budgets": [{"stage": "full", "epochs": 5}],
            "runtime_options": {
                "managed_execution": managed,
                "max_concurrent_trials": 2 if managed else 1,
                "device": "cpu",
                "verification_config": str(minimal_config),
                "validation_pairs": str(pairs),
                "training_exclusion_pairs": str(pairs),
                "training_control": {"min_epochs": 2, "patience": 1},
            },
        },
        budget={"max_training_runs": 2, "max_retries": 0},
    )
    result = agent.run_workflow(request)
    assert result.status == "success", result.to_dict()
    experiment = result.experiment_ids["hpo"]
    service = HPOService(ExperimentTracker(experiments))
    trials = service.list_trials(experiment)
    assert {t.status for t in trials} == {"completed", "stopped"}
    stopped = next(t for t in trials if t.status == "stopped")
    assert stopped.stop_reason == "validation_plateau"
    assert stopped.cost["training_control"]["epoch"] == 2
    from agent.hpo.feedback import HPOFeedbackAnalyzer

    feedback = HPOFeedbackAnalyzer().analyze(
        service.load_study(experiment), trials
    )
    example = feedback["training_control"]["stopped_examples"][0]
    assert example["parameters"]["lr"] == 0.002
    assert example["observed_epoch"] == 2
    assert example["curve"]["valid_loss"]["last"] == 1.0
    assert (
        result.summary["campaign"]["study_summaries"][-1]["resource_summary"][
            "training_control"
        ]["stopped_trials"]
        == 1
    )
    assert "eer" not in stopped.metrics
    assert result.summary["best_parameters"]["lr"] == 0.001
    assert len(list(experiments.rglob("*.evaluated"))) == 1
    before = {str(p): p.read_bytes() for p in experiments.rglob("fake.ckpt")}
    resumed = agent.run_workflow(
        AgentTaskRequest(
            action=agent.action,
            objective="resume",
            context={"resume_experiment_id": experiment},
        )
    )
    assert resumed.status == "success", resumed.to_dict()
    assert before == {
        str(p): p.read_bytes() for p in experiments.rglob("fake.ckpt")
    }
    assert (
        resumed.summary["hpo_feedback"]["evaluation_role"] == "hpo_validation"
    )


def _distributed_epoch_worker(rank, rendezvous, output):
    from types import SimpleNamespace
    import torch
    from agent.hpo.training_control import (
        initialize_training_control,
        observe_training_epoch,
    )
    from tests.unit.test_training_control import specification

    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2
    )
    try:
        brain = SimpleNamespace(
            device="cpu",
            hparams=SimpleNamespace(
                output_folder=output,
                epoch_counter=SimpleNamespace(current=0, limit=10),
                hpo_training_control=specification(min_epochs=2, patience=1),
            ),
        )
        initialize_training_control(brain)
        for epoch in (1, 2):
            observe_training_epoch(
                brain,
                epoch,
                {
                    "train_loss": 1.0 + 2 * rank,
                    "valid_loss": 1.0 + 2 * rank,
                    "valid_error_rate": 0.5,
                },
            )
        brain.hparams.epoch_counter.current = 2
        brain.hparams.epoch_counter.limit = 10
        initialize_training_control(brain)
        (Path(output) / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "limit": brain.hparams.epoch_counter.limit,
                    "decision": brain.hpo_training_decision,
                }
            )
        )
    finally:
        torch.distributed.destroy_process_group()


def test_distributed_epoch_decision_is_shared_across_ranks(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo backend unavailable")
    torch.multiprocessing.spawn(
        _distributed_epoch_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    first, second = [
        json.loads((tmp_path / f"rank{rank}.json").read_text())
        for rank in (0, 1)
    ]
    assert first == second
    assert first["limit"] == 2
    assert (
        first["decision"]["should_stop"] and first["decision"]["value"] == 2.0
    )
