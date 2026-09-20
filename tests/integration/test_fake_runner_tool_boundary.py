import json
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from agent.runners import RUNNER_ADAPTERS, register_runner_adapter
from agent.hpo import HPOService, Objective, SearchParameter, SearchSpace, TrialBudget
from agent.tools import evaluation_tools, training_tools
from agent.utils.experiment_tracker import ExperimentTracker
from tests.fakes import FakeRunnerAdapter
import agent.hpo.service as hpo_service_module
from agent.hpo.protocol import resolve_hpo_validation_protocol


def _validation_protocol(tmp_path, config_path):
    validation_config = tmp_path / "validation.yaml"
    validation_config.write_text(
        Path(config_path).read_text(encoding="utf-8"), encoding="utf-8"
    )
    pairs = tmp_path / "validation_pairs.txt"
    pairs.write_text(
        "1 id00001/session/a.wav id00001/session/b.wav\n"
        "0 id00001/session/a.wav id00002/session/c.wav\n",
        encoding="utf-8",
    )
    runtime = resolve_hpo_validation_protocol(
        {
            "verification_config": str(validation_config),
            "validation_pairs": str(pairs),
            "training_exclusion_pairs": str(pairs),
        },
        require_explicit=True,
    )
    execution = {
        "evaluation_config_path": runtime["verification_config"],
        "validation_pairs_path": runtime["validation_pairs"],
        "training_exclusion_pairs_path": runtime["training_exclusion_pairs"],
        **{
            key: runtime[key]
            for key in (
                "verification_config_sha256",
                "validation_pairs_sha256",
                "training_exclusion_pairs_sha256",
                "metric_protocol",
            )
        },
    }
    return runtime, execution


def test_evaluation_tool_uses_registered_runner_and_records_result(
    tmp_path,
    minimal_config,
    dataset_dir,
    monkeypatch,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    protocol, protocol_execution = _validation_protocol(tmp_path, minimal_config)
    checkpoint = tmp_path / "fake.ckpt"
    checkpoint.write_text("fake", encoding="utf-8")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
        task={"type": "speaker_verification", "dataset": str(dataset_dir)},
        model={"family": "ecapa_tdnn", "implementation": "fake"},
        execution={"runner": "fake", **protocol_execution},
    )
    tracker.update_hpo_experiment(
        experiment_id,
        artifacts=[{"type": "checkpoint", "name": "best", "path": str(checkpoint)}],
    )
    previous = RUNNER_ADAPTERS.get("fake")
    register_runner_adapter(FakeRunnerAdapter())
    monkeypatch.setattr(evaluation_tools, "ExperimentTracker", lambda *args, **kwargs: tracker)
    monkeypatch.setattr(
        evaluation_tools,
        "get_experiment_artifact_dir",
        lambda *args, **kwargs: tmp_path / "evaluation",
    )
    (tmp_path / "evaluation").mkdir()

    try:
        payload = json.loads(evaluation_tools._run_evaluation(
            experiment_id=experiment_id,
            verification_config=protocol["verification_config"],
            verification_pairs=protocol["validation_pairs"],
            evaluation_split="test",
        ))
    finally:
        if previous is None:
            RUNNER_ADAPTERS.pop("fake", None)
        else:
            RUNNER_ADAPTERS["fake"] = previous

    record = tracker.get_experiment(experiment_id)
    assert payload["status"] == "success"
    assert payload["metrics"]["test"]["eer"] == 0.03
    assert record["metrics"]["test"]["min_dcf"] == 0.12


def test_checkpoint_selection_is_scoped_to_trial() -> None:
    record = {
        "artifacts": [
            {"type": "checkpoint", "path": "trials/trial_a/output/a.ckpt", "metadata": {"trial_id": "trial_a"}},
            {"type": "checkpoint", "path": "trials/trial_b/output/b.ckpt", "metadata": {"trial_id": "trial_b"}},
        ]
    }
    assert evaluation_tools._checkpoint_path(record, "trial_b").endswith("b.ckpt")
    assert evaluation_tools._checkpoint_path(record, "missing") is None


@pytest.mark.parametrize("explicit_config", [False, True])
def test_training_tool_applies_budget_and_synchronizes_trial(
    tmp_path,
    minimal_config,
    dataset_dir,
    monkeypatch,
    explicit_config,
) -> None:
    captured = {}
    processed_dataset = tmp_path / "processed_dataset"
    processed_dataset.mkdir()

    class CapturingRunner(FakeRunnerAdapter):
        def __init__(self):
            self.runner = "capturing"

        def run_training(self, config_path, overrides):
            captured.update(overrides)
            captured["config_path"] = config_path
            captured["config_content"] = Path(config_path).read_text(encoding="utf-8")
            return super().run_training(config_path, overrides)

    tracker = ExperimentTracker(tmp_path / "experiments")
    protocol, protocol_execution = _validation_protocol(tmp_path, minimal_config)
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
        task={"type": "speaker_verification", "dataset": str(dataset_dir)},
        model={"family": "ecapa_tdnn", "implementation": "fake"},
        execution={"runner": "capturing", **protocol_execution},
    )
    monkeypatch.setattr(
        hpo_service_module,
        "get_experiment_artifact_dir",
        lambda *args, **kwargs: tmp_path / "hpo_artifacts",
    )
    service = HPOService(tracker)
    study = service.create_study(
        experiment_id,
        SearchSpace([SearchParameter("lr", "categorical", choices=[0.1])]),
        [Objective("valid_error_rate", "min")],
        [TrialBudget("small", epochs=2, data_fraction=0.25, max_duration_seconds=10)],
        initial_trial_count=1,
        max_training_runs=1,
    )
    trial = service.suggest_trials(study, 1)[0]
    snapshot = tracker.get_config_snapshot(experiment_id)
    original_config = snapshot.read_text(encoding="utf-8")
    minimal_config.write_text("modified_config: true\n", encoding="utf-8")
    previous = RUNNER_ADAPTERS.get("capturing")
    register_runner_adapter(CapturingRunner())
    monkeypatch.setattr(training_tools, "ExperimentTracker", lambda *args, **kwargs: tracker)
    monkeypatch.setattr(training_tools, "get_experiment_dir", lambda *args, **kwargs: tmp_path / "run")
    (tmp_path / "run").mkdir()

    try:
        payload = json.loads(training_tools.TrainModel.invoke({
            "config_path": str(minimal_config) if explicit_config else None,
            "experiment_id": experiment_id,
            "trial_id": trial.trial_id,
            "parameters_json": json.dumps(trial.parameters),
            "budget_json": json.dumps(trial.budget.to_dict()),
            "training_exclusion_pairs": protocol["training_exclusion_pairs"],
            "runner": "capturing",
            "data_folder": str(processed_dataset),
            "device": "cuda:0",
            "precision": "fp32",
            "eval_precision": "fp32",
        }))
    finally:
        if previous is None:
            RUNNER_ADAPTERS.pop("capturing", None)
        else:
            RUNNER_ADAPTERS["capturing"] = previous

    recorded = service.load_trial(experiment_id, trial.trial_id)
    assert payload["status"] == "success"
    assert recorded.status == "running"
    assert recorded.metrics["valid_error_rate"] == 0.08
    assert recorded.cost["training"]["status"] == "success"
    assert captured["number_of_epochs"] == 2
    assert Path(captured["config_path"]) == snapshot
    assert captured["config_content"] == original_config
    assert captured["_hpo_data_fraction"] == 0.25
    assert captured["_hpo_max_duration_seconds"] == 10
    assert captured["data_folder"] == str(processed_dataset)
    assert captured["verification_file"] == protocol["training_exclusion_pairs"]
    assert captured["_run_opts"] == {"device": "cuda:0", "precision": "fp32", "eval_precision": "fp32"}
    assert payload["task"]["dataset"] == str(processed_dataset)
    assert payload["task"]["primary_metric"] == "valid_error_rate"
    assert payload["task"]["metric_mode"] == "min"
    record = tracker.get_experiment(experiment_id)
    assert record["task"]["primary_metric"] == "valid_error_rate"
    assert record["stage"] == "optimization"
    assert record["execution"].get("trial_id") is None
    assert record["parameters"] == {}
    summary = (record["extensions"]["optimization"]["trial_summary"])[0]
    assert summary["phase"] == "evaluation_pending"
    assert summary["artifacts"][0]["metadata"]["trial_id"] == trial.trial_id

def test_checkpoint_selection_reads_trial_summary_artifacts() -> None:
    record = {
        "extensions": {
            "optimization": {
                "trial_summary": [
                    {
                        "trial_id": "trial_a",
                        "artifacts": [
                            {
                                "type": "checkpoint",
                                "path": "trials/trial_a/output/a.ckpt",
                                "metadata": {"trial_id": "trial_a"},
                            }
                        ],
                    }
                ]
            }
        }
    }

    assert evaluation_tools._checkpoint_path(record, "trial_a").endswith("a.ckpt")


def test_study_objective_remains_authoritative_during_trial_evaluation(
    tmp_path, minimal_config, dataset_dir, monkeypatch
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    protocol, protocol_execution = _validation_protocol(tmp_path, minimal_config)
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
        task={
            "type": "speaker_verification",
            "dataset": str(dataset_dir),
            "primary_metric": "min_dcf",
            "metric_mode": "min",
            "metric_protocol": protocol["metric_protocol"],
            "metric_units": protocol["metric_units"],
        },
        model={"family": "ecapa_tdnn", "implementation": "fake"},
        execution={"runner": "fake", **protocol_execution},
    )
    service = HPOService(tracker)
    study = service.create_study(
        experiment_id,
        SearchSpace([SearchParameter("lr", "categorical", choices=[0.1])]),
        [Objective("min_dcf", "min")],
        [TrialBudget("full", epochs=1)],
        initial_trial_count=1,
        max_training_runs=1,
    )
    trial = service.suggest_trials(study, 1)[0]
    service.record_trial(study, trial.trial_id, status="running")
    checkpoint = tmp_path / "fake.ckpt"
    checkpoint.write_text("fake", encoding="utf-8")
    runner = FakeRunnerAdapter()
    previous = RUNNER_ADAPTERS.get("fake")
    register_runner_adapter(runner)
    monkeypatch.setattr(evaluation_tools, "ExperimentTracker", lambda *_a, **_k: tracker)
    monkeypatch.setattr(
        evaluation_tools,
        "get_experiment_artifact_dir",
        lambda *_a, **_k: tmp_path / "evaluation",
    )
    try:
        payload = json.loads(evaluation_tools.RunEvaluation.invoke({
            "experiment_id": experiment_id,
            "trial_id": trial.trial_id,
            "model_path": str(checkpoint),
            "runner": "fake",
            "implementation": "fake",
        }))
    finally:
        if previous is None:
            RUNNER_ADAPTERS.pop("fake", None)
        else:
            RUNNER_ADAPTERS["fake"] = previous

    assert payload["status"] == "success"
    assert payload["metrics"] == {"validation": {"eer": 0.03, "min_dcf": 0.12}}
    assert payload["task"]["primary_metric"] == "min_dcf"
    assert payload["task"]["metric_mode"] == "min"
    assert payload["execution"]["evaluation_split"] == "validation"
    assert service.load_trial(experiment_id, trial.trial_id).metrics["min_dcf"] == 0.12
    record = tracker.get_experiment(experiment_id)
    assert record["task"]["primary_metric"] == "min_dcf"
    assert record["task"]["metric_protocol"] == protocol["metric_protocol"]
