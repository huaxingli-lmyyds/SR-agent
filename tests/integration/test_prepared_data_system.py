import csv
import json
from pathlib import Path

from agent.agents.communication import AgentTaskResult
from agent.agents.coordination import AgentRegistration
from agent.agents.orchestrator import CoordinatorAgent
from agent.runners import RUNNER_ADAPTERS
from tests.fakes import FakeRunnerAdapter
from tests.unit.test_audio_preparation import raw_audio


def test_raw_audio_to_hpo_and_registered_evaluation_agent(
    tmp_path, minimal_config, monkeypatch
):
    source = raw_audio(tmp_path / "raw")
    trained, evaluated = [], []

    class Runner(FakeRunnerAdapter):
        def run_training(self, config_path, overrides):
            assert overrides["skip_prep"] is True
            assert overrides["sample_rate"] == 16000
            with Path(overrides["train_annotation"]).open() as stream:
                rows = list(csv.DictReader(stream))
            assert overrides["out_n_neurons"] == len(
                {r["spk_id"] for r in rows}
            )
            assert not str(overrides["data_folder"]) == str(source)
            trained.append(overrides)
            return super().run_training(config_path, overrides)

    class Evaluation:
        def execute_task(self, request):
            previous = request.context["previous_results"]
            assert previous["hpo_agent"]["status"] == "success"
            prepared = request.context["data_handoff"]["prepared_data"]
            assert (
                request.context["hpo_feedback"]["protocol_sha256"]
                == prepared["protocol_sha256"]
            )
            assert (Path(prepared["root"]) / "final_test_pairs.txt").is_file()
            evaluated.append(request)
            return AgentTaskResult("success", request_id=request.request_id)

    monkeypatch.setitem(RUNNER_ADAPTERS, "prepared_fake", Runner())
    coordinator = CoordinatorAgent(
        config_path=str(minimal_config),
        output_dir=str(tmp_path / "run"),
        runner="prepared_fake",
        implementation="fake",
        verbose=False,
    )
    coordinator.register_agent(
        AgentRegistration(
            "evaluation_agent",
            ("evaluate",),
            Evaluation,
            requires=("hpo_agent",),
        )
    )
    result = coordinator.run(
        context={
            "data_folder": str(source),
            "data_processing": {"prepare": {"seed": 42}},
            "controller_mode": "fixed",
            "strategy": "random_search",
            "search_space": {
                "parameters": [
                    {
                        "name": "lr",
                        "parameter_type": "categorical",
                        "choices": [0.001],
                    }
                ]
            },
            "budgets": [{"stage": "full", "epochs": 1}],
            "runtime_options": {
                "verification_config": str(minimal_config),
                "device": "cpu",
            },
        },
        budget={"max_training_runs": 1, "max_retries": 0},
    )
    assert result.status == "success", [r for r in result.task_results]
    assert len(trained) == len(evaluated) == 1
    record = json.loads(
        next(
            (tmp_path / "run/hpo").glob("*/experiment_record.json")
        ).read_text()
    )
    runtime = record["execution"]["training_runtime_options"]
    assert runtime["prepared_data"]["root"] == trained[0]["data_folder"]
    assert "final_test_pairs" not in runtime
    experiment = next(
        (tmp_path / "run/hpo").glob("*/experiment_record.json")
    ).parent.name
    resumed = coordinator.run(context={"resume_experiment_id": experiment})
    assert resumed.status == "success", resumed.task_results
    assert len(trained) == 1 and len(evaluated) == 2
