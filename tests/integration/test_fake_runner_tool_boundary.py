import json

import pytest

pytest.importorskip("langchain_core")

from agent.runners import RUNNER_ADAPTERS, register_runner_adapter
from agent.hpo import HPOService, Objective, SearchParameter, SearchSpace, TrialBudget
from agent.tools import evaluation_tools, training_tools
from agent.utils.experiment_tracker import ExperimentTracker
from tests.fakes import FakeRunnerAdapter
import agent.hpo.service as hpo_service_module


def test_evaluation_tool_uses_registered_runner_and_records_result(
    tmp_path,
    minimal_config,
    dataset_dir,
    monkeypatch,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    checkpoint = tmp_path / "fake.ckpt"
    checkpoint.write_text("fake", encoding="utf-8")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
        task={"type": "speaker_verification", "dataset": str(dataset_dir)},
        model={"family": "ecapa_tdnn", "implementation": "fake"},
        execution={"runner": "fake"},
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
        payload = json.loads(evaluation_tools._run_evaluation(experiment_id=experiment_id))
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


def test_training_tool_applies_budget_and_synchronizes_trial(
    tmp_path,
    minimal_config,
    dataset_dir,
    monkeypatch,
) -> None:
    captured = {}
    processed_dataset = tmp_path / "processed_dataset"
    processed_dataset.mkdir()

    class CapturingRunner(FakeRunnerAdapter):
        def __init__(self):
            self.runner = "capturing"

        def run_training(self, config_path, overrides):
            captured.update(overrides)
            return super().run_training(config_path, overrides)

    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
        task={"type": "speaker_verification", "dataset": str(dataset_dir)},
        model={"family": "ecapa_tdnn", "implementation": "fake"},
        execution={"runner": "capturing"},
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
    previous = RUNNER_ADAPTERS.get("capturing")
    register_runner_adapter(CapturingRunner())
    monkeypatch.setattr(training_tools, "ExperimentTracker", lambda *args, **kwargs: tracker)
    monkeypatch.setattr(training_tools, "get_experiment_dir", lambda *args, **kwargs: tmp_path / "run")
    (tmp_path / "run").mkdir()

    try:
        payload = json.loads(training_tools.TrainModel.invoke({
            "experiment_id": experiment_id,
            "trial_id": trial.trial_id,
            "parameters_json": json.dumps(trial.parameters),
            "budget_json": json.dumps(trial.budget.to_dict()),
            "runner": "capturing",
            "data_folder": str(processed_dataset),
        }))
    finally:
        if previous is None:
            RUNNER_ADAPTERS.pop("capturing", None)
        else:
            RUNNER_ADAPTERS["capturing"] = previous

    recorded = service.load_trial(experiment_id, trial.trial_id)
    assert payload["status"] == "success"
    assert recorded.status == "completed"
    assert recorded.metrics["valid_error_rate"] == 0.08
    assert captured["number_of_epochs"] == 2
    assert captured["_hpo_data_fraction"] == 0.25
    assert captured["_hpo_max_duration_seconds"] == 10
    assert captured["data_folder"] == str(processed_dataset)
    assert payload["task"]["dataset"] == str(processed_dataset)
