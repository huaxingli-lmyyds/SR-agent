import pytest

pytest.importorskip("langgraph")

from agent.hpo import (
    HPOScheduler,
    HPOService,
    Objective,
    RetryPolicy,
    SearchParameter,
    SearchSpace,
    TrialBudget,
)
from agent.utils.experiment_tracker import ExperimentTracker
import agent.hpo.service as hpo_service_module


def test_scheduler_retries_recoverable_failure_and_completes(
    tmp_path,
    minimal_config,
    dataset_dir,
    monkeypatch,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
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
        [Objective("eer", "min")],
        [TrialBudget("small", epochs=1)],
        initial_trial_count=1,
        max_training_runs=2,
    )

    def executor(trial, attempt):
        if attempt == 1:
            service.record_trial(study, trial.trial_id, status="failed", stop_reason="timeout")
            return {"status": "failed", "error": "timeout"}
        return {"status": "success", "metrics": {"eer": 0.04}}

    result = HPOScheduler(
        service,
        executor,
        retry_policy=RetryPolicy(max_retries=1),
    ).run(study)

    assert result.study.status == "completed"
    assert result.trials[0].metrics["eer"] == 0.04
    assert result.trials[0].cost["attempts"] == 2
    assert service.remaining_training_runs(result.study) == 0


def test_scheduler_exits_when_promotion_limit_is_exhausted(
    tmp_path,
    minimal_config,
    dataset_dir,
    monkeypatch,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
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
        [Objective("eer", "min")],
        [TrialBudget("small", epochs=1), TrialBudget("large", epochs=2)],
        initial_trial_count=1,
        promotion_limits=[0],
        max_training_runs=2,
    )

    result = HPOScheduler(
        service,
        lambda trial, attempt: {"status": "success", "metrics": {"eer": 0.04}},
    ).run(study)

    assert result.study.status == "completed"
    assert [trial.status for trial in result.trials] == ["completed"]


def test_scheduler_advisor_failure_does_not_control_execution(
    tmp_path,
    minimal_config,
    dataset_dir,
    monkeypatch,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
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
        [Objective("eer", "min")],
        [TrialBudget("small", epochs=1)],
        max_training_runs=1,
    )

    def failing_advisor(current_study):
        raise RuntimeError("advisor unavailable")

    result = HPOScheduler(
        service,
        lambda trial, attempt: {"status": "success", "metrics": {"eer": 0.04}},
        strategy_advisor=failing_advisor,
    ).run(study)

    assert result.study.status == "completed"
    assert "advisor unavailable" in result.advice["advice_error"]
