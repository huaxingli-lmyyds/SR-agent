from dataclasses import asdict
import json

import pytest

from agent.hpo import (
    HPOService,
    HPOScheduler,
    Objective,
    SearchParameter,
    SearchSpace,
    TrialBudget,
)
from agent.hpo.training_control import (
    EpochTrainingController,
    TrainingControlOptions,
    build_trial_control,
)
from agent.utils import ExperimentTracker


def specification(**options):
    return {
        "schema_version": 1,
        "policy": asdict(TrainingControlOptions(**options)),
        "budget": {"epochs": 10},
        "objective": {"metric": "eer", "mode": "min"},
        "peer_thresholds": {},
        "peer_trial_ids": [],
    }


def test_epoch_control_warmup_plateau_recovery_and_budget_completion(tmp_path):
    spec = specification(min_epochs=3, patience=2)
    path = tmp_path / "control.json"
    controller = EpochTrainingController(spec, path)
    for epoch, loss in enumerate([1.0, 0.9, 0.8, 0.8], 1):
        assert not controller.observe(epoch, {"valid_loss": loss})[
            "should_stop"
        ]
    restored = EpochTrainingController(spec, path)
    decision = restored.observe(5, {"valid_loss": 0.8})
    assert (
        decision["should_stop"] and decision["reason"] == "validation_plateau"
    )
    assert len(json.loads(path.read_text())["history"]) == 5
    spec["budget"]["epochs"] = 3
    finished = EpochTrainingController(spec)
    for epoch in range(1, 4):
        decision = finished.observe(epoch, {"valid_loss": 1})
    assert (
        decision["reason"] == "budget_completed" and not decision["should_stop"]
    )


def test_nonfinite_and_peer_evidence_decisions_do_not_use_primary_or_test_metrics():
    spec = specification(min_epochs=2, patience=8, peer_patience=2)
    spec["peer_thresholds"] = {"2": 1.5, "3": 1.5}
    controller = EpochTrainingController(spec)
    for epoch in range(1, 4):
        decision = controller.observe(
            epoch, {"valid_loss": 2.0, "eer": 0.01, "test_eer": 0}
        )
    assert decision["reason"] == "uncompetitive_same_budget_peers"
    assert (
        "eer" not in controller.history[0]
        and "test_eer" not in controller.history[0]
    )
    assert EpochTrainingController(spec).observe(
        1, {"valid_loss": float("nan")}
    )["should_stop"]


@pytest.mark.parametrize(
    "options",
    [
        {"metric": "eer"},
        {"metric": "test_loss"},
        {"min_epochs": True},
        {"patience": 0},
        {"peer_factor": float("nan")},
        {"unknown": 1},
    ],
)
def test_invalid_control_options_are_rejected(options):
    with pytest.raises(ValueError):
        TrainingControlOptions.from_dict(options)


def test_hpo_freezes_peer_evidence_and_stopped_trials_never_become_winners(
    tmp_path, minimal_config, dataset_dir
):
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config), data_folder=str(dataset_dir)
    )
    service = HPOService(tracker)
    study = service.create_study(
        experiment_id,
        SearchSpace(
            [SearchParameter("lr", "categorical", choices=[0.1, 0.2, 0.3, 0.4])]
        ),
        [Objective("eer", "min")],
        [TrialBudget("full", epochs=10)],
        strategy="grid_search",
        max_training_runs=4,
        candidate_batch_size=1,
    )
    stopped = []

    def execute(trial, attempt):
        spec = build_trial_control(
            service,
            service.load_study(experiment_id),
            trial,
            {"peer_min_trials": 2},
        )
        if spec["peer_thresholds"]:
            assert spec["peer_thresholds"]["3"] == pytest.approx(1.001)
            assert len(spec["peer_trial_ids"]) >= 2
            stopped.append(trial.trial_id)
            return {
                "status": "stopped",
                "stop_reason": "uncompetitive_same_budget_peers",
                "cost": {"training_control": {"should_stop": True}},
                "metrics": {"eer": -1},
            }
        return {
            "status": "success",
            "metrics": {"eer": 0.1},
            "intermediate_metrics": [{"epoch": 3, "valid_loss": 0.5}],
        }

    result = HPOScheduler(service, execute).run(study)
    assert result.study.status == "completed"
    assert len(stopped) == 2
    assert result.study.best_trial_id not in stopped
    assert not result.errors
    for trial in result.trials:
        if trial.trial_id in stopped:
            assert trial.status == "stopped" and "eer" not in trial.metrics
            assert trial.cost["attempts"] == 1
