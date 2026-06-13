from agent.hpo import HPOService, Objective, SearchParameter, SearchSpace, TrialBudget
from agent.utils.experiment_tracker import ExperimentTracker
import agent.hpo.service as hpo_service_module


def test_hpo_study_trial_lifecycle_without_training(
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
        SearchSpace([SearchParameter("lr", "float", low=1e-4, high=1e-2)]),
        [Objective("eer", "min")],
        [TrialBudget("small", epochs=1)],
        max_trials=2,
    )
    trials = service.suggest_trials(study, 2)
    service.record_trial(study, trials[0].trial_id, status="completed", metrics={"eer": 0.04})
    service.record_trial(study, trials[1].trial_id, status="completed", metrics={"eer": 0.03})
    loaded = service.load_study(experiment_id)

    assert len(trials) == 2
    assert loaded.best_trial_id == trials[1].trial_id
    assert service.best_metric_value(loaded) == 0.03
