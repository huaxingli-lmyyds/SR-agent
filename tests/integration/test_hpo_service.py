from agent.hpo import HPOService, Objective, SearchParameter, SearchSpace, StrategyProposal, Trial, TrialBudget
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
    service.record_trial(study, trials[0].trial_id, status="running")
    service.record_trial(study, trials[0].trial_id, status="completed", metrics={"eer": 0.04})
    service.record_trial(study, trials[1].trial_id, status="running")
    service.record_trial(study, trials[1].trial_id, status="completed", metrics={"eer": 0.03})
    loaded = service.load_study(experiment_id)

    assert len(trials) == 2
    assert loaded.best_trial_id == trials[1].trial_id
    assert service.best_metric_value(loaded) == 0.03


def test_hpo_quotas_promotion_and_strict_completion(
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
        SearchSpace([SearchParameter("lr", "categorical", choices=[0.1, 0.2, 0.3])]),
        [Objective("eer", "min")],
        [TrialBudget("small", epochs=1), TrialBudget("large", epochs=2)],
        initial_trial_count=3,
        promotion_limits=[1],
        max_training_runs=4,
        min_completed_per_rung=2,
    )
    trials = service.suggest_trials(study, 10)
    assert len(trials) == 3
    for index, trial in enumerate(trials):
        service.record_trial(study, trial.trial_id, status="running")
        service.record_trial(study, trial.trial_id, status="completed", metrics={"eer": 0.1 + index})

    promoted = service.promote_trials(study)
    assert len(promoted) == 1
    assert promoted[0].rung == 1
    assert service.remaining_training_runs(study) == 0

    try:
        service.complete_study(study)
        assert False, "active promoted trial must block completion"
    except ValueError as exc:
        assert "terminal status" in str(exc)

    service.record_trial(study, promoted[0].trial_id, status="running")
    service.record_trial(study, promoted[0].trial_id, status="completed", metrics={"eer": 0.05})
    assert service.complete_study(study).status == "completed"


def test_successive_halving_waits_for_startup_cohort_before_promotion(
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
        SearchSpace([SearchParameter("lr", "categorical", choices=[0.1, 0.2, 0.3])]),
        [Objective("eer", "min")],
        [TrialBudget("small", epochs=1), TrialBudget("large", epochs=2)],
        initial_trial_count=3,
        promotion_limits=[1],
        max_training_runs=4,
        min_completed_per_rung=1,
    )
    trials = service.suggest_trials(study, 10)
    service.record_trial(study, trials[0].trial_id, status="running")
    service.record_trial(study, trials[0].trial_id, status="completed", metrics={"eer": 0.1})

    assert service.promote_trials(study) == []

    for index, trial in enumerate(trials[1:], start=1):
        service.record_trial(study, trial.trial_id, status="running")
        service.record_trial(study, trial.trial_id, status="completed", metrics={"eer": 0.1 + index})

    assert len(service.promote_trials(study)) == 1

def test_grid_search_strategy_is_selected_by_service(
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
        SearchSpace([
            SearchParameter("model_family", "categorical", choices=["ecapa", "resnet"]),
            SearchParameter("batch_size", "categorical", choices=[16, 32]),
        ]),
        [Objective("eer", "min")],
        [TrialBudget("full", epochs=1)],
        strategy="grid_search",
        max_training_runs=4,
    )

    trials = service.suggest_trials(study, 10)

    assert len(trials) == 4
    assert len({tuple(sorted(trial.parameters.items())) for trial in trials}) == 4


def test_runtime_review_updates_only_future_candidate_generation(
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
        SearchSpace([SearchParameter("lr", "float", low=0.0, high=1.0)]),
        [Objective("eer", "min")],
        [TrialBudget("full", epochs=1)],
        strategy="random_search",
        max_training_runs=4,
    )
    trials = service.suggest_trials(study, 2)
    for index, trial in enumerate(trials):
        service.record_trial(study, trial.trial_id, status="running")
        service.record_trial(study, trial.trial_id, status="completed", metrics={"eer": 0.2 - index * 0.1})

    review = service.review_strategy(study, trigger="after_2_trials")

    assert study.strategy == "random_search"
    assert study.candidate_strategy == "adaptive_search"
    assert review["feedback"]["completed_trials"] == 2
    assert study.strategy_reviews

    blocked = service.review_strategy(study, StrategyProposal(
        action="adjust_budget",
        budgets=[{"stage": "unsafe", "epochs": 100}],
        max_training_runs=100,
        requested_strategy="successive_halving",
        requested_pruner="successive_halving",
        initial_trial_count=3,
        promotion_limits=[1],
        reduction_factor=4,
    ))
    fields = {item["field"] for item in blocked["decision"]["rejected_fields"]}
    assert fields == {
        "budgets",
        "max_training_runs",
        "requested_strategy",
        "requested_pruner",
        "initial_trial_count",
        "promotion_limits",
        "reduction_factor",
    }
    assert study.budgets[0].stage == "full"
    assert study.max_training_runs == 4


def test_training_runs_used_includes_retries(
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
        [TrialBudget("full", epochs=1)],
        max_training_runs=3,
    )
    trial = service.suggest_trials(study, 1)[0]
    service.record_trial(study, trial.trial_id, status="running")
    service.record_trial(study, trial.trial_id, status="failed", stop_reason="timeout")
    service.retry_trial(study, trial.trial_id, "timeout")

    assert service.training_runs_used(study) == 2
    assert service.remaining_training_runs(study) == 1


def test_record_trial_deduplicates_artifacts(
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
        [TrialBudget("full", epochs=1)],
        max_training_runs=1,
    )
    trial = service.suggest_trials(study, 1)[0]
    artifact = {"type": "predictions", "name": "scores", "path": "scores.txt"}

    service.record_trial(study, trial.trial_id, status="running", artifacts=[artifact])
    service.record_trial(study, trial.trial_id, status="completed", metrics={"eer": 0.03}, artifacts=[artifact])

    saved = service.load_trial(experiment_id, trial.trial_id)
    assert saved.artifacts == [artifact]

def test_warm_start_history_guides_new_study_sampler(
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
    warm_trial = Trial(
        "prior_trial",
        {"lr": 0.4},
        TrialBudget("screen", epochs=1),
        status="completed",
        metrics={"eer": 0.1},
    )
    study = service.create_study(
        experiment_id,
        SearchSpace([SearchParameter("lr", "float", low=0.0, high=1.0)]),
        [Objective("eer", "min")],
        [TrialBudget("screen", epochs=1), TrialBudget("confirm", epochs=2)],
        strategy="successive_halving",
        sampler_strategy="adaptive_search",
        pruner_strategy="successive_halving",
        initial_trial_count=2,
        promotion_limits=[1],
        max_training_runs=3,
        warm_start_trials=[warm_trial.to_dict()],
    )

    suggestions = service.suggest_trials(study, 2)

    assert study.sampler_strategy == "adaptive_search"
    assert study.pruner_strategy == "successive_halving"
    assert len(study.warm_start_trials) == 1
    assert len(suggestions) == 2
    assert all(trial.parameters["lr"] != 0.4 for trial in suggestions)
    assert all(abs(trial.parameters["lr"] - 0.4) <= 0.11 for trial in suggestions)
