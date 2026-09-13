import pytest

from agent.hpo import EvidenceGate, HPOService, Objective, SearchParameter, SearchSpace, StrategyProposal, Trial, TrialBudget
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


def test_prepare_resume_reopens_running_trial_without_new_candidate(
    tmp_path,
    minimal_config,
    dataset_dir,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
    )
    service = HPOService(tracker)
    study = service.create_study(
        experiment_id,
        SearchSpace([
            SearchParameter("lr", "categorical", choices=[0.1, 0.2]),
        ]),
        [Objective("eer", "min")],
        [TrialBudget("full", epochs=1)],
        strategy="grid_search",
        max_training_runs=2,
        candidate_batch_size=2,
    )
    trials = service.suggest_trials(study, 2)
    service.record_trial(study, trials[0].trial_id, status="running")
    interrupted_study = service.load_study(experiment_id)
    interrupted_study.trial_ids.remove(trials[1].trial_id)
    service.update_scheduler_state(interrupted_study, current_trial_id=trials[0].trial_id)

    recovered = service.prepare_resume(service.load_study(experiment_id))
    persisted = service.list_trials(experiment_id)

    assert recovered["recovered_trial_ids"] == [trials[0].trial_id]
    assert {trial.trial_id for trial in persisted} == {
        trial.trial_id for trial in trials
    }
    assert all(trial.status == "suggested" for trial in persisted)
    resumed_trial = service.load_trial(experiment_id, trials[0].trial_id)
    assert resumed_trial.cost["resume_count"] == 1
    assert resumed_trial.provenance["resume_events"][0]["reason"] == (
        "owning_process_interrupted"
    )
    persisted_study = service.load_study(experiment_id)
    assert set(persisted_study.trial_ids) == {trial.trial_id for trial in trials}
    assert persisted_study.scheduler_state["resume_count"] == 1
    assert recovered["reconciled_trial_ids"] == [trials[1].trial_id]


def test_agent_proposal_candidates_are_validated_and_audited_as_one_strategy(
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
    validated = []
    service = HPOService(tracker, parameter_validator=lambda value: validated.append(value))
    study = service.create_study(
        experiment_id,
        SearchSpace([SearchParameter("lr", "float", low=0.0, high=1.0)]),
        [Objective("eer", "min")],
        [TrialBudget("full", epochs=1)],
        strategy="agent_proposal",
        max_training_runs=4,
        hypotheses=[{
            "id": "h_local",
            "claim": "The middle of the range is a useful controlled probe.",
            "evidence_trial_ids": [],
            "expected_signal": {"eer": "lower"},
        }],
        candidate_proposals=[
            {
                "candidate_id": "valid",
                "parameters": {"lr": 0.25},
                "hypothesis_id": "h_local",
                "confidence": 0.8,
            },
            {
                "candidate_id": "out_of_range",
                "parameters": {"lr": 2.0},
                "hypothesis_id": "h_local",
            },
        ],
        proposal_id="proposal_planning",
    )

    review = next(
        item for item in study.candidate_proposal_reviews
        if item["trigger"] == "study_planning"
    )
    trials = service.suggest_trials(study, 2)

    assert len(review["accepted"]) == 1
    assert len(review["rejected"]) == 1
    assert validated == [{"lr": 0.25}, {"lr": 0.25}]
    assert len(trials) == 1
    agent_trial = trials[0]
    assert agent_trial.candidate_source == "sampler:agent_proposal"
    assert agent_trial.parameters == {"lr": 0.25}
    assert agent_trial.proposal_id == "proposal_planning"
    assert agent_trial.hypothesis_id == "h_local"
    assert agent_trial.provenance["sampler"] == "agent_proposal"


def test_candidate_validation_is_independent_of_conditional_field_order(
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
            SearchParameter("momentum", "float", low=0.0, high=1.0, condition={"optimizer": "sgd"}),
            SearchParameter("optimizer", "categorical", choices=["sgd", "adam"]),
        ]),
        [Objective("eer", "min")],
        [TrialBudget("full", epochs=1)],
        max_training_runs=1,
    )

    assert service.validate_candidate_parameters(
        study,
        {"optimizer": "sgd", "momentum": 0.9},
    ) == {"momentum": 0.9, "optimizer": "sgd"}
    with __import__("pytest").raises(ValueError, match="inactive conditional parameter"):
        service.validate_candidate_parameters(
            study,
            {"optimizer": "adam", "momentum": 0.9},
        )


def test_runtime_candidate_is_rejected_after_generation_capacity_is_exhausted(
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
        max_training_runs=1,
    )
    trial = service.suggest_trials(study, 1)[0]
    service.record_trial(study, trial.trial_id, status="running")
    service.record_trial(study, trial.trial_id, status="completed", metrics={"eer": 0.1})

    review = service.review_strategy(study, StrategyProposal(
        action="switch_strategy",
        requested_sampler="agent_proposal",
        candidate_proposals=[{"parameters": {"lr": 0.75}}],
    ))

    assert review["candidate_review"]["accepted"] == []
    assert "no future candidate-generation capacity" in review[
        "candidate_review"
    ]["rejected"][0]["reason"]
    assert study.pending_candidate_proposals == []


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
    service.record_trial(study, promoted[0].trial_id, status="completed", metrics={"eer": 0.5})
    completed_study = service.complete_study(study)
    assert completed_study.status == "completed"
    assert completed_study.best_trial_id == promoted[0].trial_id


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
    assert fields == {"proposal"}
    assert blocked["controller_mode"] == "rule"
    assert study.budgets[0].stage == "full"
    assert study.max_training_runs == 4


def test_llm_runtime_review_cannot_change_frozen_allocation(
    tmp_path, minimal_config, dataset_dir, monkeypatch,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config), data_folder=str(dataset_dir),
    )
    monkeypatch.setattr(
        hpo_service_module, "get_experiment_artifact_dir",
        lambda *args, **kwargs: tmp_path / "hpo_artifacts",
    )
    service = HPOService(
        tracker, evidence_gate=EvidenceGate(min_same_fidelity_trials=1)
    )
    study = service.create_study(
        experiment_id,
        SearchSpace([SearchParameter("lr", "float", low=0.0, high=1.0)]),
        [Objective("eer", "min")], [TrialBudget("full", epochs=1)],
        strategy="random_search", max_training_runs=3, controller_mode="llm",
    )
    trial = service.suggest_trials(study, 1)[0]
    service.record_trial(study, trial.trial_id, status="running")
    service.record_trial(
        study, trial.trial_id, status="completed", metrics={"eer": 0.2}
    )

    review = service.review_strategy(study, StrategyProposal(
        action="adjust_budget", confidence=0.9,
        budgets=[{"stage": "unsafe", "epochs": 100}],
        max_training_runs=100,
        requested_strategy="successive_halving",
        requested_pruner="successive_halving",
        initial_trial_count=3,
        promotion_limits=[1],
        reduction_factor=4,
    ))
    fields = {item["field"] for item in review["decision"]["rejected_fields"]}
    assert fields == {
        "budgets", "max_training_runs", "requested_strategy",
        "requested_pruner", "initial_trial_count", "promotion_limits",
        "reduction_factor",
    }
    assert study.budgets == [TrialBudget("full", epochs=1)]
    assert study.max_training_runs == 3


def test_fixed_controller_never_applies_runtime_proposals(
    tmp_path, minimal_config, dataset_dir, monkeypatch,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config), data_folder=str(dataset_dir),
    )
    monkeypatch.setattr(
        hpo_service_module, "get_experiment_artifact_dir",
        lambda *args, **kwargs: tmp_path / "hpo_artifacts",
    )
    service = HPOService(tracker)
    study = service.create_study(
        experiment_id,
        SearchSpace([SearchParameter("lr", "float", low=0.0, high=1.0)]),
        [Objective("eer", "min")], [TrialBudget("full", epochs=1)],
        strategy="random_search", max_training_runs=2, controller_mode="fixed",
    )

    review = service.review_strategy(study, StrategyProposal(
        action="switch_strategy", requested_sampler="adaptive_search",
        confidence=1.0,
    ))

    assert review["controller_mode"] == "fixed"
    assert review["applied"] is False
    assert study.candidate_strategy == "random_search"
    assert {item["field"] for item in review["decision"]["rejected_fields"]} == {
        "proposal"
    }


def test_llm_planning_candidate_requires_evidence_confidence(
    tmp_path, minimal_config, dataset_dir, monkeypatch,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config), data_folder=str(dataset_dir),
    )
    monkeypatch.setattr(
        hpo_service_module, "get_experiment_artifact_dir",
        lambda *args, **kwargs: tmp_path / "hpo_artifacts",
    )
    study = HPOService(tracker).create_study(
        experiment_id,
        SearchSpace([SearchParameter("lr", "float", low=0.0, high=1.0)]),
        [Objective("eer", "min")], [TrialBudget("full", epochs=1)],
        strategy="agent_proposal", fallback_sampler_strategy="random_search",
        max_training_runs=2, controller_mode="llm",
        candidate_proposals=[{"parameters": {"lr": 0.5}, "confidence": 0.4}],
    )

    assert study.candidate_strategy == "random_search"
    planning = next(
        item for item in study.candidate_proposal_reviews
        if item["trigger"] == "study_planning"
    )
    assert planning["accepted"] == []
    assert "confidence must be at least" in planning["rejected"][0]["reason"]


def test_applied_decision_is_attributed_only_to_affected_trials(
    tmp_path, minimal_config, dataset_dir, monkeypatch,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config), data_folder=str(dataset_dir),
    )
    monkeypatch.setattr(
        hpo_service_module, "get_experiment_artifact_dir",
        lambda *args, **kwargs: tmp_path / "hpo_artifacts",
    )
    service = HPOService(
        tracker, evidence_gate=EvidenceGate(min_same_fidelity_trials=1)
    )
    study = service.create_study(
        experiment_id,
        SearchSpace([SearchParameter("lr", "float", low=0.0, high=1.0)]),
        [Objective("eer", "min")], [TrialBudget("full", epochs=1)],
        strategy="random_search", max_training_runs=2, controller_mode="llm",
    )
    before = service.suggest_trials(study, 1)[0]
    service.record_trial(study, before.trial_id, status="running")
    service.record_trial(
        study, before.trial_id, status="completed", metrics={"eer": 0.2}
    )
    review = service.review_strategy(study, StrategyProposal(
        action="switch_strategy", requested_sampler="adaptive_search",
        confidence=0.9,
    ))
    after = service.suggest_trials(study, 1)[0]
    service.record_trial(study, after.trial_id, status="running")
    service.record_trial(
        study, after.trial_id, status="completed", metrics={"eer": 0.1},
        cost={"training": {"duration_seconds": 10.0}},
    )

    decision_id = review["decision"]["decision_id"]
    loaded = service.load_trial(study.experiment_id, after.trial_id)
    assert loaded.provenance["decision_id"] == decision_id
    saved_review = study.strategy_reviews[-1]
    assert saved_review["effective_from_trial_index"] == 1
    assert saved_review["affected_trial_ids"] == [after.trial_id]
    assert saved_review["realized_outcome"] == {
        "affected_trial_count": 1,
        "completed_trial_count": 1,
        "highest_completed_rung": 0,
        "best_primary_metric": 0.1,
        "average_primary_metric": 0.1,
        "training_seconds": 10.0,
        "evaluation_seconds": 0.0,
    }
    assert saved_review["effect_estimate"]["causal_claim"] is False
    assert saved_review["effect_estimate"]["status"] == "available"
    matched = saved_review["effect_estimate"]["comparisons"][0]
    assert matched["reference_trial_ids"] == [before.trial_id]
    assert matched["objective_aligned_mean_lift"] == 0.1


def test_study_rejects_non_finite_training_deadline(
    tmp_path, minimal_config, dataset_dir,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config), data_folder=str(dataset_dir),
    )
    with pytest.raises(ValueError, match="finite positive"):
        HPOService(tracker).create_study(
            experiment_id,
            SearchSpace([SearchParameter("lr", "float", low=0.0, high=1.0)]),
            [Objective("eer", "min")],
            [TrialBudget("full", epochs=1, max_duration_seconds=float("nan"))],
            strategy="random_search", max_training_runs=1,
        )


def test_sampler_exhaustion_defers_later_advice_to_next_study(
    tmp_path, minimal_config, dataset_dir, monkeypatch,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config), data_folder=str(dataset_dir),
    )
    monkeypatch.setattr(
        hpo_service_module, "get_experiment_artifact_dir",
        lambda *args, **kwargs: tmp_path / "hpo_artifacts",
    )
    service = HPOService(tracker)
    study = service.create_study(
        experiment_id,
        SearchSpace([SearchParameter("lr", "categorical", choices=[0.1])]),
        [Objective("eer", "min")], [TrialBudget("full", epochs=1)],
        strategy="random_search", max_training_runs=2, controller_mode="llm",
    )
    trial = service.suggest_trials(study, 1)[0]
    service.record_trial(study, trial.trial_id, status="running")
    service.record_trial(
        study, trial.trial_id, status="completed", metrics={"eer": 0.2}
    )

    assert service.suggest_trials(study, 1) == []
    assert service.has_future_candidate_capacity(study) is False
    original_space = study.search_space.to_dict()
    review = service.review_strategy(study, StrategyProposal(
        action="expand_search_space", confidence=0.9,
        search_space=SearchSpace([
            SearchParameter("lr", "categorical", choices=[0.1, 0.2]),
        ]).to_dict(),
    ))

    assert review["scope"] == "next_study"
    assert review["applied"] is False
    assert study.search_space.to_dict() == original_space
    assert study.next_study_proposal is not None


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
        provenance={"history_context": service._build_history_context(experiment_id, Objective("eer", "min"))},
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
