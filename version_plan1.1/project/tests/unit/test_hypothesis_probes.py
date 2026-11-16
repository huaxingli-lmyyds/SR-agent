from copy import deepcopy

import pytest

from agent.hpo import (
    HPOService,
    Objective,
    SearchParameter,
    SearchSpace,
    StrategyProposal,
    TrialBudget,
)
from agent.hpo.hypothesis_probes import opportunity, performance_summary
from agent.utils import ExperimentTracker


@pytest.fixture
def probe_study(tmp_path, minimal_config, dataset_dir):
    tracker = ExperimentTracker(tmp_path / "experiments")
    eid = tracker.create_hpo_experiment(
        config_path=str(minimal_config), data_folder=str(dataset_dir)
    )
    service = HPOService(tracker)
    study = service.create_study(
        eid,
        SearchSpace(
            [
                SearchParameter("lr", "float", low=0.0005, high=0.003),
                SearchParameter("margin", "float", low=0.15, high=0.3),
                SearchParameter("weight_decay", "float", low=5e-7, high=2e-5),
            ]
        ),
        [Objective("eer", "min")],
        [TrialBudget("screen", epochs=1)],
        strategy="tpe",
        controller_mode="llm",
        max_trials=9,
        max_training_runs=9,
        initial_trial_count=9,
        candidate_batch_size=3,
        sampler_config={"n_startup_trials": 3},
        hypothesis_probes_enabled=True,
    )
    initial = service.suggest_trials(study, 3)
    for index, trial in enumerate(initial):
        service.record_trial(study, trial.trial_id, status="running")
        service.record_trial(
            study,
            trial.trial_id,
            status="completed",
            metrics={"eer": 0.1 + index * 0.01},
        )
    return service, study, initial


def proposal(initial, interaction=False):
    anchor = initial[0]
    base = anchor.parameters
    lr = (base["lr"] + 0.0005) / 2
    margin = (base["margin"] + 0.15) / 2
    changes = (
        [{"lr": lr}, {"margin": margin}, {"lr": lr, "margin": margin}]
        if interaction
        else [{"lr": lr}]
    )
    return StrategyProposal(
        action="switch_strategy",
        requested_sampler="agent_proposal",
        confidence=0.9,
        hypotheses=[
            {
                "id": "h_test",
                "claim": "A lower rate improves quality.",
                "anchor_trial_id": anchor.trial_id,
                "evidence_trial_ids": [anchor.trial_id],
                "probe_design": "interaction"
                if interaction
                else "single_factor",
                "tested_parameters": ["lr", "margin"]
                if interaction
                else ["lr"],
                "expected_signal": {"eer": "lower"},
                "falsification_criterion": {
                    "metric": "eer",
                    "comparison": "interaction_gain"
                    if interaction
                    else "improvement_over_anchor",
                    "minimum_improvement": 0.001,
                },
            }
        ],
        candidate_proposals=[
            {
                "parameters": {**base, **change},
                "hypothesis_id": "h_test",
                "role": "joint"
                if len(change) == 2
                else "single_factor_control",
                "rationale": "Test this perturbation against the observed anchor.",
                "expected_signal": {"eer": "lower"},
                "confidence": 0.9,
            }
            for change in changes
        ],
    )


def complete(service, study, trial, eer):
    service.record_trial(study, trial.trial_id, status="running")
    service.record_trial(
        study,
        trial.trial_id,
        status="completed",
        metrics={"eer": eer},
        cost={
            "attempt_training_gpu_seconds": 10,
            "attempt_evaluation_gpu_seconds": 1,
        },
    )


def test_low_sample_probe_replaces_candidates_and_returns_to_tpe(probe_study):
    service, study, initial = probe_study
    original_space = deepcopy(study.search_space.to_dict())
    review = service.review_strategy(study, proposal(initial))
    assert review["evidence_gate"]["targeted_probe_exception"]
    agent_trials = service.suggest_trials(study, 3)
    assert len(agent_trials) == 1
    assert agent_trials[0].candidate_source == "sampler:agent_proposal"
    complete(service, study, agent_trials[0], 0.08)
    assert not opportunity(study, service.list_trials(study.experiment_id))[
        "eligible"
    ]
    service.review_strategy(study, StrategyProposal(action="keep_strategy"))
    regular = service.suggest_trials(study, 3)
    assert all(t.candidate_source == "sampler:tpe" for t in regular)
    assert study.search_space.to_dict() == original_space
    assert study.max_training_runs == 9
    summary = performance_summary(
        study, service.list_trials(study.experiment_id)
    )
    assert summary["agent_candidate_count"] == 1
    assert summary["agent_candidate_gpu_seconds"] == 11
    assert summary["outcomes"][0]["status"] == "supported_observation"
    assert summary["outcomes"][0]["score"] == pytest.approx(0.02)
    assert not summary["outcomes"][0]["causal_claim"]


def test_interaction_is_a_complete_three_arm_design_and_survives_resume(
    probe_study,
):
    service, study, initial = probe_study
    service.review_strategy(study, proposal(initial, interaction=True))
    loaded = service.load_study(study.experiment_id)
    assert loaded.scheduler_state["hypothesis_probe_policy"]
    assert len(loaded.pending_candidate_proposals) == 3
    trials = service.suggest_trials(loaded, 3)
    by_changes = {
        frozenset(
            p
            for p in t.parameters
            if t.parameters[p] != initial[0].parameters[p]
        ): t
        for t in trials
    }
    complete(service, loaded, by_changes[frozenset(["lr"])], 0.09)
    complete(service, loaded, by_changes[frozenset(["margin"])], 0.095)
    complete(service, loaded, by_changes[frozenset(["lr", "margin"])], 0.07)
    outcome = performance_summary(
        loaded, service.list_trials(loaded.experiment_id)
    )["outcomes"][0]
    assert outcome["score"] == pytest.approx(0.015)
    assert outcome["status"] == "supported_observation"
    assert service.load_study(loaded.experiment_id).hypotheses[0]["result"][
        "score"
    ] == pytest.approx(0.015)
    assert (
        opportunity(loaded, service.list_trials(loaded.experiment_id))[
            "maximum_candidates"
        ]
        == 0
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_control",
        "wrong_control",
        "unknown_evidence",
        "wrong_fidelity",
        "unrelated_change",
        "no_falsification",
        "duplicate",
        "space_change",
        "too_many",
        "wrong_metric",
        "reused_id",
    ],
)
def test_invalid_design_is_rejected_atomically(probe_study, invalid):
    service, study, initial = probe_study
    p = proposal(initial, interaction=True)
    if invalid == "missing_control":
        p.candidate_proposals.pop()
    elif invalid == "wrong_control":
        p.candidate_proposals[-1]["parameters"]["lr"] *= 1.01
    elif invalid == "unknown_evidence":
        p.hypotheses[0]["evidence_trial_ids"] = ["invented"]
    elif invalid == "wrong_fidelity":
        t = service.load_trial(study.experiment_id, initial[0].trial_id)
        t.budget = TrialBudget("other", epochs=6)
        service.save_trial(study.experiment_id, t)
    elif invalid == "unrelated_change":
        p.candidate_proposals[0]["parameters"]["weight_decay"] = 1e-6
    elif invalid == "no_falsification":
        p.hypotheses[0].pop("falsification_criterion")
    elif invalid == "duplicate":
        p.candidate_proposals[0]["parameters"] = initial[0].parameters
    elif invalid == "space_change":
        p.search_space = study.search_space.to_dict()
    elif invalid == "too_many":
        p.candidate_proposals *= 2
    elif invalid == "wrong_metric":
        p.hypotheses[0]["falsification_criterion"]["metric"] = "test_eer"
    elif invalid == "reused_id":
        study.candidate_proposal_reviews.append(
            {"accepted": [{"candidate_id": "already_used"}]}
        )
        p.candidate_proposals[0]["candidate_id"] = "already_used"
    review = service.review_strategy(study, p)
    assert not study.pending_candidate_proposals
    assert not study.hypotheses
    assert study.candidate_strategy == "tpe"
    assert any(
        v["field"] == "hypothesis_probe"
        for v in review["decision"]["rejected_fields"]
    )


def test_failure_remains_inconclusive_and_consumes_probe_quota(probe_study):
    service, study, initial = probe_study
    service.review_strategy(study, proposal(initial, interaction=True))
    ts = service.suggest_trials(study, 3)
    service.record_trial(study, ts[0].trial_id, status="running")
    service.record_trial(
        study, ts[0].trial_id, status="failed", stop_reason="timeout"
    )
    for t in ts[1:]:
        complete(service, study, t, 0.08)
    summary = performance_summary(
        study, service.list_trials(study.experiment_id)
    )
    assert summary["outcomes"][0]["status"] == "inconclusive"
    assert (
        opportunity(study, service.list_trials(study.experiment_id))[
            "agent_candidates_used"
        ]
        == 3
    )


def test_no_improvement_rejects_hypothesis_without_changing_space(probe_study):
    service, study, initial = probe_study
    service.review_strategy(study, proposal(initial))
    t = service.suggest_trials(study, 1)[0]
    complete(service, study, t, 0.11)
    assert (
        performance_summary(study, service.list_trials(study.experiment_id))[
            "outcomes"
        ][0]["status"]
        == "not_supported"
    )


def test_boundary_probe_requires_an_actual_endpoint(probe_study):
    service, study, initial = probe_study
    p = proposal(initial)
    p.hypotheses[0]["probe_design"] = "boundary"
    service.review_strategy(study, p)
    assert not study.pending_candidate_proposals
    p = proposal(initial)
    p.hypotheses[0]["probe_design"] = "boundary"
    p.candidate_proposals[0]["parameters"]["lr"] = 0.0005
    service.review_strategy(study, p)
    assert len(study.pending_candidate_proposals) == 1


def test_consultation_quota_survives_reload_and_early_evidence_does_not_prune(
    probe_study,
):
    service, study, initial = probe_study
    for _ in range(2):
        service.review_strategy(study, StrategyProposal(action="keep_strategy"))
    loaded = service.load_study(study.experiment_id)
    status = opportunity(loaded, service.list_trials(loaded.experiment_id))
    assert status["reason"] == "consultation_limit_reached"
    service.review_strategy(loaded, proposal(initial))
    assert not loaded.pending_candidate_proposals
    assert loaded.search_space.to_dict() == study.search_space.to_dict()


def test_at_least_three_successful_initial_observations_required(probe_study):
    service, study, initial = probe_study
    trial = service.load_trial(study.experiment_id, initial[0].trial_id)
    trial.status = "failed"
    service.save_trial(study.experiment_id, trial)
    service.review_strategy(study, proposal(initial))
    assert not study.pending_candidate_proposals
    assert (
        opportunity(study, service.list_trials(study.experiment_id))["reason"]
        == "collect_initial_tpe_evidence"
    )


def test_maximize_objective_scores_positive_gains(probe_study):
    service, study, initial = probe_study
    study.objectives[0].mode = "max"
    service.review_strategy(study, proposal(initial))
    trial = service.suggest_trials(study, 1)[0]
    complete(service, study, trial, 0.12)
    outcome = performance_summary(
        study, service.list_trials(study.experiment_id)
    )["outcomes"][0]
    assert outcome["score"] == pytest.approx(0.02)
    assert outcome["status"] == "supported_observation"


def test_pending_candidates_prevent_premature_hypothesis_result(probe_study):
    service, study, initial = probe_study
    p = proposal(initial)
    second = deepcopy(p.candidate_proposals[0])
    second["parameters"]["lr"] = (second["parameters"]["lr"] + 0.0005) / 2
    p.candidate_proposals.append(second)
    service.review_strategy(study, p)
    trial = service.suggest_trials(study, 1)[0]
    complete(service, study, trial, 0.08)
    assert len(study.pending_candidate_proposals) == 1
    assert (
        performance_summary(study, service.list_trials(study.experiment_id))[
            "outcomes"
        ][0]["status"]
        == "inconclusive"
    )


def test_retries_reduce_capacity_for_complete_probe_design(probe_study):
    service, study, initial = probe_study
    trial = service.load_trial(study.experiment_id, initial[0].trial_id)
    trial.cost["retry_count"] = 5
    service.save_trial(study.experiment_id, trial)
    assert (
        opportunity(study, service.list_trials(study.experiment_id))[
            "maximum_candidates"
        ]
        == 1
    )
    service.review_strategy(study, proposal(initial, interaction=True))
    assert not study.pending_candidate_proposals
