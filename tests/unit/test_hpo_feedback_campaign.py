from agent.hpo import (
    CampaignPolicy,
    HPOFeedbackAnalyzer,
    HPOStudy,
    Objective,
    OptimizationCampaign,
    SearchParameter,
    SearchSpace,
    Trial,
    TrialBudget,
)


def _study() -> HPOStudy:
    return HPOStudy(
        study_id="study_1",
        experiment_id="exp_1",
        strategy="random_search",
        search_space=SearchSpace([SearchParameter("lr", "float", low=0.0, high=1.0)]),
        objectives=[Objective("eer", "min")],
        budgets=[TrialBudget("full", epochs=1)],
        max_training_runs=4,
    )


def test_feedback_clusters_failures_and_detects_search_boundary() -> None:
    study = _study()
    trials = [
        Trial("a", {"lr": 0.01}, study.budgets[0], status="completed", metrics={"eer": 0.1}),
        Trial("b", {"lr": 0.5}, study.budgets[0], status="failed", cost={"failure_category": "resource"}),
        Trial("c", {"lr": 0.6}, study.budgets[0], status="failed", cost={"failure_category": "resource"}),
    ]

    feedback = HPOFeedbackAnalyzer().analyze(study, trials)
    proposal = HPOFeedbackAnalyzer().propose(study, feedback)

    assert feedback["failure_clusters"] == {"resource": 2}
    assert feedback["boundary_hits"][0]["parameter"] == "lr"
    assert feedback["best_parameters"] == {"lr": 0.01}
    assert feedback["ranked_trials"][0]["trial_id"] == "a"
    assert feedback["ranked_trials"][0]["primary_metric"] == 0.1
    assert proposal.requested_strategy == "random_search"
    assert proposal.search_space["parameters"][0]["low"] < 0.0


def test_feedback_expands_log_boundaries_locally() -> None:
    study = HPOStudy(
        study_id="study_log",
        experiment_id="exp_log",
        strategy="random_search",
        search_space=SearchSpace([
            SearchParameter("weight_decay", "float", low=5e-7, high=2e-5, scale="log"),
        ]),
        objectives=[Objective("eer", "min")],
        budgets=[TrialBudget("full", epochs=1)],
        max_training_runs=4,
    )
    trials = [
        Trial(
            "a",
            {"weight_decay": 5e-7},
            study.budgets[0],
            status="completed",
            metrics={"eer": 0.1},
        )
    ]

    feedback = HPOFeedbackAnalyzer().analyze(study, trials)
    proposal = HPOFeedbackAnalyzer().propose(study, feedback)
    weight_decay = proposal.search_space["parameters"][0]

    assert feedback["boundary_hits"] == [{"parameter": "weight_decay", "edge": "low", "value": 5e-7}]
    assert weight_decay["low"] == 5e-7 / 3.0
    assert weight_decay["low"] > 1e-12

def test_feedback_uses_adaptive_generation_when_history_is_available() -> None:
    study = _study()
    trials = [
        Trial("a", {"lr": 0.2}, study.budgets[0], status="completed", metrics={"eer": 0.2}),
        Trial("b", {"lr": 0.4}, study.budgets[0], status="completed", metrics={"eer": 0.1}),
    ]

    proposal = HPOFeedbackAnalyzer().propose(study, HPOFeedbackAnalyzer().analyze(study, trials))

    assert proposal.requested_strategy == "adaptive_search"


def test_feedback_preserves_tpe_and_does_not_select_it_when_unavailable() -> None:
    study = _study()
    trials = [
        Trial(
            f"trial_{index}",
            {"lr": 0.2 + index * 0.1},
            study.budgets[0],
            status="completed",
            metrics={"eer": 0.2 - index * 0.01},
        )
        for index in range(5)
    ]
    analyzer = HPOFeedbackAnalyzer()
    feedback = analyzer.analyze(study, trials)

    assert analyzer.propose(study, feedback, ["adaptive_search"]).requested_strategy == "adaptive_search"
    study.candidate_strategy = "tpe"
    assert analyzer.propose(study, feedback, ["tpe"]).requested_strategy == "tpe"


def test_failed_trial_metric_is_not_counted_as_completed_feedback() -> None:
    study = _study()
    feedback = HPOFeedbackAnalyzer().analyze(study, [
        Trial("failed", {"lr": 0.1}, study.budgets[0], status="failed", metrics={"eer": 0.01}),
    ])

    assert feedback["completed_trials"] == 0
    assert feedback["failed_trials"] == 1
    assert feedback["best_metric"] is None


def test_campaign_stops_on_target_patience_and_total_cost() -> None:
    policy = CampaignPolicy()
    target = OptimizationCampaign(Objective("eer", "min"), target_value=0.05, max_studies=5, patience=2)
    policy.record_study(target, experiment_id="a", study_id="a", best_value=0.04, training_runs=2)
    assert not policy.should_continue(target)
    assert target.stop_reason == "target_reached"

    patience = OptimizationCampaign(Objective("eer", "min"), max_studies=5, patience=2, min_improvement=0.01)
    policy.record_study(patience, experiment_id="a", study_id="a", best_value=0.10, training_runs=1)
    assert policy.should_continue(patience)
    policy.record_study(patience, experiment_id="b", study_id="b", best_value=0.095, training_runs=1)
    policy.record_study(patience, experiment_id="c", study_id="c", best_value=0.094, training_runs=1)
    assert not policy.should_continue(patience)
    assert patience.stop_reason == "patience_exhausted"

    cost = OptimizationCampaign(Objective("eer", "min"), max_studies=5, patience=5, max_total_training_runs=2)
    policy.record_study(cost, experiment_id="a", study_id="a", best_value=0.1, training_runs=2)
    assert not policy.should_continue(cost)
    assert cost.stop_reason == "max_total_training_runs_reached"
