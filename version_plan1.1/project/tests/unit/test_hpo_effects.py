from agent.hpo import Objective, Trial, TrialBudget
from agent.hpo.effects import summarize_decision_effect


def test_decision_effect_uses_only_predecision_same_fidelity_references() -> None:
    screening = TrialBudget("screening", epochs=2, data_fraction=0.25)
    full = TrialBudget("full", epochs=10, data_fraction=1.0)
    trials = [
        Trial(
            "reference_same",
            {"lr": 0.001},
            screening,
            status="completed",
            rung=0,
            metrics={"eer": 0.30},
            created_at="2026-09-01T00:00:00",
        ),
        Trial(
            "reference_wrong_budget",
            {"lr": 0.002},
            full,
            status="completed",
            rung=0,
            metrics={"eer": 0.10},
            created_at="2026-09-01T00:00:00",
        ),
        Trial(
            "reference_after_decision",
            {"lr": 0.003},
            screening,
            status="completed",
            rung=0,
            metrics={"eer": 0.05},
            created_at="2026-09-03T00:00:00",
        ),
        Trial(
            "affected",
            {"lr": 0.0008},
            screening,
            status="completed",
            rung=0,
            metrics={"eer": 0.20},
            provenance={"decision_id": "decision_1"},
            created_at="2026-09-03T00:00:00",
        ),
    ]

    result = summarize_decision_effect(
        trials,
        "decision_1",
        Objective("eer", "min"),
        decision_created_at="2026-09-02T00:00:00",
    )

    estimate = result["effect_estimate"]
    comparison = estimate["comparisons"][0]
    assert estimate["status"] == "available"
    assert estimate["causal_claim"] is False
    assert comparison["reference_trial_ids"] == ["reference_same"]
    assert comparison["objective_aligned_mean_lift"] == 0.1


def test_decision_effect_does_not_invent_a_causal_control() -> None:
    budget = TrialBudget("full", epochs=10, data_fraction=1.0)
    affected = Trial(
        "affected",
        {"lr": 0.001},
        budget,
        status="completed",
        metrics={"eer": 0.2},
        provenance={"decision_id": "decision_1"},
    )

    result = summarize_decision_effect(
        [affected],
        "decision_1",
        Objective("eer", "min"),
        decision_created_at="2026-09-02T00:00:00",
    )

    assert result["effect_estimate"]["status"] == (
        "insufficient_same_fidelity_reference"
    )
    assert result["effect_estimate"]["causal_status"] == (
        "not_identified_without_randomized_control"
    )
