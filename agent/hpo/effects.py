"""Auditable decision outcomes and conservative observational effect estimates."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from agent.core.metrics import is_finite_metric

from .contracts import Objective, Trial
from .history import budget_key


def summarize_decision_effect(
    trials: Iterable[Trial],
    decision_id: str,
    objective: Objective,
    *,
    decision_created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Return realized results plus a same-fidelity pre-decision reference.

    The reference is deliberately labelled observational. It improves audit
    usefulness without claiming that sequential HPO observations identify a
    causal effect.
    """
    items = list(trials)
    affected = [
        trial for trial in items
        if (trial.provenance or {}).get("decision_id") == decision_id
    ]
    completed = _valid_trials(affected, objective)
    values = [float(trial.metrics[objective.metric]) for trial in completed]
    outcome = {
        "affected_trial_count": len(affected),
        "completed_trial_count": len(completed),
        "highest_completed_rung": max((trial.rung for trial in completed), default=None),
        "best_primary_metric": _best(values, objective.mode),
        "average_primary_metric": round(sum(values) / len(values), 8) if values else None,
        "training_seconds": round(sum(_duration(item, "training") for item in affected), 3),
        "evaluation_seconds": round(sum(_duration(item, "evaluation") for item in affected), 3),
    }
    return {
        "affected_trial_ids": [trial.trial_id for trial in affected],
        "realized_outcome": outcome,
        "effect_estimate": _observational_reference(
            items,
            completed,
            decision_id,
            objective,
            decision_created_at=decision_created_at,
        ),
    }


def _valid_trials(trials: Iterable[Trial], objective: Objective) -> List[Trial]:
    return [
        trial for trial in trials
        if trial.status in {"completed", "promoted"}
        and is_finite_metric(trial.metrics.get(objective.metric))
    ]


def _duration(trial: Trial, stage: str) -> float:
    nested = trial.cost.get(stage) or {}
    value = nested.get("duration_seconds")
    if value is None:
        value = trial.cost.get(f"attempt_{stage}_seconds")
    return float(value or 0.0)


def _best(values: List[float], mode: str) -> Optional[float]:
    if not values:
        return None
    return min(values) if mode == "min" else max(values)


def _fidelity_key(trial: Trial) -> tuple[Any, ...]:
    return int(trial.rung), *budget_key(trial.budget.to_dict())


def _observational_reference(
    trials: List[Trial],
    affected: List[Trial],
    decision_id: str,
    objective: Objective,
    *,
    decision_created_at: Optional[str],
) -> Dict[str, Any]:
    base = {
        "method": "pre_decision_same_study_same_fidelity_reference",
        "reference_kind": "observational_matched_history",
        "causal_claim": False,
        "causal_status": "not_identified_without_randomized_control",
        "randomized_control_trial_ids": [],
        "identification_requirements": {
            "same_confirmation_protocol": True,
            "same_budget_and_rung": True,
            "contemporaneous_random_assignment": True,
            "multiple_seeds": True,
        },
        "objective": objective.to_dict(),
        "comparisons": [],
        "aggregate_objective_aligned_mean_lift": None,
        "limitations": [
            "reference_trials_are_not_randomized",
            "sequential_time_and_sampler_confounding_remain",
            "use_frozen_multi_seed_benchmarks_for_causal_system_claims",
        ],
    }
    if not affected:
        return {**base, "status": "no_completed_affected_trials"}

    affected_by_fidelity: Dict[tuple[Any, ...], List[Trial]] = {}
    for trial in affected:
        try:
            affected_by_fidelity.setdefault(_fidelity_key(trial), []).append(trial)
        except ValueError:
            continue

    eligible_reference = [
        trial for trial in _valid_trials(trials, objective)
        if (trial.provenance or {}).get("decision_id") != decision_id
        and decision_created_at is not None
        and bool(trial.created_at)
        and trial.created_at <= decision_created_at
    ]
    comparisons: List[Dict[str, Any]] = []
    weighted_lift = 0.0
    weight = 0
    for fidelity, affected_group in sorted(
        affected_by_fidelity.items(), key=lambda item: repr(item[0])
    ):
        reference_group = []
        for trial in eligible_reference:
            try:
                if _fidelity_key(trial) == fidelity:
                    reference_group.append(trial)
            except ValueError:
                continue
        reference_group.sort(key=lambda trial: trial.created_at, reverse=True)
        reference_group = reference_group[: len(affected_group)]
        affected_values = [float(trial.metrics[objective.metric]) for trial in affected_group]
        reference_values = [float(trial.metrics[objective.metric]) for trial in reference_group]
        affected_mean = sum(affected_values) / len(affected_values)
        reference_mean = (
            sum(reference_values) / len(reference_values) if reference_values else None
        )
        lift = None
        if reference_mean is not None:
            lift = (
                reference_mean - affected_mean
                if objective.mode == "min" else affected_mean - reference_mean
            )
            matched_weight = min(len(affected_values), len(reference_values))
            weighted_lift += lift * matched_weight
            weight += matched_weight
        comparisons.append({
            "rung": fidelity[0],
            "budget": affected_group[0].budget.to_dict(),
            "affected_trial_ids": [trial.trial_id for trial in affected_group],
            "reference_trial_ids": [trial.trial_id for trial in reference_group],
            "affected_mean": round(affected_mean, 8),
            "affected_best": _best(affected_values, objective.mode),
            "reference_mean": round(reference_mean, 8) if reference_mean is not None else None,
            "reference_best": _best(reference_values, objective.mode),
            "objective_aligned_mean_lift": round(lift, 8) if lift is not None else None,
        })
    return {
        **base,
        "status": "available" if weight else "insufficient_same_fidelity_reference",
        "comparisons": comparisons,
        "aggregate_objective_aligned_mean_lift": (
            round(weighted_lift / weight, 8) if weight else None
        ),
    }


__all__ = ["summarize_decision_effect"]
