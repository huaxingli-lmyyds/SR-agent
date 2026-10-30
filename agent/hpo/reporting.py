"""Campaign history and metric summaries shared by new and resumed Studies."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.core.metrics import is_finite_metric

from .contracts import Objective, OptimizationCampaign
from .history import observation_signature
from .protocol import METRIC_PROTOCOL_ID, METRIC_UNITS


def campaign_from_dict(
    value: Any,
    fallback_objective: Objective,
) -> OptimizationCampaign:
    data = dict(value or {}) if isinstance(value, dict) else {}
    objective_data = data.get("objective") or fallback_objective.to_dict()
    if (
        objective_data.get("metric") != fallback_objective.metric
        or objective_data.get("mode") != fallback_objective.mode
    ):
        raise ValueError(
            "persisted Campaign objective conflicts with Study objective"
        )
    return OptimizationCampaign(
        objective=fallback_objective,
        target_value=data.get("target_value"),
        max_studies=max(int(data.get("max_studies") or 1), 1),
        patience=max(int(data.get("patience") or 1), 1),
        min_improvement=float(data.get("min_improvement") or 0.0),
        max_total_training_runs=data.get("max_total_training_runs"),
        resource_budget=dict(data.get("resource_budget") or {}),
        confirmation_signature=dict(data.get("confirmation_signature") or {}),
        campaign_id=str(
            data.get("campaign_id")
            or f"campaign_resume_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        ),
        study_summaries=[
            dict(item)
            for item in (data.get("study_summaries") or [])
            if isinstance(item, dict)
        ],
        best_value=data.get("best_value"),
        best_experiment_id=data.get("best_experiment_id"),
        status=str(data.get("status") or "running"),
        stop_reason=data.get("stop_reason"),
        created_at=str(data.get("created_at") or datetime.now().isoformat()),
        updated_at=str(data.get("updated_at") or datetime.now().isoformat()),
    )


def merge_campaign_history(
    existing: list[dict[str, Any]],
    trials: list[Any],
    objective: Objective,
    pruner: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Keep fidelity-compatible observations for subsequent Study samplers."""
    merged = {
        observation_signature(item): dict(item)
        for item in existing
        if isinstance(item, dict)
        and is_finite_metric((item.get("metrics") or {}).get(objective.metric))
    }
    for trial in trials:
        if (
            pruner == "successive_halving"
            and int(getattr(trial, "rung", 0)) != 0
        ):
            continue
        value = (getattr(trial, "metrics", {}) or {}).get(objective.metric)
        if not is_finite_metric(value):
            continue
        if getattr(trial, "status", None) not in {"completed", "promoted"}:
            continue
        parameters = dict(getattr(trial, "parameters", {}) or {})
        observation = {
            "trial_id": f"warm_{trial.trial_id}",
            "parameters": parameters,
            "budget": trial.budget.to_dict(),
            "status": "completed",
            "parent_trial_id": getattr(trial, "parent_trial_id", None),
            "rung": int(getattr(trial, "rung", 0)),
            "metrics": dict(getattr(trial, "metrics", {}) or {}),
            "intermediate_metrics": [],
            "cost": {"source": "prior_study"},
            "artifacts": [],
            "stop_reason": None,
            "candidate_source": getattr(trial, "candidate_source", "optimizer"),
            "search_phase": int(getattr(trial, "search_phase", 0)),
            "proposal_id": getattr(trial, "proposal_id", None),
            "hypothesis_id": getattr(trial, "hypothesis_id", None),
            "provenance": dict(getattr(trial, "provenance", {}) or {}),
            "created_at": getattr(trial, "created_at", ""),
            "updated_at": getattr(trial, "updated_at", ""),
        }
        merged[observation_signature(observation)] = observation
    return list(merged.values())[-max(int(limit), 1) :]


def study_learning_summary(
    study: Any, best_trial: Any, objective: Objective
) -> dict[str, Any]:
    reviews = list(getattr(study, "strategy_reviews", []) or [])
    last_review = reviews[-1] if reviews else {}
    decision = last_review.get("decision") or {}
    proposal = last_review.get("proposal") or {}
    final_sampler = (
        getattr(study, "candidate_strategy", None)
        or getattr(study, "sampler_strategy", None)
        or (
            "random_search"
            if getattr(study, "strategy", None) == "successive_halving"
            else getattr(study, "strategy", None)
        )
    )
    final_pruner = getattr(study, "pruner_strategy", None) or (
        "successive_halving"
        if getattr(study, "strategy", None) == "successive_halving"
        else "none"
    )
    final_strategy = (
        "successive_halving"
        if final_pruner == "successive_halving"
        else final_sampler
    )
    final_search_space = study.search_space.to_dict()
    deferred_proposal = dict(getattr(study, "next_study_proposal", None) or {})
    return {
        "local_search_anchor": {
            "trial_id": getattr(best_trial, "trial_id", None),
            "parameters": dict(getattr(best_trial, "parameters", {}) or {}),
            "metric": objective.metric,
            "mode": objective.mode,
            "value": (getattr(best_trial, "metrics", {}) or {}).get(
                objective.metric
            ),
        },
        "final_strategy": final_strategy,
        "final_candidate_strategy": final_sampler,
        "final_sampler": final_sampler,
        "final_sampler_config": dict(
            getattr(study, "sampler_config", {}) or {}
        ),
        "final_pruner": final_pruner,
        "warm_start_trial_count": len(
            getattr(study, "warm_start_trials", []) or []
        ),
        "final_search_space": final_search_space,
        "search_phases": list(getattr(study, "search_phases", []) or []),
        "search_phase_summaries": (
            (last_review.get("feedback") or {}).get("search_phase_summaries")
            or []
        ),
        "last_review": {
            "trigger": last_review.get("trigger"),
            "proposal_action": proposal.get("action"),
            "requested_strategy": proposal.get("requested_strategy"),
            "reason_codes": proposal.get("reason_codes") or [],
            "accepted_fields": decision.get("accepted_fields") or [],
            "rejected_fields": decision.get("rejected_fields") or [],
            "applied_candidate_strategy": last_review.get(
                "applied_candidate_strategy"
            ),
        },
        "next_study_recommendation": {
            "strategy": final_strategy,
            "sampler": final_sampler,
            "sampler_config": dict(getattr(study, "sampler_config", {}) or {}),
            "pruner": final_pruner,
            "search_space": final_search_space,
            "anchor_parameters": dict(
                getattr(best_trial, "parameters", {}) or {}
            ),
            "reason_codes": proposal.get("reason_codes") or [],
            "deferred_proposal": deferred_proposal or None,
        },
    }


def cross_study_memory(campaign: dict[str, Any]) -> dict[str, Any]:
    summaries = list(campaign.get("study_summaries") or [])
    recent = summaries[-3:]
    best_summary = None
    best_value = campaign.get("best_value")
    best_experiment_id = campaign.get("best_experiment_id")
    if best_experiment_id:
        best_summary = next(
            (
                item
                for item in summaries
                if item.get("experiment_id") == best_experiment_id
            ),
            None,
        )
    if best_summary is None and summaries:
        best_summary = summaries[-1]
    latest_learning = (
        recent[-1].get("learning_summary") if recent else None
    ) or {}
    return {
        "prior_study_count": len(summaries),
        "best_value": best_value,
        "best_experiment_id": best_experiment_id,
        "best_parameters": (best_summary or {}).get("best_parameters") or {},
        "local_search_anchor": latest_learning.get("local_search_anchor"),
        "next_study_recommendation": latest_learning.get(
            "next_study_recommendation"
        ),
        "recent_learnings": [
            {
                "experiment_id": item.get("experiment_id"),
                "study_id": item.get("study_id"),
                "best_value": item.get("best_value"),
                "improvement": item.get("improvement"),
                "improved": item.get("improved"),
                "best_parameters": item.get("best_parameters") or {},
                "learning_summary": item.get("learning_summary") or {},
            }
            for item in recent
        ],
    }


def best_metric_record(trial: Any, objective: Objective) -> dict[str, Any]:
    metrics = dict(trial.metrics or {})
    training, evaluation = split_trial_metrics(metrics, objective.metric)
    primary_value = metrics.get(objective.metric)
    return {
        "trial_id": trial.trial_id,
        "primary_metric": objective.metric,
        "primary_mode": objective.mode,
        "primary_value": primary_value,
        "metric_protocol": METRIC_PROTOCOL_ID,
        "metric_units": dict(METRIC_UNITS),
        **metrics,
        "training": training,
        "evaluation": evaluation,
    }


def split_trial_metrics(
    metrics: dict[str, Any],
    primary_metric: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}
    training_names = {
        "valid_error_rate",
        "final_epoch",
        "final_lr",
        "final_train_loss",
        "final_valid_loss",
        "final_valid_error_rate",
        "total_epochs",
        "best_epoch",
        "best_valid_loss",
        "best_error_rate",
    }
    evaluation_names = {
        primary_metric,
        "eer",
        "min_dcf",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
    }
    for key, value in metrics.items():
        normalized = key.lower()
        if key in training_names or normalized.startswith(
            ("train_", "valid_", "final_", "best_")
        ):
            training[key] = value
        elif key in evaluation_names:
            evaluation[key] = value
    return training, evaluation


def compact_campaign(campaign: dict[str, Any]) -> dict[str, Any]:
    objective = campaign.get("objective") or {}
    summaries = list(campaign.get("study_summaries") or [])
    return {
        "campaign_id": campaign.get("campaign_id"),
        "status": campaign.get("status"),
        "stop_reason": campaign.get("stop_reason"),
        "objective": {
            "metric": objective.get("metric"),
            "mode": objective.get("mode"),
        },
        "target_value": campaign.get("target_value"),
        "max_studies": campaign.get("max_studies"),
        "max_total_training_runs": campaign.get("max_total_training_runs"),
        "resource_budget": campaign.get("resource_budget") or {},
        "confirmation_signature": campaign.get("confirmation_signature") or {},
        "best_value": campaign.get("best_value"),
        "study_count": len(summaries),
        "recent_studies": summaries[-3:],
    }


def compact_study(study: Any) -> dict[str, Any]:
    return {
        "study_id": getattr(study, "study_id", None),
        "experiment_id": getattr(study, "experiment_id", None),
        "status": getattr(study, "status", None),
        "strategy": getattr(study, "strategy", None),
        "sampler_strategy": getattr(study, "sampler_strategy", None),
        "pruner_strategy": getattr(study, "pruner_strategy", None),
        "candidate_strategy": getattr(study, "candidate_strategy", None),
        "controller_mode": getattr(study, "controller_mode", None),
        "sampler_config": dict(getattr(study, "sampler_config", {}) or {}),
        "search_phases": list(getattr(study, "search_phases", []) or []),
        "best_trial_id": getattr(study, "best_trial_id", None),
        "objective": (
            study.objectives[0].to_dict()
            if getattr(study, "objectives", None)
            else None
        ),
        "metric_protocol": (getattr(study, "history_context", {}) or {}).get(
            "metric_protocol"
        ),
        "trial_count": len(getattr(study, "trial_ids", []) or []),
        "max_training_runs": getattr(study, "max_training_runs", None),
        "initial_trial_count": getattr(study, "initial_trial_count", None),
        "promotion_limits": list(getattr(study, "promotion_limits", []) or []),
        "reduction_factor": getattr(study, "reduction_factor", None),
        "warm_start_trial_count": len(
            getattr(study, "warm_start_trials", []) or []
        ),
        "search_space": study.search_space.to_dict(),
        "budgets": [item.to_dict() for item in getattr(study, "budgets", [])],
        "recent_reviews": list(getattr(study, "strategy_reviews", []) or [])[
            -3:
        ],
        "next_study_proposal": getattr(study, "next_study_proposal", None),
    }
