"""A small menu of executable next-Study plans; no LLM-authored budgets."""

from copy import deepcopy
from dataclasses import replace
from math import ceil

from agent.core.metrics import is_finite_metric

from . import cross_stage
from .campaign import CampaignPolicy
from .records import search_space_from_dict


def options(service, study, campaign):
    state = campaign.control_state
    remaining = CampaignPolicy.remaining_runs(campaign)
    cap = (
        min(state["per_study_limit"], remaining)
        if remaining is not None
        else state["per_study_limit"]
    )
    probe = bool(
        study.scheduler_state.get("hypothesis_probe_policy")
        or cross_stage.enabled(study)
    )
    sampler = (
        "tpe" if probe else study.candidate_strategy or study.sampler_strategy
    )
    # A finite candidate queue belongs to the old Study, not the new one.
    if sampler == "agent_proposal":
        sampler = (
            "tpe" if "tpe" in service.available_samplers() else "random_search"
        )
    configs = (
        study.sampler_config
        if sampler == (study.candidate_strategy or study.sampler_strategy)
        else {}
    )
    allocations = []
    initial_limit = (
        state["per_study_limit"]
        if study.pruner_strategy == "none"
        else state["initial_trial_count"]
    )
    for count in range(min(initial_limit, cap), 0, -1):
        promotions = []
        current = count
        if study.pruner_strategy == "successive_halving":
            for limit in state["promotion_limits"]:
                current = min(ceil(current / study.reduction_factor), limit)
                promotions.append(current)
        required = count + sum(promotions)
        if promotions and any(
            n < study.min_completed_per_rung for n in [count, *promotions[:-1]]
        ):
            continue
        if required > cap or (
            len(study.budgets) > 1
            and (
                len(promotions) != len(study.budgets) - 1 or not all(promotions)
            )
        ):
            continue
        preview = replace(
            study,
            initial_trial_count=count,
            promotion_limits=promotions,
            max_training_runs=required,
            max_trials=required,
            sampler_strategy=sampler,
            candidate_strategy=sampler,
        )
        if cross_stage.enabled(study):
            try:
                cross_stage.validate_plan(preview)
            except ValueError:
                continue
        allocations.append(
            {
                "max_training_runs": required,
                "initial_trial_count": count,
                "promotion_limits": promotions,
            }
        )
    if not allocations:
        return {}
    base = {
        **allocations[0],
        "max_training_runs": cap,
        "sampler": sampler,
        "sampler_config": configs,
        "search_space": study.search_space.to_dict(),
    }
    proposed = {"continue": base}
    smaller = [
        a
        for a in allocations
        if a["initial_trial_count"] < base["initial_trial_count"]
    ]
    if smaller:
        compact = min(
            smaller,
            key=lambda a: abs(
                a["initial_trial_count"] - ceil(base["initial_trial_count"] / 2)
            ),
        )
        proposed["compact"] = {**base, **compact}
    exploratory = "tpe" if probe else "random_search"
    global_plan = {
        **base,
        "sampler": exploratory,
        "sampler_config": {},
        "search_space": state["original_search_space"],
    }
    if global_plan != base:
        proposed["global_explore"] = global_plan
    focus = study.scheduler_state.get("cross_stage_focus")
    outcome = cross_stage.focus_outcome(
        study, service.list_trials(study.experiment_id)
    )
    if (
        focus
        and outcome
        and outcome["status"] == "evaluated"
        and outcome.get("initial_budget_improvement_over_reference", 0) > 0
    ):
        space = deepcopy(study.search_space.to_dict())
        for parameter in space["parameters"]:
            if parameter["name"] in focus["bounds"]:
                parameter.update(focus["bounds"][parameter["name"]])
        proposed["supported_focus"] = {**base, "search_space": space}
    result = {}
    for name, plan in proposed.items():
        try:
            service.validate_study_plan(
                search_space_from_dict(plan["search_space"]),
                study.objectives,
                study.budgets,
                sampler_strategy=plan["sampler"],
                pruner_strategy=study.pruner_strategy,
                sampler_config=plan["sampler_config"],
                reduction_factor=study.reduction_factor,
                max_training_runs=plan["max_training_runs"],
                initial_trial_count=plan["initial_trial_count"],
                promotion_limits=plan["promotion_limits"],
                candidate_batch_size=study.candidate_batch_size,
            )
        except (ValueError, KeyError):
            continue
        result[name] = deepcopy(plan)
    return result


def context(service, study, campaign, plans, resource_budget):
    trials = service.list_trials(study.experiment_id)
    for plan in plans.values():
        estimates = {"gpu_hours": 0.0, "stage_seconds": 0.0}
        complete = True
        for rung, count in enumerate(
            [plan["initial_trial_count"], *plan["promotion_limits"]]
        ):
            samples = [
                cross_stage.observed_cost([t])
                for t in trials
                if t.rung == rung
                and t.budget.to_dict() == study.budgets[rung].to_dict()
                and t.status in {"completed", "promoted"}
            ]
            samples = [s for s in samples if not s["cost_may_be_incomplete"]]
            if not samples:
                complete = False
                continue
            estimates["gpu_hours"] += count * max(
                s["gpu_hours"] for s in samples
            )
            estimates["stage_seconds"] += count * max(
                s["training_seconds"] + s["evaluation_seconds"] for s in samples
            )
        plan["estimated_cost"] = {
            **estimates,
            "complete": complete,
            "method": "same-rung maximum observed cost; not a runtime guarantee",
        }
    screened = {}
    for name, plan in list(plans.items()):
        estimate = plan["estimated_cost"]
        for cost_key, remaining_key in (
            ("gpu_hours", "gpu_hours_remaining"),
            ("stage_seconds", "wall_seconds_remaining"),
        ):
            remaining = (resource_budget or {}).get(remaining_key)
            if (
                estimate["complete"]
                and is_finite_metric(remaining)
                and estimate[cost_key] > remaining
            ):
                screened[name] = {
                    "reason": "estimated_complete_bracket_exceeds_remaining_budget",
                    "estimated_cost": estimate,
                }
                del plans[name]
                break
    summaries = campaign.study_summaries
    return {
        "objective": campaign.objective.to_dict(),
        "best_value": campaign.best_value,
        "metric_units": study.history_context.get("metric_units", {}),
        "remaining_training_runs": CampaignPolicy.remaining_runs(campaign),
        "resource_budget": resource_budget or {},
        "plans": plans,
        "budget_screened_plans": screened,
        "stop_allowed": len(summaries) >= 2 and not summaries[-1]["improved"],
        "studies": [
            {
                key: item.get(key)
                for key in (
                    "study_id",
                    "best_value",
                    "improvement",
                    "improved",
                    "training_runs",
                    "resource_summary",
                )
            }
            for item in summaries
        ],
        "confirmation_budget": study.budgets[-1].to_dict(),
    }


def validate_response(value, evidence):
    if not isinstance(value, dict) or set(value) != {
        "plan_id",
        "evidence_study_ids",
        "reason",
    }:
        raise ValueError(
            "return exactly plan_id, evidence_study_ids and reason"
        )
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("reason must be a nonempty string")
    allowed = set(evidence["plans"]) | (
        {"stop"} if evidence["stop_allowed"] else set()
    )
    if not isinstance(value["plan_id"], str) or value["plan_id"] not in allowed:
        raise ValueError("plan_id is not an allowed decision")
    ids = value["evidence_study_ids"]
    known = {s["study_id"] for s in evidence["studies"]}
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(s, str) for s in ids)
        or not set(ids) <= known
        or evidence["studies"][-1]["study_id"] not in ids
    ):
        raise ValueError(
            "cite observed Study IDs including the most recent Study"
        )


def rule_choice(evidence):
    plans = evidence["plans"]
    latest = evidence["studies"][-1]
    if not latest["improved"] and "compact" in plans:
        return "compact"
    # Resource screening has already removed unaffordable complete brackets.
    return next(
        (name for name in ("continue", "compact") if name in plans),
        next(iter(plans)),
    )
