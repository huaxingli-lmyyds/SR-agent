"""Budget-neutral paired promotion and one temporary TPE region per Study."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from math import ceil, isfinite, log
from uuid import NAMESPACE_URL, uuid5

from agent.core.metrics import is_finite_metric

from .contracts import Trial

FIELDS = {
    "action",
    "hypothesis",
    "anchor_trial_id",
    "challenger_trial_id",
    "evidence_trial_ids",
    "minimum_late_advantage",
    "region",
    "reason",
    "confidence",
}
ACTIONS = {"keep_search", "paired_promotion", "focus_region"}


def enabled(study):
    return bool(study.scheduler_state.get("cross_stage_policy"))


def validate_plan(study):
    if (
        study.controller_mode != "llm"
        or study.sampler_strategy != "tpe"
        or study.pruner_strategy != "successive_halving"
    ):
        raise ValueError(
            "cross-stage probes require an LLM controller with TPE and successive halving"
        )
    if study.scheduler_state.get("hypothesis_probe_policy"):
        raise ValueError(
            "select either same-budget hypothesis probes or cross-stage probes for a Study"
        )
    initial = study.initial_trial_count or study.max_trials or 0
    if (
        initial < 6
        or not study.candidate_batch_size
        or not 3 <= study.candidate_batch_size <= initial // 2
    ):
        raise ValueError(
            "cross-stage probes require >=6 initial trials and a batch size between 3 and half the initial quota"
        )
    if (
        not study.promotion_limits
        or min(
            study.promotion_limits[0], ceil(initial / study.reduction_factor)
        )
        < 3
    ):
        raise ValueError(
            "cross-stage probes need two paired slots and at least one remaining merit promotion slot"
        )
    if (
        len(study.budgets) < 2
        or study.budgets[0].data_fraction != study.budgets[1].data_fraction
    ):
        raise ValueError(
            "cross-stage probes currently compare training duration with an unchanged data fraction"
        )
    if (
        not study.budgets[0].epochs
        or not study.budgets[1].epochs
        or study.budgets[1].epochs <= study.budgets[0].epochs
    ):
        raise ValueError("cross-stage probes require increasing epoch budgets")


def _observed(study, trial, rung):
    return (
        0 <= rung < len(study.budgets)
        and trial.rung == rung
        and trial.status in {"completed", "promoted"}
        and trial.budget.to_dict() == study.budgets[rung].to_dict()
        and is_finite_metric(trial.metrics.get(study.objectives[0].metric))
    )


def pair_outcome(study, trials):
    pair = study.scheduler_state.get("cross_stage_pair")
    if not pair:
        return None
    result = {
        "pair_id": pair["pair_id"],
        "status": "inconclusive",
        "causal_claim": False,
        "claim_scope": "configuration ranking between the two recorded budgets",
        "source_trial_ids": pair["source_trial_ids"],
        "child_trial_ids": pair["child_trial_ids"],
        "varying_parameters": pair["varying_parameters"],
        "full_budget_comparison": len(study.budgets) == 2,
        "cycle_phase_explanation": "unverified; this experiment does not intervene on the scheduler",
    }
    by_id = {t.trial_id: t for t in trials}
    records = [
        by_id.get(tid)
        for tid in pair["source_trial_ids"] + pair["child_trial_ids"]
    ]
    if any(t is None for t in records):
        return result
    anchor, challenger, anchor_late, challenger_late = records
    if not all(
        _observed(study, t, rung)
        for t, rung in zip(records, (0, 0, 1, 1), strict=True)
    ):
        return result
    if (
        anchor_late.parent_trial_id != anchor.trial_id
        or challenger_late.parent_trial_id != challenger.trial_id
        or anchor_late.parameters != anchor.parameters
        or challenger_late.parameters != challenger.parameters
    ):
        return result
    metric = study.objectives[0].metric
    sign = 1 if study.objectives[0].mode == "min" else -1
    early_advantage = sign * (
        challenger.metrics[metric] - anchor.metrics[metric]
    )
    late_advantage = sign * (
        anchor_late.metrics[metric] - challenger_late.metrics[metric]
    )
    threshold = pair["minimum_late_advantage"]
    result.update(
        status="supported_observation"
        if early_advantage > 0 and late_advantage >= threshold
        else "not_supported",
        metric=metric,
        early_anchor_advantage=early_advantage,
        late_challenger_advantage=late_advantage,
        minimum_late_advantage=threshold,
        target_winner_trial_id=(
            challenger_late if late_advantage > 0 else anchor_late
        ).trial_id,
    )
    return result


def evidence(study, trials):
    """Keep linked budgets separate; never pool short and full-budget metrics."""
    by_id = {t.trial_id: t for t in trials}
    lineages = {}
    invalid_links = []
    for trial in trials:
        root = trial
        visited = set()
        while root.parent_trial_id:
            parent = by_id.get(root.parent_trial_id)
            if (
                root.trial_id in visited
                or parent is None
                or root.rung != parent.rung + 1
                or root.parameters != parent.parameters
            ):
                invalid_links.append(trial.trial_id)
                root = None
                break
            visited.add(root.trial_id)
            root = parent
        if root is None:
            continue
        lineages.setdefault(
            root.trial_id,
            {
                "root_trial_id": root.trial_id,
                "parameters": root.parameters,
                "observations": [],
            },
        )["observations"].append(
            {
                "trial_id": trial.trial_id,
                "parent_trial_id": trial.parent_trial_id,
                "rung": trial.rung,
                "budget": trial.budget.to_dict(),
                "status": trial.status,
                "metric": trial.metrics.get(study.objectives[0].metric),
                "training_observation": trial.cost.get("training_observation")
                or {},
            }
        )
    return {
        "metric": study.objectives[0].metric,
        "mode": study.objectives[0].mode,
        "lineages": list(lineages.values()),
        "invalid_lineage_trial_ids": invalid_links,
        "pair_outcome": pair_outcome(study, trials),
        "limitations": [
            "Promoted observations are selection-biased; missing higher-budget results are unknown.",
            "Promotions restart training; they are not one continuous learning curve.",
            "The selected checkpoint can precede the final epoch; absent phase metadata stays unknown.",
        ],
    }


def opportunity(study, trials):
    result = {
        "enabled": enabled(study),
        "eligible": False,
        "phase": None,
        "reason": "disabled",
    }
    if not result["enabled"]:
        return result
    result["reason"] = "waiting_for_completed_batch"
    if any(t.status in {"suggested", "running"} for t in trials):
        return result
    initial = [t for t in trials if t.rung == 0]
    observed = [t for t in initial if _observed(study, t, 0)]
    remaining = (study.initial_trial_count or study.max_trials) - len(initial)
    reviews = study.scheduler_state.get("cross_stage_reviews") or []
    reviewed = {r["phase"] for r in reviews}
    if (
        "paired_promotion" not in reviewed
        and len(observed) >= max(3, study.min_completed_per_rung)
        and remaining > 0
    ):
        if (
            not any(t.rung > 0 for t in trials)
            and study.promotion_limits[0] >= 3
        ):
            result.update(
                eligible=True,
                phase="paired_promotion",
                reason="ready",
                eligible_trial_ids=[t.trial_id for t in observed],
                destination_budget=study.budgets[1].to_dict(),
                remaining_merit_slots=study.promotion_limits[0] - 2,
            )
    elif "region_review" not in reviewed and remaining > 0:
        outcome = pair_outcome(study, trials)
        if outcome and outcome["status"] != "inconclusive":
            result.update(
                eligible=True,
                phase="region_review",
                reason="ready",
                pair_outcome=outcome,
                maximum_candidates=min(
                    remaining,
                    study.candidate_batch_size,
                    (study.initial_trial_count or study.max_trials) // 3,
                ),
            )
    if not result["eligible"]:
        result["reason"] = "no_remaining_decision_opportunity"
    return result


def validate_response(value, phase):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError(
            f"cross-stage response requires exactly {sorted(FIELDS)}"
        )
    allowed = {
        "keep_search",
        "paired_promotion" if phase == "paired_promotion" else "focus_region",
    }
    if value["action"] not in allowed:
        raise ValueError(f"allowed actions for {phase}: {sorted(allowed)}")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("reason must be a nonempty string")
    c = value["confidence"]
    if (
        isinstance(c, bool)
        or not isinstance(c, int | float)
        or not isfinite(c)
        or not 0 <= c <= 1
    ):
        raise ValueError("confidence must be a finite number in [0,1]")
    if not isinstance(value["evidence_trial_ids"], list) or any(
        not isinstance(x, str) for x in value["evidence_trial_ids"]
    ):
        raise ValueError(
            "evidence_trial_ids must be an array of exact observed IDs"
        )
    if value["action"] == "keep_search":
        if (
            any(
                value[key] is not None
                for key in (
                    "hypothesis",
                    "anchor_trial_id",
                    "challenger_trial_id",
                    "minimum_late_advantage",
                    "region",
                )
            )
            or value["evidence_trial_ids"]
        ):
            raise ValueError(
                "keep_search requires all decision fields null and evidence_trial_ids=[]"
            )
    elif value["action"] == "paired_promotion":
        if value["region"] is not None:
            raise ValueError(
                "paired_promotion cannot also change the search region"
            )
        for key in ("hypothesis", "anchor_trial_id", "challenger_trial_id"):
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError(f"paired_promotion requires {key}")
        delta = value["minimum_late_advantage"]
        if (
            isinstance(delta, bool)
            or not isinstance(delta, int | float)
            or not isfinite(delta)
            or delta <= 0
        ):
            raise ValueError(
                "minimum_late_advantage must be a positive finite number in objective units"
            )
    else:
        if any(
            value[key] is not None
            for key in (
                "hypothesis",
                "anchor_trial_id",
                "challenger_trial_id",
                "minimum_late_advantage",
            )
        ):
            raise ValueError(
                "focus_region requires the paired-promotion fields to be null"
            )
        if (
            not isinstance(value["region"], dict)
            or not 1 <= len(value["region"]) <= 2
        ):
            raise ValueError(
                "region must contain bounds for one or two numeric search parameters"
            )


def _validate_region(study, trials, proposal, status):
    outcome = status["pair_outcome"]
    required = set(outcome["source_trial_ids"] + outcome["child_trial_ids"])
    if not required <= set(proposal["evidence_trial_ids"]):
        raise ValueError(
            "region evidence must include both configurations at both budgets"
        )
    by_id = {t.trial_id: t for t in trials}
    if any(
        tid not in by_id or not _observed(study, by_id[tid], by_id[tid].rung)
        for tid in proposal["evidence_trial_ids"]
    ):
        raise ValueError("region evidence contains unobserved Trial IDs")
    winner = by_id[outcome["target_winner_trial_id"]]
    parameters = {p.name: p for p in study.search_space.parameters}
    for name, bounds in proposal["region"].items():
        if name not in outcome["varying_parameters"] or name not in parameters:
            raise ValueError(
                "focus only parameters that differed in the observed pair"
            )
        p = parameters[name]
        if p.parameter_type not in {"float", "int"} or p.condition:
            raise ValueError(
                "region focus supports unconditional numeric parameters only"
            )
        if not isinstance(bounds, dict) or set(bounds) != {"low", "high"}:
            raise ValueError("region bounds require exactly low and high")
        low, high = bounds["low"], bounds["high"]
        if any(
            isinstance(v, bool)
            or not isinstance(v, int | float)
            or not isfinite(v)
            for v in (low, high)
        ):
            raise ValueError("region bounds must be finite numbers")
        if p.parameter_type == "int" and any(
            type(v) is not int for v in (low, high)
        ):
            raise ValueError("integer parameter bounds must be integers")
        if (
            not p.low <= low < high <= p.high
            or not low <= winner.parameters[name] <= high
        ):
            raise ValueError(
                "region must remain inside the original domain and include the target-budget winner"
            )
        transform = log if p.scale == "log" else float
        width = transform(p.high) - transform(p.low)
        if transform(high) - transform(low) + 1e-12 * width < 0.5 * width:
            raise ValueError(
                "limited paired evidence cannot remove over half a parameter's linear/log interval"
            )


def review(service, study, proposal):
    trials = service.list_trials(study.experiment_id)
    status = opportunity(study, trials)
    if not status["eligible"]:
        return {"accepted": False, "reason": status["reason"]}
    now = datetime.now().isoformat()
    record = {
        "phase": status["phase"],
        "proposal": proposal,
        "accepted": False,
        "created_at": now,
        "reason": "keep_search",
    }
    try:
        validate_response(proposal, status["phase"])
        if (
            proposal["action"] != "keep_search"
            and proposal["confidence"] < 0.65
        ):
            raise ValueError(
                "cross-stage interventions require confidence >=0.65 and observed evidence"
            )
        if proposal["action"] == "paired_promotion":
            by_id = {t.trial_id: t for t in trials}
            source_ids = [
                proposal["anchor_trial_id"],
                proposal["challenger_trial_id"],
            ]
            if len(set(source_ids)) != 2 or not set(source_ids) <= set(
                status["eligible_trial_ids"]
            ):
                raise ValueError(
                    "pair must cite two distinct completed initial-budget configurations"
                )
            if not set(source_ids) <= set(
                proposal["evidence_trial_ids"]
            ) or not set(proposal["evidence_trial_ids"]) <= set(
                status["eligible_trial_ids"]
            ):
                raise ValueError(
                    "pair evidence must include its two actual source trials"
                )
            a, b = [by_id[tid] for tid in source_ids]
            metric = study.objectives[0].metric
            sign = 1 if study.objectives[0].mode == "min" else -1
            best = min(
                sign * by_id[tid].metrics[metric]
                for tid in status["eligible_trial_ids"]
            )
            if (
                sign * a.metrics[metric] != best
                or sign * a.metrics[metric] >= sign * b.metrics[metric]
            ):
                raise ValueError(
                    "anchor must be the current initial-budget winner; challenger must currently rank worse"
                )
            if a.parameters == b.parameters:
                raise ValueError("pair must compare different configurations")
            if service.remaining_training_runs(
                study
            ) < service.reserved_training_runs(study):
                raise ValueError(
                    "remaining training allowance cannot preserve the original allocation"
                )
            child_ids = [
                f"trial_{uuid5(NAMESPACE_URL, f'{study.study_id}:{tid}:1').hex}"
                for tid in source_ids
            ]
            study.scheduler_state["cross_stage_pair"] = {
                "pair_id": f"pair_{uuid5(NAMESPACE_URL, study.study_id).hex}",
                "source_trial_ids": source_ids,
                "child_trial_ids": child_ids,
                "hypothesis": proposal["hypothesis"],
                "minimum_late_advantage": proposal["minimum_late_advantage"],
                "varying_parameters": [
                    name
                    for name in a.parameters
                    if a.parameters[name] != b.parameters.get(name)
                ],
                "created_at": now,
                "promotion_training_mode": "restart",
            }
            record.update(accepted=True, reason="paired_promotion_reserved")
        elif proposal["action"] == "focus_region":
            _validate_region(study, trials, proposal, status)
            study.scheduler_state["cross_stage_focus"] = {
                "region_id": f"region_{uuid5(NAMESPACE_URL, study.study_id).hex}",
                "bounds": proposal["region"],
                "maximum_candidates": status["maximum_candidates"],
                "reason": proposal["reason"],
                "consumed": False,
                "source_pair_id": status["pair_outcome"]["pair_id"],
                "reference_trial_ids": [
                    t.trial_id for t in trials if _observed(study, t, 0)
                ],
            }
            record.update(accepted=True, reason="one_tpe_region_batch_reserved")
    except (ValueError, TypeError, KeyError) as exc:
        record["reason"] = str(exc)
    study.scheduler_state.setdefault("cross_stage_reviews", []).append(record)
    study.scheduler_state.pop("cross_stage_pending_review", None)
    # Commit intent before creating children. Resume finishes this exact pair without another consultation.
    service._save_study(study)
    ensure_pair_trials(service, study)
    return record


def ensure_pair_trials(service, study):
    pair = study.scheduler_state.get("cross_stage_pair")
    if not pair:
        return
    by_id = {t.trial_id: t for t in service.list_trials(study.experiment_id)}
    missing = [tid for tid in pair["child_trial_ids"] if tid not in by_id]
    if not missing:
        return
    if service.remaining_training_runs(study) < len(missing):
        raise ValueError(
            "insufficient reserved training capacity to recover paired promotion"
        )
    destination_count = sum(t.rung == 1 for t in by_id.values())
    if destination_count + len(missing) > study.promotion_limits[0]:
        raise ValueError(
            "paired promotion would exceed the frozen promotion allocation"
        )
    for source_id, child_id in zip(
        pair["source_trial_ids"], pair["child_trial_ids"], strict=True
    ):
        if child_id in by_id:
            continue
        parent = by_id[source_id]
        now = datetime.now().isoformat()
        child = Trial(
            trial_id=child_id,
            parameters=dict(parent.parameters),
            budget=study.budgets[1],
            parent_trial_id=source_id,
            rung=1,
            candidate_source="promotion",
            search_phase=parent.search_phase,
            proposal_id=parent.proposal_id,
            hypothesis_id=parent.hypothesis_id,
            provenance={
                **parent.provenance,
                "promoted_from": source_id,
                "original_candidate_source": parent.candidate_source,
                "cross_stage_pair_id": pair["pair_id"],
                "promotion_training_mode": "restart",
            },
            created_at=now,
            updated_at=now,
        )
        service.save_trial(study.experiment_id, child)
        parent.status = "promoted"
        parent.updated_at = now
        service.save_trial(study.experiment_id, parent)
        if child_id not in study.trial_ids:
            study.trial_ids.append(child_id)
    service._save_study(study)


def sampling_region(study, trials):
    focus = study.scheduler_state.get("cross_stage_focus")
    if (
        not focus
        or focus["consumed"]
        or any(
            t.provenance.get("region_id") == focus["region_id"] for t in trials
        )
    ):
        return study.search_space, None
    parameters = [
        replace(p, **focus["bounds"][p.name])
        if p.name in focus["bounds"]
        else p
        for p in study.search_space.parameters
    ]
    return replace(study.search_space, parameters=parameters), focus


def focus_outcome(study, trials):
    focus = study.scheduler_state.get("cross_stage_focus")
    if not focus:
        return None
    focused = [
        t
        for t in trials
        if t.rung == 0 and t.provenance.get("region_id") == focus["region_id"]
    ]
    completed = [t for t in focused if _observed(study, t, 0)]
    references = [
        t
        for t in trials
        if t.trial_id in focus["reference_trial_ids"] and _observed(study, t, 0)
    ]
    result = {
        "region_id": focus["region_id"],
        "bounds": focus["bounds"],
        "candidate_count": len(focused),
        "completed_count": len(completed),
        "status": "inconclusive",
        "causal_claim": False,
        "fallback_reason": focus.get("fallback_reason"),
        "global_domain_restored_for_next_batch": focus["consumed"],
    }
    if completed and len(completed) == len(focused) and references:
        metric = study.objectives[0].metric
        sign = 1 if study.objectives[0].mode == "min" else -1
        reference_best = min(sign * t.metrics[metric] for t in references)
        focused_best = min(sign * t.metrics[metric] for t in completed)
        result.update(
            status="evaluated",
            metric=metric,
            initial_budget_improvement_over_reference=reference_best
            - focused_best,
            interpretation="Same-budget best-of-batch comparison only, not confirmed full-budget improvement.",
        )
    return result


def summary(study, trials):
    if not enabled(study):
        return {"enabled": False}
    focused = [
        t for t in trials if t.rung == 0 and t.provenance.get("region_id")
    ]
    return {
        "enabled": True,
        "pair_outcome": pair_outcome(study, trials),
        "reviews": study.scheduler_state.get("cross_stage_reviews") or [],
        "focused_candidate_count": len(focused),
        "focused_trial_ids": [t.trial_id for t in focused],
        "global_search_space_preserved": study.search_space.to_dict()
        == study.scheduler_state["cross_stage_policy"].get(
            "original_search_space"
        ),
        "focus_outcome": focus_outcome(study, trials),
        "interpretation": "A two-configuration comparison is exploratory, not causal or full-budget proof unless both reach that budget.",
    }
