"""Bounded, falsifiable candidate experiments inside an unchanged TPE budget."""

from __future__ import annotations

from math import isfinite
from typing import Any

from agent.core.metrics import is_finite_metric

from .candidates import _candidate_signature
from .contracts import StrategyProposal


def policy_for(enabled: bool) -> dict:
    if type(enabled) is not bool:
        raise ValueError("hypothesis_probes must be a boolean")
    return (
        {
            "version": 1,
            "warmup_trials": 3,
            "max_candidates_per_batch": 3,
            "max_candidate_fraction": 1 / 3,
            "max_reviews": 2,
        }
        if enabled
        else {}
    )


def opportunity(study, trials) -> dict:
    policy = study.scheduler_state.get("hypothesis_probe_policy") or {}
    if not policy:
        return {"enabled": False, "eligible": False, "reason": "disabled"}
    initial = [t for t in trials if t.rung == 0]
    observed = [
        t
        for t in initial
        if t.status in {"completed", "promoted"}
        and t.budget.to_dict() == study.budgets[0].to_dict()
        and is_finite_metric(t.metrics.get(study.objectives[0].metric))
    ]
    limit = int(
        (study.max_training_runs or study.max_trials or 0)
        if study.pruner_strategy == "none"
        else (
            study.initial_trial_count
            or study.max_trials
            or study.max_training_runs
            or 0
        )
    )
    cap = int(limit * policy["max_candidate_fraction"])
    used = sum(t.candidate_source == "sampler:agent_proposal" for t in initial)
    pending = len(study.pending_candidate_proposals)
    reviews = sum(
        bool(r.get("hypothesis_probe_opportunity", {}).get("eligible"))
        for r in study.strategy_reviews
    )
    run_limit = study.max_training_runs or study.max_trials or limit
    remaining_runs = (
        run_limit
        - len(trials)
        - sum(int(t.cost.get("retry_count", 0)) for t in trials)
    )
    available = max(
        0,
        min(
            cap - used - pending,
            remaining_runs - pending,
            limit - len(initial) - pending,
            policy["max_candidates_per_batch"],
            study.candidate_batch_size or 3,
        ),
    )
    reason = "ready"
    if len(observed) < policy["warmup_trials"]:
        reason = "collect_initial_tpe_evidence"
    elif pending or any(t.status in {"suggested", "running"} for t in initial):
        reason = "batch_in_progress"
    elif (
        study.candidate_strategy or study.sampler_strategy
    ) == "agent_proposal":
        reason = "return_to_tpe_after_probe"
    elif available == 0:
        reason = "probe_quota_or_candidate_capacity_exhausted"
    elif reviews >= policy["max_reviews"]:
        reason = "consultation_limit_reached"
    return {
        "enabled": True,
        "eligible": reason == "ready",
        "reason": reason,
        "maximum_candidates": available,
        "candidate_cap": cap,
        "agent_candidates_used": used,
        "reviews_used": reviews,
        "observed_trials": [
            {
                "trial_id": t.trial_id,
                "parameters": t.parameters,
                "budget": t.budget.to_dict(),
                "metric": t.metrics[study.objectives[0].metric],
            }
            for t in observed
        ],
        "hypothesis_outcomes": [
            hypothesis_outcome(study, trials, h)
            for h in study.hypotheses
            if h.get("probe_design")
        ],
    }


def validate_proposal(
    study, trials, proposal, validate_parameters, additional_existing=()
):
    """Reject an entire incomplete design before committing any policy change."""
    status = opportunity(study, trials)
    if (
        not status["enabled"]
        or proposal is None
        or proposal.action == "invalid_proposal"
    ):
        return proposal, []
    try:
        if any(
            (
                proposal.search_space is not None,
                bool(proposal.sampler_config),
                proposal.requested_pruner is not None,
                proposal.budgets is not None,
                proposal.max_training_runs is not None,
                proposal.initial_trial_count is not None,
                proposal.promotion_limits is not None,
                proposal.reduction_factor is not None,
                proposal.requested_strategy is not None,
            )
        ):
            raise ValueError(
                "hypothesis probes preserve search space, sampler settings and all budgets"
            )
        if (
            proposal.action == "keep_strategy"
            and not proposal.candidate_proposals
            and not proposal.hypotheses
            and proposal.requested_sampler is None
        ):
            return proposal, []
        if not status["eligible"]:
            raise ValueError(status["reason"])
        if (
            proposal.action != "switch_strategy"
            or proposal.requested_sampler != "agent_proposal"
        ):
            raise ValueError(
                "probe mode only permits keep_strategy or an agent_proposal batch"
            )
        confidence = proposal.confidence
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, int | float)
            or not isfinite(confidence)
            or not 0.65 <= confidence <= 1
        ):
            raise ValueError(
                "probe proposal confidence must be finite and >=0.65"
            )
        if len(proposal.hypotheses) != 1:
            raise ValueError(
                "one explicit hypothesis is required per probe batch"
            )
        candidates = proposal.candidate_proposals
        if not 1 <= len(candidates) <= status["maximum_candidates"]:
            raise ValueError(
                "probe exceeds remaining candidate quota or batch capacity"
            )
        h = proposal.hypotheses[0]
        if not isinstance(h, dict):
            raise ValueError("hypothesis must be an object")
        for key in ("id", "claim", "anchor_trial_id", "probe_design"):
            if not isinstance(h.get(key), str) or not h[key].strip():
                raise ValueError(f"hypothesis requires {key}")
        if h["id"] != h["id"].strip():
            raise ValueError("hypothesis id cannot have surrounding whitespace")
        if h["id"] in {v.get("id") for v in study.hypotheses}:
            raise ValueError("hypothesis id already exists")
        observed = {t["trial_id"]: t for t in status["observed_trials"]}
        evidence = h.get("evidence_trial_ids")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(
                not isinstance(t, str) or t not in observed for t in evidence
            )
        ):
            raise ValueError(
                "evidence must reference successful, same-budget observations in this Study"
            )
        anchor = observed.get(h["anchor_trial_id"])
        if anchor is None or h["anchor_trial_id"] not in evidence:
            raise ValueError("anchor must be included in observed evidence")
        if (
            not isinstance(h.get("expected_signal"), dict)
            or not h["expected_signal"]
        ):
            raise ValueError("hypothesis requires expected_signal")
        parameters = h.get("tested_parameters")
        names = {p.name for p in study.search_space.parameters}
        if (
            not isinstance(parameters, list)
            or not parameters
            or any(not isinstance(p, str) or p not in names for p in parameters)
            or len(parameters) != len(set(parameters))
        ):
            raise ValueError(
                "tested_parameters must be distinct search parameter names"
            )
        design = h["probe_design"]
        if design not in {"single_factor", "boundary", "interaction"}:
            raise ValueError("unsupported probe_design")
        if len(parameters) != (2 if design == "interaction" else 1):
            raise ValueError(
                "single-factor/boundary requires one parameter; interaction requires two"
            )
        criterion = h.get("falsification_criterion") or {}
        if not isinstance(criterion, dict):
            raise ValueError("falsification_criterion must be an object")
        delta = criterion.get("minimum_improvement")
        if (
            criterion.get("metric") != study.objectives[0].metric
            or isinstance(delta, bool)
            or not isinstance(delta, int | float)
            or not isfinite(delta)
            or delta <= 0
        ):
            raise ValueError(
                "falsification criterion requires the objective metric and a positive finite minimum_improvement"
            )
        expected = (
            "interaction_gain"
            if design == "interaction"
            else "improvement_over_anchor"
        )
        if criterion.get("comparison") != expected:
            raise ValueError(f"falsification comparison must be {expected}")
        signatures = {_candidate_signature(t.parameters) for t in trials}
        signatures.update(
            _candidate_signature(p["parameters"])
            for p in study.pending_candidate_proposals
        )
        signatures.update(
            _candidate_signature(params) for params in additional_existing
        )
        candidate_ids = {
            item.get("candidate_id")
            for review in study.candidate_proposal_reviews
            for item in review.get("accepted", [])
            if item.get("candidate_id")
        }
        candidate_ids.update(
            item.get("candidate_id")
            for item in study.pending_candidate_proposals
        )
        changed_sets, values = [], []
        for c in candidates:
            if not isinstance(c, dict):
                raise ValueError("candidate must be an object")
            if c.get("hypothesis_id") != h["id"]:
                raise ValueError(
                    "every candidate must reference this hypothesis"
                )
            for key in ("role", "rationale"):
                if not isinstance(c.get(key), str) or not c[key].strip():
                    raise ValueError(f"candidate requires {key}")
            if (
                not isinstance(c.get("expected_signal"), dict)
                or not c["expected_signal"]
            ):
                raise ValueError("candidate requires expected_signal")
            if c.get("candidate_id"):
                if not isinstance(c["candidate_id"], str):
                    raise ValueError("candidate_id must be a string")
                if c["candidate_id"] in candidate_ids:
                    raise ValueError("duplicate candidate_id")
                candidate_ids.add(c["candidate_id"])
            confidence = c.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, int | float)
                or not isfinite(confidence)
                or not 0.65 <= confidence <= 1
            ):
                raise ValueError(
                    "candidate confidence must be finite and >=0.65"
                )
            params = validate_parameters(study, c.get("parameters") or {})
            signature = _candidate_signature(params)
            if signature in signatures:
                raise ValueError(
                    "candidate duplicates an existing or pending configuration"
                )
            signatures.add(signature)
            if set(params) != set(anchor["parameters"]):
                raise ValueError(
                    "probe cannot change conditional parameter activation"
                )
            changed = {
                p for p in params if params[p] != anchor["parameters"][p]
            }
            if not changed or not changed <= set(parameters):
                raise ValueError(
                    "candidate must change only tested parameters relative to anchor"
                )
            changed_sets.append(changed)
            values.append(params)
        if design == "interaction":
            x, y = parameters
            if len(values) != 3 or {frozenset(v) for v in changed_sets} != {
                frozenset([x]),
                frozenset([y]),
                frozenset([x, y]),
            }:
                raise ValueError(
                    "interaction requires two single-factor controls and one joint candidate"
                )
            joint = values[changed_sets.index({x, y})]
            if any(
                v[p] != joint[p]
                for v, changed in zip(values, changed_sets, strict=True)
                for p in changed
            ):
                raise ValueError(
                    "interaction controls must use the same perturbations as the joint candidate"
                )
        elif design == "boundary":
            parameter = next(
                p
                for p in study.search_space.parameters
                if p.name == parameters[0]
            )
            if parameter.low is None or parameter.high is None:
                raise ValueError("boundary probes require a numeric domain")
            levels = [
                anchor["parameters"][parameter.name],
                *[v[parameter.name] for v in values],
            ]
            if not any(
                v in {parameter.low, parameter.high} for v in levels
            ) or not any(parameter.low < v < parameter.high for v in levels):
                raise ValueError(
                    "boundary design must include a domain endpoint and an interior level"
                )
        return proposal, []
    except (ValueError, TypeError, KeyError) as exc:
        return StrategyProposal(
            action="keep_strategy",
            reason_codes=["hypothesis_probe_rejected"],
            evidence={"reason": str(exc)},
        ), [{"field": "hypothesis_probe", "reason": str(exc)}]


def hypothesis_outcome(study, trials, hypothesis) -> dict[str, Any]:
    by_id = {t.trial_id: t for t in trials}
    anchor = by_id.get(hypothesis.get("anchor_trial_id"))
    relevant = [
        t
        for t in trials
        if t.rung == 0 and t.hypothesis_id == hypothesis.get("id")
    ]
    result = {
        "hypothesis_id": hypothesis.get("id"),
        "status": "inconclusive",
        "causal_claim": False,
        "trial_ids": [t.trial_id for t in relevant],
        "anchor_trial_id": hypothesis.get("anchor_trial_id"),
    }
    if (
        not anchor
        or not relevant
        or any(
            item.get("hypothesis_id") == hypothesis.get("id")
            for item in study.pending_candidate_proposals
        )
    ):
        return result
    metric = study.objectives[0].metric
    if any(
        t.status not in {"completed", "promoted"}
        or t.budget.to_dict() != anchor.budget.to_dict()
        or not is_finite_metric(t.metrics.get(metric))
        for t in [anchor, *relevant]
    ):
        return result
    sign = 1 if study.objectives[0].mode == "min" else -1
    gains = {
        t.trial_id: sign * (anchor.metrics[metric] - t.metrics[metric])
        for t in relevant
    }
    score = max(gains.values())
    if hypothesis["probe_design"] == "interaction":
        joint = [
            t
            for t in relevant
            if sum(
                t.parameters[p] != anchor.parameters[p]
                for p in hypothesis["tested_parameters"]
            )
            == 2
        ]
        if len(relevant) != 3 or len(joint) != 1:
            return result
        score = gains[joint[0].trial_id] - sum(
            v for k, v in gains.items() if k != joint[0].trial_id
        )
    threshold = hypothesis["falsification_criterion"]["minimum_improvement"]
    result.update(
        status="supported_observation"
        if score >= threshold
        else "not_supported",
        comparison=hypothesis["falsification_criterion"]["comparison"],
        score=score,
        minimum_improvement=threshold,
        metric=metric,
        best_improvement_over_anchor=max(gains.values()),
        note="Same-budget observations; training noise and selection prevent a causal conclusion.",
    )
    return result


def performance_summary(study, trials) -> dict:
    probes = [
        t
        for t in trials
        if t.rung == 0 and t.candidate_source == "sampler:agent_proposal"
    ]
    costs = [
        (t.cost.get("attempt_training_gpu_seconds") or 0)
        + (t.cost.get("attempt_evaluation_gpu_seconds") or 0)
        for t in probes
    ]
    confirmed = [
        t
        for t in trials
        if t.rung == len(study.budgets) - 1
        and t.budget.to_dict() == study.budgets[-1].to_dict()
        and t.status in {"completed", "promoted"}
        and is_finite_metric(t.metrics.get(study.objectives[0].metric))
    ]
    values = [t.metrics[study.objectives[0].metric] for t in confirmed]
    return {
        "agent_candidate_count": len(probes),
        "agent_candidate_gpu_seconds": sum(costs),
        "agent_candidate_cost_may_be_incomplete": any(
            t.cost.get("retry_count")
            or t.cost.get("evaluation_retry_count")
            or any(
                t.cost.get(key) is None
                for key in (
                    "attempt_training_gpu_seconds",
                    "attempt_evaluation_gpu_seconds",
                )
            )
            for t in probes
        ),
        "confirmed_best_metric": (
            min(values) if study.objectives[0].mode == "min" else max(values)
        )
        if values
        else None,
        "outcomes": [
            hypothesis_outcome(study, trials, h)
            for h in study.hypotheses
            if study.scheduler_state.get("hypothesis_probe_policy")
            and h.get("probe_design")
        ],
        "interpretation": "Compare confirmed quality against total campaign GPU/wall cost; low-budget gains do not establish final improvement.",
    }
