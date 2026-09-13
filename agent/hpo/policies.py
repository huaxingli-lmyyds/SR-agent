"""Simple model-agnostic early stopping policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, prod
from typing import Any, Callable, Dict, List, Optional

from .contracts import (
    HPOStudy,
    Objective,
    SearchParameter,
    SearchSpace,
    StrategyDecisionRecord,
    StrategyProposal,
    TrialBudget,
)
from .history import budget_key


@dataclass
class StopDecision:
    should_stop: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class FailureDecision:
    category: str
    recoverable: bool
    retry_delay_seconds: float = 0.0


class FailurePolicy:
    """Classify common execution failures and decide whether to retry."""

    RECOVERABLE_MARKERS = (
        "timeout",
        "temporar",
        "connection",
        "resource busy",
        "out of memory",
        "cuda",
        "worker",
    )
    NON_RECOVERABLE_MARKERS = (
        "config",
        "not found",
        "invalid",
        "unsupported",
        "unknown adapter",
        "missing",
    )

    def classify(self, error: Optional[str]) -> FailureDecision:
        text = str(error or "").lower()
        if "invalidmetricerror" in text or "invalid metric " in text:
            return FailureDecision("invalid_metric", False)
        if any(marker in text for marker in self.NON_RECOVERABLE_MARKERS):
            return FailureDecision("configuration", False)
        if "timeout" in text:
            return FailureDecision("timeout", True)
        if "out of memory" in text or "cuda" in text:
            return FailureDecision("resource", True)
        if any(marker in text for marker in self.RECOVERABLE_MARKERS):
            return FailureDecision("transient", True)
        return FailureDecision("execution", False)


class RetryPolicy:
    """Bound retries for recoverable trial execution failures."""

    def __init__(self, max_retries: int = 1, retry_delay_seconds: float = 0.0) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def should_retry(self, attempt: int, failure: FailureDecision) -> bool:
        return failure.recoverable and attempt <= self.max_retries


class HPOPlanningPolicy:
    """Select an optimization strategy from validated request characteristics."""

    def select_strategy(
        self,
        requested: str,
        search_space: SearchSpace,
        budgets: List[TrialBudget],
        max_training_runs: int,
        available: List[str],
    ) -> str:
        if requested != "auto":
            if requested not in available:
                raise ValueError(f"unsupported HPO strategy: {requested}")
            return requested
        if len(budgets) > 1 and max_training_runs > 1:
            return "successive_halving"
        cardinality = self.grid_cardinality(search_space)
        if cardinality is not None and cardinality <= max_training_runs:
            return "grid_search"
        if max_training_runs >= 8 and "tpe" in available:
            return "tpe"
        return "adaptive_search" if max_training_runs >= 5 else "random_search"

    def select_components(
        self,
        requested_strategy: str,
        requested_sampler: Optional[str],
        requested_pruner: Optional[str],
        search_space: SearchSpace,
        budgets: List[TrialBudget],
        max_training_runs: int,
        available: List[str],
    ) -> tuple[str, str]:
        """Resolve independent candidate sampling and fidelity pruning policies."""
        sampler_request = str(requested_sampler or "auto")
        pruner_request = str(requested_pruner or "auto")
        legacy_request = str(requested_strategy or "auto")

        if sampler_request == "auto":
            if legacy_request not in {"auto", "successive_halving"}:
                sampler_request = legacy_request
            else:
                sampler_request = self._select_sampler(
                    search_space,
                    max_training_runs,
                    available,
                )
        if sampler_request == "successive_halving" or sampler_request not in available:
            raise ValueError(f"unsupported HPO sampler: {sampler_request}")

        if pruner_request == "auto":
            pruner_request = (
                "successive_halving"
                if legacy_request == "successive_halving" or len(budgets) > 1
                else "none"
            )
        if pruner_request not in {"none", "successive_halving"}:
            raise ValueError(f"unsupported HPO pruner: {pruner_request}")
        if pruner_request == "successive_halving" and len(budgets) < 2:
            raise ValueError("successive_halving pruner requires at least two budget rungs")
        return sampler_request, pruner_request

    def _select_sampler(
        self,
        search_space: SearchSpace,
        max_training_runs: int,
        available: List[str],
    ) -> str:
        cardinality = self.grid_cardinality(search_space)
        if cardinality is not None and cardinality <= max_training_runs:
            return "grid_search"
        if max_training_runs >= 8 and "tpe" in available:
            return "tpe"
        return "adaptive_search" if max_training_runs >= 5 else "random_search"

    @staticmethod
    def grid_cardinality(search_space: SearchSpace) -> Optional[int]:
        sizes = []
        for parameter in search_space.parameters:
            if parameter.choices:
                sizes.append(len(parameter.choices))
            elif (
                parameter.parameter_type == "int"
                and parameter.low is not None
                and parameter.high is not None
            ):
                sizes.append(int(parameter.high) - int(parameter.low) + 1)
            else:
                return None
        return prod(sizes)


@dataclass
class EvidenceGateResult:
    proposal: Optional[StrategyProposal]
    rejected_fields: List[Dict[str, Any]] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


class EvidenceGate:
    """Require reproducible evidence before an LLM changes live search policy."""

    PERFORMANCE_FIELDS = (
        "requested_strategy",
        "requested_sampler",
        "sampler_config",
        "search_space",
        "candidate_proposals",
    )

    def __init__(
        self,
        *,
        min_same_fidelity_trials: int = 6,
        min_confidence: float = 0.65,
        max_probe_candidates: int = 3,
    ) -> None:
        self.min_same_fidelity_trials = max(int(min_same_fidelity_trials), 1)
        self.min_confidence = float(min_confidence)
        self.max_probe_candidates = max(int(max_probe_candidates), 1)

    def review(
        self,
        study: HPOStudy,
        feedback: Dict[str, Any],
        proposal: Optional[StrategyProposal],
        *,
        compatible_history_count: Optional[Callable[[Dict[str, Any]], int]] = None,
    ) -> EvidenceGateResult:
        if proposal is None:
            return EvidenceGateResult(None)

        comparable_rung = feedback.get("comparable_rung")
        rung_summary = next(
            (
                item for item in feedback.get("rung_summaries") or []
                if item.get("rung") == comparable_rung
            ),
            None,
        )
        comparable_count = int((
            (rung_summary or {}).get("completed")
            if comparable_rung is not None
            else feedback.get("completed_trials")
        ) or 0)
        failure_driven = (
            int(feedback.get("failed_trials") or 0) >= 2
            and float(feedback.get("failure_rate") or 0.0) >= 0.5
        )
        candidate_confidences = []
        for item in proposal.candidate_proposals:
            if not isinstance(item, dict):
                continue
            value = item.get("confidence")
            if (
                value is not None
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and isfinite(float(value))
                and 0 <= float(value) <= 1
            ):
                candidate_confidences.append(float(value))
        confidence = proposal.confidence
        invalid_proposal_confidence = (
            confidence is not None
            and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1
            )
        )
        if invalid_proposal_confidence:
            confidence = None
        if confidence is None and candidate_confidences:
            confidence = min(candidate_confidences)
        targeted_probe = (
            proposal.requested_sampler == "agent_proposal"
            and 0 < len(proposal.candidate_proposals) <= self.max_probe_candidates
        )
        rejected: List[Dict[str, Any]] = []
        blocked = set()

        def reject(field_name: str, reason: str) -> None:
            if field_name in blocked:
                return
            blocked.add(field_name)
            rejected.append({"field": field_name, "reason": reason})

        has_mutation = any(
            getattr(proposal, name) not in (None, {}, [])
            for name in self.PERFORMANCE_FIELDS
        )
        if has_mutation and invalid_proposal_confidence:
            for name in self.PERFORMANCE_FIELDS:
                if getattr(proposal, name) not in (None, {}, []):
                    reject(name, "LLM confidence must be a finite number in [0, 1]")
        elif has_mutation and not failure_driven:
            if confidence is None or confidence < self.min_confidence:
                for name in self.PERFORMANCE_FIELDS:
                    if getattr(proposal, name) not in (None, {}, []):
                        reject(name, f"LLM confidence must be at least {self.min_confidence}")
            if comparable_count < self.min_same_fidelity_trials:
                for name in ("requested_strategy", "requested_sampler", "sampler_config"):
                    if getattr(proposal, name) not in (None, {}, []):
                        if targeted_probe and name == "requested_sampler":
                            continue
                        reject(
                            name,
                            "performance-driven policy changes require at least "
                            f"{self.min_same_fidelity_trials} completed same-fidelity Trials",
                        )
                if proposal.search_space is not None:
                    reject(
                        "search_space",
                        "search-space changes require at least "
                        f"{self.min_same_fidelity_trials} completed same-fidelity Trials",
                    )
                if proposal.candidate_proposals and not targeted_probe:
                    reject(
                        "candidate_proposals",
                        "candidate proposals without a bounded agent_proposal probe need "
                        "the normal evidence threshold",
                    )

        if proposal.search_space is not None and "search_space" not in blocked:
            old_parameters = {
                item.name: item.to_dict() for item in study.search_space.parameters
            }
            new_parameters = {
                item.get("name"): item
                for item in proposal.search_space.get("parameters") or []
                if isinstance(item, dict) and item.get("name")
            }
            changed = [
                name for name in set(old_parameters) | set(new_parameters)
                if old_parameters.get(name) != new_parameters.get(name)
            ]
            observations = feedback.get("parameter_observations") or {}
            unsupported = [
                name for name in changed
                if len((observations.get(name) or {}).get("unique_values") or []) < 2
            ]
            if unsupported and not failure_driven:
                reject(
                    "search_space",
                    "changed parameters lack two distinct same-fidelity observations: "
                    + ", ".join(sorted(unsupported)),
                )
            selected_sampler = proposal.requested_sampler or study.candidate_strategy or study.sampler_strategy
            if (
                selected_sampler == "tpe"
                and compatible_history_count is not None
                and not failure_driven
            ):
                retained = compatible_history_count(proposal.search_space)
                startup = int((proposal.sampler_config or study.sampler_config).get("n_startup_trials", 3))
                if retained < startup:
                    reject(
                        "search_space",
                        f"search-space change retains {retained} TPE observations; {startup} required",
                    )

        if targeted_probe and "candidate_proposals" not in blocked:
            invalid_confidence = any(
                not isinstance(item, dict)
                or isinstance(item.get("confidence"), bool)
                or not isinstance(item.get("confidence"), (int, float))
                or not isfinite(float(item["confidence"]))
                or not 0 <= float(item["confidence"]) <= 1
                or float(item["confidence"]) < self.min_confidence
                for item in proposal.candidate_proposals
            )
            if invalid_confidence and not failure_driven:
                reject(
                    "candidate_proposals",
                    f"every targeted probe requires confidence >= {self.min_confidence}",
                )
                reject("requested_sampler", "agent_proposal has no evidence-approved candidates")

        value = proposal.to_dict()
        for field_name in blocked:
            value[field_name] = {} if field_name == "sampler_config" else [] if field_name == "candidate_proposals" else None
        if not any(value.get(name) not in (None, {}, []) for name in self.PERFORMANCE_FIELDS):
            value["action"] = "keep_strategy"
        value["reason_codes"] = list(dict.fromkeys([
            *(value.get("reason_codes") or []),
            *(["evidence_gate_rejected_fields"] if rejected else []),
        ]))
        return EvidenceGateResult(
            StrategyProposal.from_dict(value, preserve_audit_fields=True),
            rejected_fields=rejected,
            reason_codes=["evidence_gate_rejected_fields"] if rejected else [],
            evidence={
                "comparable_rung": comparable_rung,
                "same_fidelity_completed_trials": comparable_count,
                "minimum_same_fidelity_trials": self.min_same_fidelity_trials,
                "proposal_confidence": confidence,
                "minimum_confidence": self.min_confidence,
                "failure_driven_exception": failure_driven,
                "targeted_probe_exception": targeted_probe,
            },
        )


class StrategyDecisionPolicy:
    """Apply only proposal fields that pass deterministic service-layer validation."""

    ACTIONS = {
        "keep_strategy",
        "refine_search_space",
        "expand_search_space",
        "switch_strategy",
        "adjust_budget",
    }

    def __init__(self, planning_policy: Optional[HPOPlanningPolicy] = None) -> None:
        self.planning_policy = planning_policy or HPOPlanningPolicy()

    def review(
        self,
        proposal: Optional[StrategyProposal],
        *,
        base_strategy: str,
        base_search_space: SearchSpace,
        base_budgets: List[TrialBudget],
        hard_max_training_runs: int,
        objectives: List[Objective],
        available_strategies: List[str],
        validate_plan: Callable[..., None],
        confirmation_budget: Optional[TrialBudget] = None,
    ) -> StrategyDecisionRecord:
        strategy = base_strategy
        search_space = base_search_space
        budgets = base_budgets
        max_training_runs = hard_max_training_runs
        accepted: List[str] = []
        rejected: List[Dict[str, Any]] = []

        def reject(field: str, reason: str) -> None:
            rejected.append({"field": field, "reason": reason})

        def validate(
            next_strategy: str,
            next_space: SearchSpace,
            next_budgets: List[TrialBudget],
            next_runs: int,
        ) -> None:
            if confirmation_budget is not None:
                if not next_budgets or budget_key(next_budgets[-1].to_dict()) != budget_key(
                    confirmation_budget.to_dict()
                ):
                    raise ValueError(
                        "final budget differs from the frozen Campaign confirmation budget"
                    )
            validate_plan(
                next_space,
                objectives,
                next_budgets,
                strategy=next_strategy,
                max_trials=next_runs,
                max_training_runs=next_runs,
            )

        validate(strategy, search_space, budgets, max_training_runs)
        if proposal is None:
            return self._record(
                "no_proposal", None, strategy, search_space, budgets, max_training_runs,
                accepted, rejected, ["base_plan_adopted"],
            )
        if proposal.action not in self.ACTIONS:
            reject("action", f"unsupported proposal action: {proposal.action}")
            return self._record(
                "rejected", proposal, strategy, search_space, budgets,
                max_training_runs, accepted, rejected, ["proposal_rejected"],
            )
        if proposal.confidence is not None and (
            isinstance(proposal.confidence, bool)
            or not isinstance(proposal.confidence, (int, float))
            or not 0 <= proposal.confidence <= 1
        ):
            reject("confidence", "confidence must be a number in [0, 1]")

        # Validate an interdependent proposal as one plan before falling back to
        # field-by-field approval. Strategy and search-space changes often only
        # become valid when applied together.
        try:
            candidate_space = search_space
            if proposal.search_space is not None:
                candidate_space = SearchSpace(
                    [SearchParameter(**item) for item in proposal.search_space.get("parameters", [])],
                    list(proposal.search_space.get("constraints") or []),
                )
            candidate_budgets = (
                [TrialBudget(**item) for item in proposal.budgets]
                if proposal.budgets is not None else budgets
            )
            candidate_runs = max_training_runs
            if proposal.max_training_runs is not None:
                if isinstance(proposal.max_training_runs, bool) or not isinstance(proposal.max_training_runs, int):
                    raise ValueError("proposed max_training_runs must be an integer")
                if proposal.max_training_runs > hard_max_training_runs:
                    raise ValueError("proposed max_training_runs exceeds the hard request limit")
                candidate_runs = proposal.max_training_runs
            candidate_strategy = (
                self.planning_policy.select_strategy(
                    str(proposal.requested_strategy),
                    candidate_space,
                    candidate_budgets,
                    candidate_runs,
                    available_strategies,
                )
                if proposal.requested_strategy is not None else strategy
            )
            validate(candidate_strategy, candidate_space, candidate_budgets, candidate_runs)
            accepted = [
                field for field, value in (
                    ("search_space", proposal.search_space),
                    ("budgets", proposal.budgets),
                    ("max_training_runs", proposal.max_training_runs),
                    ("requested_strategy", proposal.requested_strategy),
                )
                if value is not None
            ]
            decision = "approved_with_changes" if rejected else "approved"
            reasons = ["proposal_partially_approved" if rejected else "proposal_approved"]
            return self._record(
                decision,
                proposal,
                candidate_strategy,
                candidate_space,
                candidate_budgets,
                candidate_runs,
                accepted,
                rejected,
                reasons,
            )
        except Exception:
            pass

        if proposal.search_space is not None:
            try:
                candidate = SearchSpace(
                    [SearchParameter(**item) for item in proposal.search_space.get("parameters", [])],
                    list(proposal.search_space.get("constraints") or []),
                )
                validate(strategy, candidate, budgets, max_training_runs)
                search_space = candidate
                accepted.append("search_space")
            except Exception as exc:
                reject("search_space", str(exc))

        if proposal.budgets is not None:
            try:
                candidate_budgets = [TrialBudget(**item) for item in proposal.budgets]
                validate(strategy, search_space, candidate_budgets, max_training_runs)
                budgets = candidate_budgets
                accepted.append("budgets")
            except Exception as exc:
                reject("budgets", str(exc))

        if proposal.max_training_runs is not None:
            try:
                if isinstance(proposal.max_training_runs, bool) or not isinstance(proposal.max_training_runs, int):
                    raise ValueError("proposed max_training_runs must be an integer")
                candidate_runs = proposal.max_training_runs
                if candidate_runs > hard_max_training_runs:
                    raise ValueError("proposed max_training_runs exceeds the hard request limit")
                validate(strategy, search_space, budgets, candidate_runs)
                max_training_runs = candidate_runs
                accepted.append("max_training_runs")
            except Exception as exc:
                reject("max_training_runs", str(exc))

        if proposal.requested_strategy is not None:
            try:
                candidate_strategy = self.planning_policy.select_strategy(
                    str(proposal.requested_strategy),
                    search_space,
                    budgets,
                    max_training_runs,
                    available_strategies,
                )
                validate(candidate_strategy, search_space, budgets, max_training_runs)
                strategy = candidate_strategy
                accepted.append("requested_strategy")
            except Exception as exc:
                reject("requested_strategy", str(exc))

        decision = "approved"
        reasons = ["proposal_approved"]
        if rejected:
            decision = "approved_with_changes" if accepted else "rejected"
            reasons = ["proposal_partially_approved" if accepted else "proposal_rejected"]
        return self._record(
            decision, proposal, strategy, search_space, budgets, max_training_runs,
            accepted, rejected, reasons,
        )

    @staticmethod
    def _record(
        decision: str,
        proposal: Optional[StrategyProposal],
        strategy: str,
        search_space: SearchSpace,
        budgets: List[TrialBudget],
        max_training_runs: int,
        accepted: List[str],
        rejected: List[Dict[str, Any]],
        reasons: List[str],
    ) -> StrategyDecisionRecord:
        return StrategyDecisionRecord(
            decision=decision,
            proposal_id=proposal.proposal_id if proposal else None,
            proposal=proposal.to_dict() if proposal else None,
            adopted_strategy=strategy,
            adopted_search_space=search_space.to_dict(),
            adopted_budgets=[budget.to_dict() for budget in budgets],
            adopted_max_training_runs=max_training_runs,
            accepted_fields=accepted,
            rejected_fields=rejected,
            reason_codes=reasons,
        )


class OptimizationPlanDecisionPolicy:
    """Review independent sampler, pruner, and allocation changes."""

    def __init__(self, planning_policy: Optional[HPOPlanningPolicy] = None) -> None:
        self.planning_policy = planning_policy or HPOPlanningPolicy()
        self.strategy_policy = StrategyDecisionPolicy(self.planning_policy)

    def review(
        self,
        proposal: Optional[StrategyProposal],
        *,
        base_sampler: str,
        base_pruner: str,
        base_search_space: SearchSpace,
        base_budgets: List[TrialBudget],
        hard_max_training_runs: int,
        objectives: List[Objective],
        available_strategies: List[str],
        validate_plan: Callable[..., None],
        base_initial_trial_count: Optional[int] = None,
        base_promotion_limits: Optional[List[int]] = None,
        base_reduction_factor: int = 3,
        confirmation_budget: Optional[TrialBudget] = None,
    ) -> StrategyDecisionRecord:
        sampler_request = None
        if proposal is not None:
            sampler_request = proposal.requested_sampler
            if sampler_request is None and proposal.requested_strategy not in {
                None,
                "auto",
                "successive_halving",
            }:
                sampler_request = proposal.requested_strategy

        common_proposal = None
        if proposal is not None:
            common_proposal = StrategyProposal(
                action=proposal.action,
                requested_strategy=sampler_request,
                search_space=proposal.search_space,
                budgets=proposal.budgets,
                max_training_runs=proposal.max_training_runs,
                reason_codes=proposal.reason_codes,
                evidence=proposal.evidence,
                expected_effect=proposal.expected_effect,
                confidence=proposal.confidence,
            )
            common_proposal.proposal_id = proposal.proposal_id
            common_proposal.created_at = proposal.created_at

        validation_pruner = base_pruner
        if proposal is not None:
            proposed_pruner = proposal.requested_pruner
            if proposed_pruner is None and proposal.requested_strategy == "successive_halving":
                proposed_pruner = "successive_halving"
            if proposed_pruner in {"none", "successive_halving"}:
                validation_pruner = str(proposed_pruner)

        def validate_sampler_plan(
            search_space: SearchSpace,
            plan_objectives: List[Objective],
            plan_budgets: List[TrialBudget],
            *,
            strategy: str,
            **kwargs: Any,
        ) -> None:
            effective_pruner = (
                validation_pruner if plan_budgets != base_budgets else base_pruner
            )
            validate_plan(
                search_space,
                plan_objectives,
                plan_budgets,
                strategy=(
                    "successive_halving"
                    if effective_pruner == "successive_halving"
                    else strategy
                ),
                sampler_strategy=strategy,
                pruner_strategy=effective_pruner,
                **kwargs,
            )

        decision = self.strategy_policy.review(
            common_proposal,
            base_strategy=base_sampler,
            base_search_space=base_search_space,
            base_budgets=base_budgets,
            hard_max_training_runs=hard_max_training_runs,
            objectives=objectives,
            available_strategies=[
                item for item in available_strategies
                if item != "successive_halving"
            ],
            validate_plan=validate_sampler_plan,
            confirmation_budget=confirmation_budget,
        )
        if proposal is not None:
            decision.proposal_id = proposal.proposal_id
            decision.proposal = proposal.to_dict()

        accepted = list(decision.accepted_fields)
        rejected = list(decision.rejected_fields)
        if proposal is not None and proposal.requested_sampler is not None:
            accepted = [
                "requested_sampler" if item == "requested_strategy" else item
                for item in accepted
            ]

        sampler = decision.adopted_strategy
        pruner = base_pruner
        initial_count = min(
            int(base_initial_trial_count or decision.adopted_max_training_runs),
            decision.adopted_max_training_runs,
        )
        promotion_limits = list(base_promotion_limits or [])
        reduction_factor = int(base_reduction_factor)

        def reject(field: str, reason: str) -> None:
            rejected.append({"field": field, "reason": reason})

        def validate_components(
            next_pruner: str,
            next_initial: int,
            next_promotions: List[int],
            next_reduction: int,
        ) -> None:
            legacy_strategy = (
                "successive_halving"
                if next_pruner == "successive_halving"
                else sampler
            )
            validate_plan(
                search_space_from_record(decision.adopted_search_space),
                objectives,
                [TrialBudget(**item) for item in decision.adopted_budgets],
                strategy=legacy_strategy,
                sampler_strategy=sampler,
                pruner_strategy=next_pruner,
                max_trials=decision.adopted_max_training_runs,
                initial_trial_count=next_initial,
                promotion_limits=next_promotions,
                max_training_runs=decision.adopted_max_training_runs,
                reduction_factor=next_reduction,
            )

        if proposal is not None:
            requested_pruner = proposal.requested_pruner
            pruner_field = "requested_pruner"
            if requested_pruner is None and proposal.requested_strategy == "successive_halving":
                requested_pruner = "successive_halving"
                pruner_field = "requested_strategy"

            allocation_fields = [
                field for field, value in (
                    (pruner_field, requested_pruner),
                    ("initial_trial_count", proposal.initial_trial_count),
                    ("promotion_limits", proposal.promotion_limits),
                    ("reduction_factor", proposal.reduction_factor),
                )
                if value is not None
            ]
            allocation_applied = False
            if allocation_fields:
                try:
                    candidate_pruner = str(requested_pruner or pruner)
                    candidate_initial = int(
                        proposal.initial_trial_count
                        if proposal.initial_trial_count is not None else initial_count
                    )
                    candidate_promotions = (
                        [int(item) for item in proposal.promotion_limits]
                        if proposal.promotion_limits is not None
                        else list(promotion_limits)
                    )
                    if candidate_pruner == "none" and proposal.promotion_limits is None:
                        candidate_promotions = []
                    candidate_reduction = int(
                        proposal.reduction_factor
                        if proposal.reduction_factor is not None else reduction_factor
                    )
                    validate_components(
                        candidate_pruner,
                        candidate_initial,
                        candidate_promotions,
                        candidate_reduction,
                    )
                    pruner = candidate_pruner
                    initial_count = candidate_initial
                    promotion_limits = candidate_promotions
                    reduction_factor = candidate_reduction
                    accepted.extend(allocation_fields)
                    allocation_applied = True
                except Exception:
                    # An invalid combined plan falls back to independent field review
                    # so harmless parts of an LLM proposal can still be retained.
                    pass

            if not allocation_applied:
                if requested_pruner is not None:
                    try:
                        candidate_pruner = str(requested_pruner)
                        candidate_promotions = promotion_limits
                        if candidate_pruner == "none":
                            candidate_promotions = []
                        validate_components(
                            candidate_pruner,
                            initial_count,
                            candidate_promotions,
                            reduction_factor,
                        )
                        pruner = candidate_pruner
                        promotion_limits = candidate_promotions
                        if pruner_field not in accepted:
                            accepted.append(pruner_field)
                    except Exception as exc:
                        reject(pruner_field, str(exc))

                for field, raw_value in (
                    ("reduction_factor", proposal.reduction_factor),
                    ("initial_trial_count", proposal.initial_trial_count),
                    ("promotion_limits", proposal.promotion_limits),
                ):
                    if raw_value is None:
                        continue
                    try:
                        next_reduction = (
                            int(raw_value) if field == "reduction_factor" else reduction_factor
                        )
                        next_initial = (
                            int(raw_value) if field == "initial_trial_count" else initial_count
                        )
                        next_promotions = (
                            [int(item) for item in raw_value]
                            if field == "promotion_limits" else promotion_limits
                        )
                        validate_components(
                            pruner,
                            next_initial,
                            next_promotions,
                            next_reduction,
                        )
                        reduction_factor = next_reduction
                        initial_count = next_initial
                        promotion_limits = next_promotions
                        accepted.append(field)
                    except Exception as exc:
                        reject(field, str(exc))
        decision.adopted_sampler = sampler
        decision.adopted_pruner = pruner
        decision.adopted_strategy = (
            "successive_halving" if pruner == "successive_halving" else sampler
        )
        decision.adopted_initial_trial_count = initial_count
        decision.adopted_promotion_limits = promotion_limits
        decision.adopted_reduction_factor = reduction_factor
        decision.accepted_fields = list(dict.fromkeys(accepted))
        decision.rejected_fields = rejected
        if rejected:
            decision.decision = (
                "approved_with_changes" if decision.accepted_fields else "rejected"
            )
            decision.reason_codes = [
                "proposal_partially_approved"
                if decision.accepted_fields else "proposal_rejected"
            ]
        return decision


def search_space_from_record(data: Dict[str, Any]) -> SearchSpace:
    return SearchSpace(
        [SearchParameter(**item) for item in data.get("parameters") or []],
        list(data.get("constraints") or []),
    )

class EarlyStoppingPolicy:
    def __init__(self, patience: int = 3, min_improvement: float = 0.0) -> None:
        self.patience = patience
        self.min_improvement = min_improvement

    def evaluate(
        self,
        intermediate_metrics: List[Dict[str, Any]],
        *,
        metric: str,
        mode: str,
        best_known: Optional[float] = None,
    ) -> StopDecision:
        values = [
            item.get(metric) for item in intermediate_metrics
            if isinstance(item.get(metric), (int, float))
        ]
        if any(_invalid(value) for value in values):
            return StopDecision(True, "invalid_metric")
        if len(values) <= self.patience:
            return StopDecision(False)

        recent = values[-(self.patience + 1):]
        start, end = recent[0], recent[-1]
        improvement = start - end if mode == "min" else end - start
        if improvement <= self.min_improvement:
            return StopDecision(True, "no_improvement")
        if best_known is not None:
            current = min(recent) if mode == "min" else max(recent)
            worse = current > best_known if mode == "min" else current < best_known
            if worse and improvement <= self.min_improvement:
                return StopDecision(True, "unlikely_to_beat_best")
        return StopDecision(False)


def _invalid(value: Any) -> bool:
    return isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")})
