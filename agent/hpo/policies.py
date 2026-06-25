"""Simple model-agnostic early stopping policies."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Callable, Dict, List, Optional

from .contracts import (
    Objective,
    SearchParameter,
    SearchSpace,
    StrategyDecisionRecord,
    StrategyProposal,
    TrialBudget,
)


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
