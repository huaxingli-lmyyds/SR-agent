"""Simple model-agnostic early stopping policies."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Dict, List, Optional

from .contracts import SearchSpace, TrialBudget


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
