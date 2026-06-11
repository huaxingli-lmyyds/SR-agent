"""Simple model-agnostic early stopping policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StopDecision:
    should_stop: bool
    reason: Optional[str] = None


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
