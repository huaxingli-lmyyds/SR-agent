"""Shared validation for objective values, including persisted legacy results."""

from math import isfinite
from typing import Any, Mapping


class InvalidMetricError(ValueError):
    """A result cannot be used as a successful optimization observation."""


def is_finite_metric(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


def require_finite_metric(metrics: Mapping[str, Any], metric: str) -> None:
    value = metrics.get(metric)
    if not is_finite_metric(value):
        raise InvalidMetricError(
            f"invalid metric {metric!r}: expected a finite number (not bool), got {value!r}"
        )
