"""
Reward utilities for multi-metric model selection.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple


DEFAULT_WEIGHTS = {
    "eer": 1.0,
    "min_dcf": 0.3,
}


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_reward(
    metrics: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Compute a composite reward from multiple metrics.

    Reward formula (maximize):
    R = -eer - w_dcf*min_dcf
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for key, value in weights.items():
            try:
                w[key] = float(value)
            except (TypeError, ValueError):
                continue

    eer = _to_float(metrics.get("eer"))
    if eer is None:
        return None, {}

    min_dcf = _to_float(metrics.get("min_dcf"))
    reward = -eer
    breakdown = {"eer": -eer}

    if min_dcf is not None:
        term = -w["min_dcf"] * min_dcf
        reward += term
        breakdown["min_dcf"] = term

    return reward, breakdown
