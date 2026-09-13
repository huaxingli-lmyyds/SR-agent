"""Observation compatibility: rung labels alone do not identify training fidelity."""

import json
from typing import Any, Dict

from agent.core.metrics import is_finite_metric


def budget_key(budget: Dict[str, Any]) -> tuple:
    """Ignore display labels; omitted data_fraction means the full dataset."""
    if not isinstance(budget, dict) or not budget:
        raise ValueError("missing budget")
    epochs = budget.get("epochs")
    fraction = budget.get("data_fraction")
    fraction = 1.0 if fraction is None else fraction
    duration = budget.get("max_duration_seconds")
    if epochs is not None and (
        not is_finite_metric(epochs) or epochs <= 0 or int(epochs) != epochs
    ):
        raise ValueError("invalid epoch budget")
    if not is_finite_metric(fraction) or not 0 < fraction <= 1:
        raise ValueError("invalid data fraction")
    if duration is not None and (not is_finite_metric(duration) or duration <= 0):
        raise ValueError("invalid duration budget")
    return epochs, fraction, duration


def observation_signature(record: Dict[str, Any]) -> str:
    try:
        budget = budget_key(record.get("budget"))
    except ValueError:
        budget = record.get("budget")
    return json.dumps({
        "parameters": record.get("parameters") or {},
        "budget": budget,
        "context": (record.get("provenance") or {}).get("history_context"),
    }, sort_keys=True, default=str)


def incompatibility_reason(record, target_budget, target_context, metric):
    if not isinstance(record, dict):
        return "invalid_history_record"
    if any(not isinstance(record.get(field, {}), dict) for field in ("parameters", "metrics", "provenance")):
        return "invalid_history_record"
    if record.get("status") not in {"completed", "promoted"}:
        return "history_not_completed"
    if not is_finite_metric((record.get("metrics") or {}).get(metric)):
        return "invalid_primary_metric"
    try:
        if budget_key(record.get("budget")) != budget_key(target_budget):
            return "budget_mismatch"
    except ValueError:
        return "invalid_history_budget"
    context = (record.get("provenance") or {}).get("history_context")
    if not context:
        return "unverified_legacy_history_context"
    if context != target_context:
        return "experiment_context_mismatch"
    return None
