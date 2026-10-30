"""Task-level metric contracts."""

from typing import Any, Protocol


class TaskAdapter(Protocol):
    task_type: str
    primary_metric: str
    metric_mode: str

    def validate_metrics(self, metrics: dict[str, Any]) -> None: ...


class EvaluationMetricResolver(Protocol):
    """Optional task capability for scores or framework-specific metric output."""

    def resolve_evaluation_metrics(
        self, raw: dict[str, Any]
    ) -> dict[str, Any]: ...


def resolve_task_metrics(
    adapter: TaskAdapter, raw: dict[str, Any]
) -> dict[str, Any]:
    """Keep task-specific score parsing out of the generic execution tools."""
    resolver = getattr(adapter, "resolve_evaluation_metrics", None)
    metrics = resolver(raw) if callable(resolver) else raw.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise ValueError("task metric resolver must return a dict")
    return dict(metrics)


def task_metric_metadata(
    adapter: TaskAdapter, recorded: dict[str, Any]
) -> dict[str, Any]:
    """Prefer frozen semantics, falling back only to this task's own defaults."""
    return {
        "metric_protocol": recorded.get("metric_protocol")
        or getattr(adapter, "metric_protocol", None),
        "metric_units": dict(
            recorded.get("metric_units") or getattr(adapter, "metric_units", {})
        ),
    }


__all__ = [
    "TaskAdapter",
    "EvaluationMetricResolver",
    "resolve_task_metrics",
    "task_metric_metadata",
]
