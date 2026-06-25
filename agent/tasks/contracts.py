"""Task-level metric contracts."""

from typing import Any, Dict, Protocol


class TaskAdapter(Protocol):
    task_type: str
    primary_metric: str
    metric_mode: str

    def validate_metrics(self, metrics: Dict[str, Any]) -> None: ...


__all__ = ["TaskAdapter"]
