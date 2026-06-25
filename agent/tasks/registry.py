"""Registry for task metric adapters."""

from typing import Dict

from .contracts import TaskAdapter


TASK_ADAPTERS: Dict[str, TaskAdapter] = {}


def register_task_adapter(adapter: TaskAdapter) -> None:
    TASK_ADAPTERS[adapter.task_type] = adapter


def get_task_adapter(task_type: str) -> TaskAdapter:
    try:
        return TASK_ADAPTERS[task_type]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_ADAPTERS)) or "none"
        raise ValueError(f"unknown task adapter '{task_type}'; available: {available}") from exc


__all__ = ["TASK_ADAPTERS", "get_task_adapter", "register_task_adapter"]
