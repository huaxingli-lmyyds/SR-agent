"""Task adapters and metric semantics."""

from .contracts import (
    EvaluationMetricResolver,
    TaskAdapter,
    resolve_task_metrics,
    task_metric_metadata,
)
from .registry import TASK_ADAPTERS, get_task_adapter, register_task_adapter
from .speaker_verification import SpeakerVerificationTaskAdapter


if "speaker_verification" not in TASK_ADAPTERS:
    register_task_adapter(SpeakerVerificationTaskAdapter())


__all__ = [
    "TASK_ADAPTERS",
    "SpeakerVerificationTaskAdapter",
    "TaskAdapter",
    "EvaluationMetricResolver",
    "resolve_task_metrics",
    "task_metric_metadata",
    "get_task_adapter",
    "register_task_adapter",
]
