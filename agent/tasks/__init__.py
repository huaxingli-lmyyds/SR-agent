"""Task adapters and metric semantics."""

from .contracts import TaskAdapter
from .registry import TASK_ADAPTERS, get_task_adapter, register_task_adapter
from .speaker_verification import SpeakerVerificationTaskAdapter


if "speaker_verification" not in TASK_ADAPTERS:
    register_task_adapter(SpeakerVerificationTaskAdapter())


__all__ = [
    "TASK_ADAPTERS",
    "SpeakerVerificationTaskAdapter",
    "TaskAdapter",
    "get_task_adapter",
    "register_task_adapter",
]
