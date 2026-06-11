"""Model-agnostic experiment contracts and services."""

from .contracts import Artifact, OperationRequest, OperationResult
from .experiment_service import ExperimentService
from .adapters import (
    MODEL_ADAPTERS,
    RUNNER_ADAPTERS,
    TASK_ADAPTERS,
    register_model_adapter,
    register_runner_adapter,
    register_task_adapter,
    get_model_adapter,
    get_runner_adapter,
    get_task_adapter,
)

__all__ = [
    "Artifact",
    "OperationRequest",
    "OperationResult",
    "ExperimentService",
    "TASK_ADAPTERS",
    "MODEL_ADAPTERS",
    "RUNNER_ADAPTERS",
    "register_task_adapter",
    "register_model_adapter",
    "register_runner_adapter",
    "get_task_adapter",
    "get_model_adapter",
    "get_runner_adapter",
]
