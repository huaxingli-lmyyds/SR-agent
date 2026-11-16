"""Model-agnostic experiment contracts and lazily loaded adapter services."""

from importlib import import_module

from .contracts import Artifact, OperationRequest, OperationResult
from .experiment_service import ExperimentService


_LAZY_EXPORTS = {
    "TASK_ADAPTERS": ("agent.tasks", "TASK_ADAPTERS"),
    "MODEL_ADAPTERS": ("agent.models", "MODEL_ADAPTERS"),
    "RUNNER_ADAPTERS": ("agent.runners", "RUNNER_ADAPTERS"),
    "register_task_adapter": ("agent.tasks", "register_task_adapter"),
    "register_model_adapter": ("agent.models", "register_model_adapter"),
    "register_runner_adapter": ("agent.runners", "register_runner_adapter"),
    "get_task_adapter": ("agent.tasks", "get_task_adapter"),
    "get_model_adapter": ("agent.models", "get_model_adapter"),
    "get_runner_adapter": ("agent.runners", "get_runner_adapter"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "Artifact",
    "OperationRequest",
    "OperationResult",
    "ExperimentService",
    *_LAZY_EXPORTS,
]
