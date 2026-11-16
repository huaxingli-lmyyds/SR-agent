"""Resolve compatible task, model, and runner adapters through one boundary."""

from dataclasses import dataclass

from agent.models import (
    MODEL_ADAPTERS,
    ModelAdapter,
    SpeechBrainEcapaAdapter,
    get_model_adapter,
    register_model_adapter,
)
from agent.runners import (
    RUNNER_ADAPTERS,
    RunnerAdapter,
    get_runner_adapter,
    register_runner_adapter,
    validate_runner_compatibility,
)
from agent.tasks import (
    TASK_ADAPTERS,
    SpeakerVerificationTaskAdapter,
    TaskAdapter,
    get_task_adapter,
    register_task_adapter,
)


@dataclass(frozen=True)
class AdapterBundle:
    task: TaskAdapter
    model: ModelAdapter
    runner: RunnerAdapter


def resolve_adapter_bundle(
    task_type: str,
    model_family: str,
    implementation: str,
    runner: str,
) -> AdapterBundle:
    task_adapter = get_task_adapter(task_type)
    model_adapter = get_model_adapter(model_family)
    runner_adapter = get_runner_adapter(runner)
    validate_runner_compatibility(
        runner_adapter,
        implementation=implementation,
        model_family=model_family,
    )
    return AdapterBundle(task_adapter, model_adapter, runner_adapter)


__all__ = [
    "AdapterBundle",
    "MODEL_ADAPTERS",
    "RUNNER_ADAPTERS",
    "TASK_ADAPTERS",
    "ModelAdapter",
    "RunnerAdapter",
    "SpeechBrainEcapaAdapter",
    "SpeakerVerificationTaskAdapter",
    "TaskAdapter",
    "get_model_adapter",
    "get_runner_adapter",
    "get_task_adapter",
    "register_model_adapter",
    "register_runner_adapter",
    "register_task_adapter",
    "resolve_adapter_bundle",
]
