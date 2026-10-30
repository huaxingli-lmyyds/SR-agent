"""Pluggable model execution runners."""

from .contracts import (
    RunnerAdapter,
    TrainingResultCollector,
    collect_training_result,
    validate_runner_compatibility,
    validate_runner_result,
)
from .registry import (
    RUNNER_ADAPTERS,
    RUNNER_REGISTRY,
    RunnerRegistry,
    get_runner_adapter,
    register_runner_adapter,
)
from .speechbrain import SpeechBrainRunnerAdapter


if "speechbrain" not in RUNNER_ADAPTERS:
    register_runner_adapter(SpeechBrainRunnerAdapter())


__all__ = [
    "RUNNER_ADAPTERS",
    "RUNNER_REGISTRY",
    "RunnerAdapter",
    "TrainingResultCollector",
    "RunnerRegistry",
    "SpeechBrainRunnerAdapter",
    "collect_training_result",
    "get_runner_adapter",
    "register_runner_adapter",
    "validate_runner_compatibility",
    "validate_runner_result",
]
