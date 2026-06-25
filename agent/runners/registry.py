"""Registry for pluggable runner adapters."""

from __future__ import annotations

from typing import Dict

from .contracts import RunnerAdapter


class RunnerRegistry:
    def __init__(self) -> None:
        self._adapters: Dict[str, RunnerAdapter] = {}

    @property
    def adapters(self) -> Dict[str, RunnerAdapter]:
        return self._adapters

    def register(self, adapter: RunnerAdapter) -> None:
        name = str(adapter.runner).strip()
        if not name:
            raise ValueError("runner adapter must declare a non-empty runner name")
        self._adapters[name] = adapter

    def get(self, name: str) -> RunnerAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._adapters)) or "none"
            raise ValueError(f"unknown runner adapter '{name}'; available: {available}") from exc

    def describe(self) -> Dict[str, dict]:
        return {
            name: {
                "runner": name,
                "default_evaluation_config": getattr(adapter, "default_evaluation_config", None),
                "supported_implementations": list(getattr(adapter, "supported_implementations", {"*"})),
                "supported_model_families": list(getattr(adapter, "supported_model_families", {"*"})),
                "adapter_type": type(adapter).__name__,
            }
            for name, adapter in sorted(self._adapters.items())
        }


RUNNER_REGISTRY = RunnerRegistry()
RUNNER_ADAPTERS = RUNNER_REGISTRY.adapters


def register_runner_adapter(adapter: RunnerAdapter) -> None:
    RUNNER_REGISTRY.register(adapter)


def get_runner_adapter(runner: str) -> RunnerAdapter:
    return RUNNER_REGISTRY.get(runner)


__all__ = [
    "RUNNER_ADAPTERS",
    "RUNNER_REGISTRY",
    "RunnerRegistry",
    "get_runner_adapter",
    "register_runner_adapter",
]
