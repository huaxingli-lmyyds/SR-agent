"""Runner contracts shared by training and evaluation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from agent.core.contracts import OperationResult


class RunnerAdapter(Protocol):
    """Execution boundary for a training framework or external runtime."""

    runner: str

    def run_training(
        self, config_path: str, overrides: dict[str, Any]
    ) -> dict[str, Any]: ...

    def run_evaluation(
        self,
        config_path: str,
        model_path: str | None,
        data_path: str | None,
        overrides: dict[str, Any],
    ) -> dict[str, Any]: ...

    def normalize_training_result(
        self, raw: dict[str, Any]
    ) -> OperationResult: ...

    def normalize_evaluation_result(
        self, raw: dict[str, Any]
    ) -> OperationResult: ...


class TrainingResultCollector(Protocol):
    """Optional runner capability; the core runner protocol needs no collector."""

    def collect_training_result(
        self,
        raw: dict[str, Any],
        output_folder: Path | None,
        experiment_dir: Path,
    ) -> dict[str, Any]: ...


def validate_runner_result(raw: Any, stage: str) -> dict[str, Any]:
    """Reject malformed runner responses before they become successful records."""
    if not isinstance(raw, dict) or raw.get("status") not in (
        "success",
        "failed",
    ):
        raise ValueError(
            f"{stage} runner must return a dict with status 'success' or 'failed'"
        )
    if raw.get("metrics") is not None and not isinstance(raw["metrics"], dict):
        raise ValueError(f"{stage} runner metrics must be a dict")
    return raw


def collect_training_result(
    adapter: RunnerAdapter,
    raw: dict[str, Any],
    output_folder: Path | None,
    experiment_dir: Path,
) -> dict[str, Any]:
    """Use runner-specific artifact discovery when available."""

    collector = getattr(adapter, "collect_training_result", None)
    if callable(collector):
        return dict(collector(raw, output_folder, experiment_dir))
    result = dict(raw)
    result.setdefault(
        "output_folder", str(output_folder) if output_folder else None
    )
    result["metrics"] = dict(result.get("metrics") or {})
    if result.get("valid_error_rate") is not None:
        result["metrics"].setdefault(
            "valid_error_rate", result["valid_error_rate"]
        )
    result.setdefault("model_paths", [])
    return result


def validate_runner_compatibility(
    adapter: RunnerAdapter,
    *,
    implementation: str | None = None,
    model_family: str | None = None,
) -> None:
    """Reject explicitly unsupported combinations while preserving legacy adapters."""

    for value, attribute, label in (
        (implementation, "supported_implementations", "implementation"),
        (model_family, "supported_model_families", "model family"),
    ):
        declared = getattr(adapter, attribute, None)
        supported = (
            {declared} if isinstance(declared, str) else set(declared or {"*"})
        )
        if value and "*" not in supported and value not in supported:
            raise ValueError(
                f"runner '{adapter.runner}' does not support {label} '{value}'"
            )


__all__ = [
    "RunnerAdapter",
    "TrainingResultCollector",
    "collect_training_result",
    "validate_runner_compatibility",
    "validate_runner_result",
]
