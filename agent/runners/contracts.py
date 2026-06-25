"""Runner contracts shared by training and evaluation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol

from agent.core.contracts import OperationResult


class RunnerAdapter(Protocol):
    """Execution boundary for a training framework or external runtime."""

    runner: str
    default_evaluation_config: Optional[str]
    supported_implementations: Iterable[str]
    supported_model_families: Iterable[str]

    def run_training(self, config_path: str, overrides: Dict[str, Any]) -> Dict[str, Any]: ...

    def run_evaluation(
        self,
        config_path: str,
        model_path: Optional[str],
        data_path: Optional[str],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]: ...

    def collect_training_result(
        self,
        raw: Dict[str, Any],
        output_folder: Optional[Path],
        experiment_dir: Path,
    ) -> Dict[str, Any]: ...

    def normalize_training_result(self, raw: Dict[str, Any]) -> OperationResult: ...

    def normalize_evaluation_result(self, raw: Dict[str, Any]) -> OperationResult: ...


def collect_training_result(
    adapter: RunnerAdapter,
    raw: Dict[str, Any],
    output_folder: Optional[Path],
    experiment_dir: Path,
) -> Dict[str, Any]:
    """Use runner-specific artifact discovery when available."""

    collector = getattr(adapter, "collect_training_result", None)
    if callable(collector):
        return dict(collector(raw, output_folder, experiment_dir))
    result = dict(raw)
    result.setdefault("output_folder", str(output_folder) if output_folder else None)
    result.setdefault("metrics", {})
    if result.get("valid_error_rate") is not None:
        result["metrics"].setdefault("valid_error_rate", result["valid_error_rate"])
    result.setdefault("model_paths", [])
    return result


def validate_runner_compatibility(
    adapter: RunnerAdapter,
    *,
    implementation: Optional[str] = None,
    model_family: Optional[str] = None,
) -> None:
    """Reject explicitly unsupported combinations while preserving legacy adapters."""

    for value, attribute, label in (
        (implementation, "supported_implementations", "implementation"),
        (model_family, "supported_model_families", "model family"),
    ):
        supported = set(getattr(adapter, attribute, {"*"}) or {"*"})
        if value and "*" not in supported and value not in supported:
            raise ValueError(f"runner '{adapter.runner}' does not support {label} '{value}'")


__all__ = ["RunnerAdapter", "collect_training_result", "validate_runner_compatibility"]
