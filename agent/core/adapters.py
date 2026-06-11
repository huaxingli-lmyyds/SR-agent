"""Adapter contracts and the initial SpeechBrain ECAPA implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol

from .contracts import Artifact, OperationResult


class TaskAdapter(Protocol):
    task_type: str
    primary_metric: str
    metric_mode: str

    def validate_metrics(self, metrics: Dict[str, Any]) -> None: ...


class ModelAdapter(Protocol):
    model_family: str
    implementation: str

    def validate_config(self, config: Dict[str, Any]) -> None: ...


class RunnerAdapter(Protocol):
    runner: str

    def normalize_training_result(self, raw: Dict[str, Any]) -> OperationResult: ...
    def normalize_evaluation_result(self, raw: Dict[str, Any]) -> OperationResult: ...


@dataclass
class SpeakerVerificationTaskAdapter:
    task_type: str = "speaker_verification"
    primary_metric: str = "eer"
    metric_mode: str = "min"

    def validate_metrics(self, metrics: Dict[str, Any]) -> None:
        for key in ("eer", "min_dcf"):
            value = metrics.get(key)
            if value is not None and not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric")


@dataclass
class SpeechBrainEcapaAdapter:
    model_family: str = "ecapa_tdnn"
    implementation: str = "speechbrain"

    def validate_config(self, config: Dict[str, Any]) -> None:
        required = ("embedding_model", "classifier", "output_folder")
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"missing ECAPA config fields: {missing}")


@dataclass
class SpeechBrainRunnerAdapter:
    runner: str = "speechbrain"

    def normalize_training_result(self, raw: Dict[str, Any]) -> OperationResult:
        output_folder = raw.get("output_folder")
        artifacts = []
        for path in raw.get("model_paths") or []:
            artifacts.append(Artifact("checkpoint", Path(path).name, str(path)))
        if raw.get("train_log_path"):
            artifacts.append(Artifact("log", "training_log", str(raw["train_log_path"])))
        return OperationResult(
            status=raw.get("status", "failed"),
            stage="training",
            metrics={"validation": raw.get("metrics") or {}},
            artifacts=artifacts,
            extensions={
                "speechbrain": {
                    "output_folder": output_folder,
                    "epoch_data": raw.get("epoch_data") or [],
                    "final_metrics": raw.get("final_metrics") or {},
                }
            },
            error=raw.get("error"),
        )

    def normalize_evaluation_result(self, raw: Dict[str, Any]) -> OperationResult:
        artifacts = []
        for artifact_type, key, name in (
            ("log", "evaluation_log_path", "evaluation_log"),
            ("predictions", "scores_path", "scores"),
        ):
            if raw.get(key):
                artifacts.append(Artifact(artifact_type, name, str(raw[key])))
        return OperationResult(
            status=raw.get("status", "failed"),
            stage="evaluation",
            metrics={"test": raw.get("metrics") or {}},
            artifacts=artifacts,
            extensions={"speechbrain": {"output_folder": raw.get("output_folder")}},
            error=raw.get("error"),
        )


TASK_ADAPTERS = {"speaker_verification": SpeakerVerificationTaskAdapter()}
MODEL_ADAPTERS = {"ecapa_tdnn": SpeechBrainEcapaAdapter()}
RUNNER_ADAPTERS = {"speechbrain": SpeechBrainRunnerAdapter()}


def register_task_adapter(adapter: TaskAdapter) -> None:
    TASK_ADAPTERS[adapter.task_type] = adapter


def register_model_adapter(adapter: ModelAdapter) -> None:
    MODEL_ADAPTERS[adapter.model_family] = adapter


def register_runner_adapter(adapter: RunnerAdapter) -> None:
    RUNNER_ADAPTERS[adapter.runner] = adapter
