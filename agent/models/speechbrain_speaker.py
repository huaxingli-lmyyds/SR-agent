"""SpeechBrain speaker-recognition model adapters beyond ECAPA-TDNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SpeechBrainResNetAdapter:
    model_family: str = "resnet"
    implementation: str = "speechbrain"
    default_evaluation_config: str | None = "recipes/voxceleb/hparams/verification_resnet.yaml"

    def validate_config(self, config: Dict[str, Any]) -> None:
        required = ("embedding_model", "classifier", "output_folder", "compute_cost")
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"missing ResNet config fields: {missing}")

    def default_search_space(self) -> Dict[str, Any]:
        return {
            "parameters": [
                {"name": "lr", "parameter_type": "float", "low": 3e-4, "high": 3e-3, "scale": "log"},
                {"name": "batch_size", "parameter_type": "categorical", "choices": [16, 24, 32]},
                {"name": "sentence_len", "parameter_type": "categorical", "choices": [2.0, 3.0, 4.0]},
                {"name": "margin", "parameter_type": "float", "low": 0.15, "high": 0.3},
                {"name": "weight_decay", "parameter_type": "float", "low": 5e-7, "high": 2e-5, "scale": "log"},
            ],
            "constraints": [],
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        if "lr" in parameters and float(parameters["lr"]) <= 0:
            raise ValueError("ResNet parameter 'lr' must be positive")
        if "batch_size" in parameters and int(parameters["batch_size"]) <= 0:
            raise ValueError("ResNet parameter 'batch_size' must be positive")
        if "sentence_len" in parameters and float(parameters["sentence_len"]) <= 0:
            raise ValueError("ResNet parameter 'sentence_len' must be positive")
        if "margin" in parameters and float(parameters["margin"]) <= 0:
            raise ValueError("ResNet parameter 'margin' must be positive")
        if "weight_decay" in parameters and float(parameters["weight_decay"]) < 0:
            raise ValueError("ResNet parameter 'weight_decay' must be non-negative")


@dataclass
class SpeechBrainXVectorAdapter:
    model_family: str = "xvector"
    implementation: str = "speechbrain"
    default_evaluation_config: str | None = "recipes/voxceleb/hparams/verification_plda_xvector.yaml"

    def validate_config(self, config: Dict[str, Any]) -> None:
        required = ("embedding_model", "classifier", "output_folder", "lr_annealing")
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"missing x-vector config fields: {missing}")

    def default_search_space(self) -> Dict[str, Any]:
        return {
            "parameters": [
                {"name": "lr", "parameter_type": "float", "low": 1e-5, "high": 3e-3, "scale": "log"},
                {"name": "lr_final", "parameter_type": "float", "low": 1e-6, "high": 1e-3, "scale": "log"},
                {"name": "batch_size", "parameter_type": "categorical", "choices": [64, 128, 256]},
                {"name": "sentence_len", "parameter_type": "categorical", "choices": [2.0, 3.0, 4.0]},
            ],
            "constraints": [
                {"parameter": "lr_final", "operator": "lte", "value": 0.001},
            ],
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        if "lr" in parameters and float(parameters["lr"]) <= 0:
            raise ValueError("x-vector parameter 'lr' must be positive")
        if "lr_final" in parameters and float(parameters["lr_final"]) <= 0:
            raise ValueError("x-vector parameter 'lr_final' must be positive")
        if "batch_size" in parameters and int(parameters["batch_size"]) <= 0:
            raise ValueError("x-vector parameter 'batch_size' must be positive")
        if "sentence_len" in parameters and float(parameters["sentence_len"]) <= 0:
            raise ValueError("x-vector parameter 'sentence_len' must be positive")


__all__ = ["SpeechBrainResNetAdapter", "SpeechBrainXVectorAdapter"]