"""ECAPA-TDNN model configuration adapter."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SpeechBrainEcapaAdapter:
    model_family: str = "ecapa_tdnn"
    implementation: str = "speechbrain"
    default_evaluation_config: str | None = "verification_ecapa.yaml"

    def validate_config(self, config: Dict[str, Any]) -> None:
        required = ("embedding_model", "classifier", "output_folder")
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"missing ECAPA config fields: {missing}")

    def default_search_space(self) -> Dict[str, Any]:
        return {
            "parameters": [
                {"name": "lr", "parameter_type": "float", "low": 5e-4, "high": 3e-3, "scale": "log"},
                {"name": "batch_size", "parameter_type": "categorical", "choices": [16, 24, 32]},
                {"name": "margin", "parameter_type": "float", "low": 0.15, "high": 0.3},
                {"name": "weight_decay", "parameter_type": "float", "low": 5e-6, "high": 1e-4, "scale": "log"},
            ],
            "constraints": [],
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        if "lr" in parameters and float(parameters["lr"]) <= 0:
            raise ValueError("ECAPA parameter 'lr' must be positive")
        if "batch_size" in parameters and int(parameters["batch_size"]) <= 0:
            raise ValueError("ECAPA parameter 'batch_size' must be positive")
        if "margin" in parameters and float(parameters["margin"]) <= 0:
            raise ValueError("ECAPA parameter 'margin' must be positive")
        if "weight_decay" in parameters and float(parameters["weight_decay"]) < 0:
            raise ValueError("ECAPA parameter 'weight_decay' must be non-negative")


__all__ = ["SpeechBrainEcapaAdapter"]
