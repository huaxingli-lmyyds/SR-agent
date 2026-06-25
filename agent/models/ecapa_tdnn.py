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
                {"name": "lr", "parameter_type": "float", "low": 1e-5, "high": 1e-2, "scale": "log"},
                {"name": "batch_size", "parameter_type": "categorical", "choices": [16, 32, 64]},
            ],
            "constraints": [],
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        if "lr" in parameters and float(parameters["lr"]) <= 0:
            raise ValueError("ECAPA parameter 'lr' must be positive")
        if "batch_size" in parameters and int(parameters["batch_size"]) <= 0:
            raise ValueError("ECAPA parameter 'batch_size' must be positive")


__all__ = ["SpeechBrainEcapaAdapter"]
