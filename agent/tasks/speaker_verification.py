"""Speaker verification task metrics."""

from dataclasses import dataclass
from typing import Any, Dict


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


__all__ = ["SpeakerVerificationTaskAdapter"]
