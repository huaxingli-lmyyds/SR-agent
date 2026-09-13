"""Speaker verification task metrics."""

from dataclasses import dataclass
from typing import Any, Dict

from agent.core.metrics import InvalidMetricError, require_finite_metric


@dataclass
class SpeakerVerificationTaskAdapter:
    task_type: str = "speaker_verification"
    primary_metric: str = "eer"
    metric_mode: str = "min"

    def validate_metrics(self, metrics: Dict[str, Any]) -> None:
        require_finite_metric(metrics, self.primary_metric)
        for key in ("eer", "min_dcf"):
            value = metrics.get(key)
            if value is not None:
                require_finite_metric(metrics, key)
        eer = metrics.get("eer")
        if eer is not None and not 0 <= eer <= 1:
            raise InvalidMetricError(
                f"invalid metric 'eer': expected a ratio in [0, 1], got {eer!r}"
            )
        min_dcf = metrics.get("min_dcf")
        if min_dcf is not None and min_dcf < 0:
            raise InvalidMetricError(
                f"invalid metric 'min_dcf': expected a nonnegative value, got {min_dcf!r}"
            )


__all__ = ["SpeakerVerificationTaskAdapter"]
