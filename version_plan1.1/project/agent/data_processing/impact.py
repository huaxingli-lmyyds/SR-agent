"""Deterministic optional per-operation guards, before materialization."""

from .contracts import OperationImpact, QualityPolicy


def impact_rejection(impact: OperationImpact, policy: QualityPolicy) -> str | None:
    fraction = policy.max_operation_affected_fraction
    size = policy.max_operation_output_bytes
    if fraction is None and size is None:
        return None
    if not impact.creates_new_version or impact.details.get("scan_complete") is not True:
        return "impact_unknown_or_incomplete"
    counts = (impact.scanned_samples, impact.affected_samples, impact.estimated_output_bytes)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in counts):
        return "invalid_impact_counts"
    if impact.affected_samples > impact.scanned_samples:
        return "invalid_impact_counts"
    if fraction is not None:
        if impact.scanned_samples == 0:
            return "impact_sample_count_unknown"
        if impact.affected_samples / impact.scanned_samples > fraction:
            return "affected_fraction_exceeded"
    if size is not None and impact.estimated_output_bytes > size:
        return "estimated_output_bytes_exceeded"
    return None
