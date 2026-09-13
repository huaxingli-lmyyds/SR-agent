"""Deterministic data-quality gate."""

from __future__ import annotations

from .contracts import DataProfile, QualityDecision, QualityPolicy


class DefaultQualityGate:
    """Turn profile issues into an explicit downstream training decision."""

    def __init__(self, policy: QualityPolicy | None = None) -> None:
        self.policy = policy or QualityPolicy()

    def evaluate(self, profile: DataProfile) -> QualityDecision:
        blockers = [
            issue.code
            for issue in profile.issues
            if issue.severity in {"error", "critical"}
        ]
        warnings = [
            issue.code
            for issue in profile.issues
            if issue.severity == "warning"
        ]
        status = "block" if blockers else "warn" if warnings else "pass"
        return QualityDecision(
            status=status,
            training_allowed=not blockers,
            blockers=list(dict.fromkeys(blockers)),
            warnings=list(dict.fromkeys(warnings)),
        )


__all__ = ["DefaultQualityGate"]
