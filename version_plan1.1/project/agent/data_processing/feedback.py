"""Small, scoped evidence envelopes; feedback is not an execution command."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def normalize_feedback(value: Any, dataset_uri: str) -> dict[str, Any]:
    def rejected(reason: str) -> dict[str, Any]:
        return {"status": "rejected", "reason": reason}

    if value is None or value == {}:
        return {"status": "absent", "reason": "no_feedback"}
    if not isinstance(value, dict):
        return rejected("expected_object")
    source = value.get("dataset_uri")
    if not isinstance(source, str) or not source:
        return rejected("missing_dataset_uri")
    try:
        if Path(source).resolve() != Path(dataset_uri).resolve():
            return rejected("dataset_mismatch")
    except (ValueError, OSError):
        return rejected("invalid_dataset_uri")
    if value.get("evaluation_role") != "hpo_validation":
        return rejected("not_hpo_validation")
    study = value.get("study_id")
    count = value.get("trial_count")
    if not isinstance(study, str) or not study or len(study) > 128:
        return rejected("missing_or_invalid_study_id")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return rejected("missing_trial_evidence")
    diagnostics = value.get("diagnostics", [])
    if not isinstance(diagnostics, list) or any(not isinstance(x, str) for x in diagnostics):
        return rejected("invalid_diagnostics")
    # Bound individual fields, not a raw JSON string that may lose provenance.
    evidence = {
        "study_id": study, "dataset_uri": str(Path(source).resolve()),
        "evaluation_role": "hpo_validation", "trial_count": count,
        "diagnostics": [x[:240] for x in diagnostics[:5]],
        "diagnostics_omitted": max(0, len(diagnostics) - 5),
    }
    metrics = value.get("metrics")
    if isinstance(metrics, dict):
        evidence["metrics"] = {
            key: float(metrics[key]) for key in ("eer", "min_dcf")
            if isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics[key], bool) and math.isfinite(metrics[key])
        }
    return {"status": "accepted", "reason": "scoped_validation_evidence", "evidence": evidence}


def consultation_opportunity(state: dict[str, Any]) -> dict[str, Any]:
    mode = state.get("consultation_mode", "always")
    if mode not in {"always", "evidence"}:
        raise ValueError("consultation_mode must be always or evidence")
    profile = state.get("profile") or {}
    issues = profile.get("issues") or []
    actionable = sum(bool(x.get("suggested_operation")) for x in issues if isinstance(x, dict))
    feedback = (state.get("hpo_feedback") or {}).get("status") == "accepted"
    requested = bool(state.get("requested_operations"))
    consult = mode == "always" or actionable > 0 or feedback or requested
    return {"should_consult": consult, "mode": mode, "actionable_issue_count": actionable,
            "has_validation_feedback": feedback, "has_explicit_operations": requested,
            "reason": "evidence_available" if consult else "no_actionable_evidence"}
