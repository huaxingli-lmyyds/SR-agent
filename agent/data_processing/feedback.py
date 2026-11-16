"""Small, scoped evidence envelopes; feedback is not an execution command."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def export_hpo_feedback(study, trials, handoff):
    """Publish observations for later data tasks, without claiming data causality."""
    completed = [
        trial for trial in trials if trial.status in {"completed", "promoted"}
    ]
    metric = study.objectives[0].metric
    eligible = [
        trial
        for trial in completed
        if isinstance(trial.metrics.get(metric), (int, float))
        and math.isfinite(trial.metrics[metric])
    ]
    if not eligible:
        return None
    rung = max(trial.rung for trial in eligible)
    eligible = [trial for trial in eligible if trial.rung == rung]
    best = sorted(
        eligible,
        key=lambda trial: trial.metrics[metric],
        reverse=study.objectives[0].mode == "max",
    )[0]
    stopped = sum(
        bool(trial.cost.get("training_control")) and trial.status == "stopped"
        for trial in trials
    )
    return {
        "dataset_uri": handoff["consumer_uri"],
        "dataset_version": handoff.get("dataset_version"),
        "protocol_sha256": (handoff.get("prepared_data") or {}).get(
            "protocol_sha256"
        ),
        "study_id": study.study_id,
        "evaluation_role": "hpo_validation",
        "trial_count": len(completed),
        "metrics": {
            key: best.metrics[key]
            for key in ("eer", "min_dcf")
            if key in best.metrics
        },
        "diagnostics": [
            f"{len(completed)} completed/promoted trials; {stopped} stopped by internal validation rules.",
            f"Best comparable observation is from rung {rung}; these results do not establish a data-quality cause.",
        ],
    }


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
    fingerprint = value.get("protocol_sha256")
    if fingerprint is not None:
        from hashlib import sha256

        protocol = Path(dataset_uri) / "prepared_protocol.json"
        if (
            not isinstance(fingerprint, str)
            or not protocol.is_file()
            or sha256(protocol.read_bytes()).hexdigest() != fingerprint
        ):
            return rejected("dataset_protocol_mismatch")
    if value.get("evaluation_role") != "hpo_validation":
        return rejected("not_hpo_validation")
    study = value.get("study_id")
    count = value.get("trial_count")
    if not isinstance(study, str) or not study or len(study) > 128:
        return rejected("missing_or_invalid_study_id")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return rejected("missing_trial_evidence")
    diagnostics = value.get("diagnostics", [])
    if not isinstance(diagnostics, list) or any(
        not isinstance(x, str) for x in diagnostics
    ):
        return rejected("invalid_diagnostics")
    # Bound individual fields, not a raw JSON string that may lose provenance.
    evidence = {
        "study_id": study,
        "dataset_uri": str(Path(source).resolve()),
        "evaluation_role": "hpo_validation",
        "trial_count": count,
        "diagnostics": [x[:240] for x in diagnostics[:5]],
        "diagnostics_omitted": max(0, len(diagnostics) - 5),
        "protocol_sha256": fingerprint,
    }
    metrics = value.get("metrics")
    if isinstance(metrics, dict):
        evidence["metrics"] = {
            key: float(metrics[key])
            for key in ("eer", "min_dcf")
            if isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics[key], bool)
            and math.isfinite(metrics[key])
        }
    return {
        "status": "accepted",
        "reason": "scoped_validation_evidence",
        "evidence": evidence,
    }


def consultation_opportunity(state: dict[str, Any]) -> dict[str, Any]:
    mode = state.get("consultation_mode", "always")
    if mode not in {"always", "evidence"}:
        raise ValueError("consultation_mode must be always or evidence")
    profile = state.get("profile") or {}
    issues = profile.get("issues") or []
    actionable = sum(
        bool(x.get("suggested_operation"))
        for x in issues
        if isinstance(x, dict)
    )
    feedback = (state.get("hpo_feedback") or {}).get("status") == "accepted"
    requested = bool(state.get("requested_operations"))
    consult = mode == "always" or actionable > 0 or feedback or requested
    return {
        "should_consult": consult,
        "mode": mode,
        "actionable_issue_count": actionable,
        "has_validation_feedback": feedback,
        "has_explicit_operations": requested,
        "reason": "evidence_available" if consult else "no_actionable_evidence",
    }
