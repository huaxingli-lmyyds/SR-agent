"""Immutable validation and metric protocol for HPO experiments."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from agent.utils import ConfigParser
from agent.utils.path_tool import resolve_config_path, resolve_project_path

METRIC_PROTOCOL_ID = "speaker_verification.v2.eer_ratio.mindcf_x100"
METRIC_UNITS = {
    "eer": "ratio_0_1",
    "min_dcf": "x100",
}


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _resolve_required_file(value: Any, field: str, *, config: bool = False) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"HPO requires {field}")
    path = resolve_config_path(str(value)) if config else resolve_project_path(str(value))
    if not path.is_file():
        raise ValueError(f"HPO {field} file not found: {path}")
    return path


def _pair_members(path: Path) -> tuple[set[str], set[str]]:
    utterances: set[str] = set()
    speakers: set[str] = set()
    pair_count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(
                    f"malformed validation pair at {path}:{line_number}: "
                    "expected '<label> <enrol_utt> <test_utt>'"
                )
            try:
                label = int(parts[0])
            except ValueError as exc:
                raise ValueError(
                    f"malformed validation label at {path}:{line_number}: {parts[0]!r}"
                ) from exc
            if label not in {0, 1}:
                raise ValueError(
                    f"validation label must be 0 or 1 at {path}:{line_number}"
                )
            for raw in parts[1:]:
                utterance = raw[:-4] if raw.lower().endswith(".wav") else raw
                utterance = utterance.replace("\\", "/").strip("/")
                if not utterance or "/" not in utterance:
                    raise ValueError(
                        f"invalid validation utterance at {path}:{line_number}: {raw!r}"
                    )
                utterances.add(utterance)
                speakers.add(utterance.split("/", 1)[0])
            pair_count += 1
    if pair_count == 0:
        raise ValueError(f"validation_pairs contains no pairs: {path}")
    return utterances, speakers


def _assert_exclusion_covers_validation(
    validation_pairs: Path,
    training_exclusion_pairs: Path,
) -> None:
    validation_utterances, validation_speakers = _pair_members(validation_pairs)
    excluded_utterances, excluded_speakers = _pair_members(training_exclusion_pairs)
    if not (
        validation_utterances <= excluded_utterances
        or validation_speakers <= excluded_speakers
    ):
        missing = sorted(validation_speakers - excluded_speakers)
        raise ValueError(
            "training_exclusion_pairs does not cover all validation speakers; "
            f"missing: {', '.join(missing[:10])}"
        )


def resolve_hpo_validation_protocol(
    runtime_options: Any,
    *,
    persisted_execution: dict[str, Any] | None = None,
    require_explicit: bool,
) -> dict[str, Any]:
    """Resolve and verify the immutable validation inputs used by an HPO Study."""
    runtime = dict(runtime_options or {}) if isinstance(runtime_options, dict) else {}
    persisted = dict(persisted_execution or {})
    if runtime.get("test_pairs") is not None:
        raise ValueError("test_pairs cannot be supplied to HPO; use them only after model lock")

    if require_explicit:
        verification_value = runtime.get("verification_config")
        validation_value = runtime.get("validation_pairs")
    else:
        verification_value = (
            runtime.get("verification_config")
            or persisted.get("evaluation_config_path")
        )
        validation_value = (
            runtime.get("validation_pairs")
            or persisted.get("validation_pairs_path")
        )

    verification_config = _resolve_required_file(
        verification_value, "verification_config", config=True
    )
    try:
        ConfigParser(str(verification_config)).load_config(resolve_references=True)
    except Exception as exc:
        raise ValueError(
            f"invalid HPO verification_config: {verification_config}: {exc}"
        ) from exc
    validation_pairs = _resolve_required_file(validation_value, "validation_pairs")
    # Never silently equate validation pairs with the complete training
    # exclusion protocol.  A held-out final-test protocol may contain other
    # speakers that must be excluded from training without being exposed to
    # HPO evaluation.  New Studies provide this file explicitly; resumed
    # Studies recover the already frozen path from their execution record.
    training_exclusion_value = runtime.get("training_exclusion_pairs")
    if not require_explicit:
        training_exclusion_value = (
            training_exclusion_value
            or persisted.get("training_exclusion_pairs_path")
        )
    training_exclusion_pairs = _resolve_required_file(
        training_exclusion_value, "training_exclusion_pairs"
    )
    _assert_exclusion_covers_validation(validation_pairs, training_exclusion_pairs)

    resolved = {
        **runtime,
        "verification_config": str(verification_config),
        "validation_pairs": str(validation_pairs),
        "training_exclusion_pairs": str(training_exclusion_pairs),
        "verification_config_sha256": file_sha256(verification_config),
        "validation_pairs_sha256": file_sha256(validation_pairs),
        "training_exclusion_pairs_sha256": file_sha256(training_exclusion_pairs),
        "metric_protocol": METRIC_PROTOCOL_ID,
        "metric_units": dict(METRIC_UNITS),
    }
    for field in (
        "verification_config_sha256",
        "validation_pairs_sha256",
        "training_exclusion_pairs_sha256",
    ):
        expected = persisted.get(field)
        if expected is not None and expected != resolved[field]:
            raise ValueError(f"persisted HPO validation input changed: {field}")
    persisted_protocol = persisted.get("metric_protocol")
    if persisted and persisted_protocol != METRIC_PROTOCOL_ID:
        raise ValueError(
            "legacy or incompatible HPO metric protocol cannot be resumed: "
            f"{persisted_protocol!r}"
        )
    return resolved


def metric_protocol_record() -> dict[str, Any]:
    return {
        "metric_protocol": METRIC_PROTOCOL_ID,
        "metric_units": dict(METRIC_UNITS),
    }


__all__ = [
    "METRIC_PROTOCOL_ID",
    "METRIC_UNITS",
    "file_sha256",
    "metric_protocol_record",
    "resolve_hpo_validation_protocol",
]
