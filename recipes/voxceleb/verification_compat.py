"""Verification-pair parsing helpers for VoxCeleb recipes."""

from __future__ import annotations

from typing import Optional


def normalize_utterance_id(value: str) -> str:
    """Return a VoxCeleb utterance id without the audio file suffix."""
    value = value.strip()
    if value.lower().endswith(".wav"):
        value = value[:-4]
    return value


def parse_verification_pair(
    line: str,
    *,
    source: str = "<verification_pairs>",
    line_number: int | None = None,
) -> Optional[tuple[int, str, str]]:
    """Parse one VoxCeleb verification-pair line.

    Valid lines have the form: ``label enrol_utt test_utt``. Blank lines are
    ignored. A malformed non-blank line raises a ValueError with enough context
    to diagnose the dataset or downloaded metadata file.
    """
    stripped = line.strip()
    if not stripped:
        return None

    parts = stripped.split()
    location = f"{source}:{line_number}" if line_number is not None else source
    if len(parts) < 3:
        raise ValueError(
            f"Malformed verification pair at {location}: expected "
            f"'<label> <enrol_utt> <test_utt>', got {line!r}"
        )
    try:
        label = int(parts[0])
    except ValueError as exc:
        raise ValueError(
            f"Malformed verification label at {location}: expected integer, "
            f"got {parts[0]!r}"
        ) from exc

    return label, normalize_utterance_id(parts[1]), normalize_utterance_id(parts[2])


__all__ = ["normalize_utterance_id", "parse_verification_pair"]