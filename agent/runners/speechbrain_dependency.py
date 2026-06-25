"""Runtime validation for the external SpeechBrain dependency."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Final


SUPPORTED_SPEECHBRAIN_VERSION: Final[str] = "1.0.3"


def require_speechbrain() -> str:
    """Return the installed version or raise an actionable compatibility error."""
    try:
        installed = version("speechbrain")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "SpeechBrain is not installed. Install project dependencies with "
            "'python -m pip install -e .'."
        ) from exc
    if installed != SUPPORTED_SPEECHBRAIN_VERSION:
        raise RuntimeError(
            "Unsupported SpeechBrain version "
            f"{installed}; expected {SUPPORTED_SPEECHBRAIN_VERSION}. "
            "Reinstall project dependencies to restore the tested version."
        )
    return installed


__all__ = ["SUPPORTED_SPEECHBRAIN_VERSION", "require_speechbrain"]
