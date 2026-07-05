"""Runtime validation for the external SpeechBrain dependency."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Final


SUPPORTED_SPEECHBRAIN_VERSION: Final[str] = "1.0.3"


def patch_torchaudio_compatibility() -> list[str]:
    """Patch torchaudio APIs removed in newer releases but used by SpeechBrain.

    SpeechBrain 1.0.3 may call torchaudio.list_audio_backends(),
    get_audio_backend(), or set_audio_backend(). Newer torchaudio releases use
    dispatcher-based loading and no longer expose these functions. The shim is
    intentionally small and only restores the legacy query/set surface.
    """
    try:
        import torchaudio
    except Exception:
        return []

    patched: list[str] = []
    if not hasattr(torchaudio, "list_audio_backends"):
        def list_audio_backends() -> list[str]:
            return ["ffmpeg", "soundfile"]

        torchaudio.list_audio_backends = list_audio_backends  # type: ignore[attr-defined]
        patched.append("list_audio_backends")
    if not hasattr(torchaudio, "get_audio_backend"):
        def get_audio_backend() -> str:
            return "soundfile"

        torchaudio.get_audio_backend = get_audio_backend  # type: ignore[attr-defined]
        patched.append("get_audio_backend")
    if not hasattr(torchaudio, "set_audio_backend"):
        def set_audio_backend(_backend: str | None = None) -> None:
            return None

        torchaudio.set_audio_backend = set_audio_backend  # type: ignore[attr-defined]
        patched.append("set_audio_backend")
    return patched


def require_speechbrain() -> str:
    """Return the installed version or raise an actionable compatibility error."""
    patch_torchaudio_compatibility()
    try:
        installed = version("speechbrain")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "SpeechBrain is not installed. First install a CUDA-matched "
            "torch/torchaudio stack, then install SR-agent with "
            "'python -m pip install -e .[speech]'."
        ) from exc
    if installed != SUPPORTED_SPEECHBRAIN_VERSION:
        raise RuntimeError(
            "Unsupported SpeechBrain version "
            f"{installed}; expected {SUPPORTED_SPEECHBRAIN_VERSION}. "
            "Install the tested extra with 'python -m pip install -e .[speech]'."
        )
    return installed


__all__ = ["SUPPORTED_SPEECHBRAIN_VERSION", "patch_torchaudio_compatibility", "require_speechbrain"]
