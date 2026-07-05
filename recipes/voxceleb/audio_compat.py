"""Compatibility wrapper for SpeechBrain audio loading APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    from speechbrain.dataio import audio_io as audio_io  # type: ignore
except ImportError:
    import torchaudio

    @dataclass(frozen=True)
    class _TorchAudioIO:
        """Expose the old speechbrain.dataio.audio_io.load surface."""

        def load(
            self,
            path: str,
            num_frames: int | None = None,
            frame_offset: int = 0,
            **kwargs: Any,
        ):
            if num_frames is None:
                num_frames = -1
            return torchaudio.load(
                path,
                frame_offset=int(frame_offset),
                num_frames=int(num_frames),
                **kwargs,
            )

    audio_io = _TorchAudioIO()


__all__ = ["audio_io"]