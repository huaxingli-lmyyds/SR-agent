"""Compatibility wrapper for SpeechBrain audio loading APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.runners.speechbrain_dependency import patch_torchaudio_compatibility

patch_torchaudio_compatibility()


try:
    from speechbrain.dataio import audio_io as audio_io  # type: ignore
except ImportError:
    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise ImportError(
            "soundfile is required for the SR-agent SpeechBrain audio fallback. "
            "Install project dependencies with: pip install -e .[speech]"
        ) from exc

    @dataclass(frozen=True)
    class _SoundFileAudioIO:
        """Expose the old speechbrain.dataio.audio_io.load surface.

        Newer torchaudio releases may require TorchCodec for torchaudio.load().
        VoxCeleb recipes only need WAV/FLAC-style loading, so soundfile is a
        smaller and more stable fallback for this project.
        """

        def load(
            self,
            path: str,
            num_frames: int | None = None,
            frame_offset: int = 0,
            **_kwargs: Any,
        ):
            frames = -1 if num_frames is None else int(num_frames)
            data, sample_rate = sf.read(
                path,
                start=int(frame_offset),
                frames=frames,
                dtype="float32",
                always_2d=True,
            )
            # soundfile returns [frames, channels]; SpeechBrain recipes expect
            # the torchaudio-style [channels, frames] tensor.
            tensor = torch.from_numpy(np.ascontiguousarray(data.T))
            return tensor, sample_rate

    audio_io = _SoundFileAudioIO()


__all__ = ["audio_io"]
