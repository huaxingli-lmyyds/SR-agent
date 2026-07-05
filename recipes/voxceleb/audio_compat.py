"""Compatibility wrapper for SpeechBrain audio loading APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.runners.speechbrain_dependency import patch_torchaudio_compatibility

patch_torchaudio_compatibility()


@dataclass(frozen=True)
class _SoundFileAudioIO:
    """Expose the old speechbrain.dataio.audio_io.load surface.

    Newer torchaudio releases may require TorchCodec for torchaudio.load().
    The local VoxCeleb recipes only need deterministic waveform reads, so this
    project uses soundfile directly and keeps SpeechBrain away from that API.
    """

    def load(
        self,
        path: str,
        num_frames: int | None = None,
        frame_offset: int = 0,
        **_kwargs: Any,
    ):
        np, sf, torch = _audio_dependencies()
        frames = -1 if num_frames is None else int(num_frames)
        data, sample_rate = sf.read(
            path,
            start=int(frame_offset),
            frames=frames,
            dtype="float32",
            always_2d=True,
        )
        if frames > 0 and data.shape[0] < frames:
            padding = np.zeros((frames - data.shape[0], data.shape[1]), dtype="float32")
            data = np.concatenate([data, padding], axis=0)
        # soundfile returns [frames, channels]; SpeechBrain recipes expect the
        # torchaudio-style [channels, frames] tensor.
        tensor = torch.from_numpy(np.ascontiguousarray(data.T))
        return tensor, sample_rate



def _audio_dependencies():
    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise ImportError(
            "numpy, soundfile, and torch are required for the SR-agent "
            "SpeechBrain audio loader. Install project dependencies with: "
            "pip install -e .[speech]"
        ) from exc
    return np, sf, torch


audio_io = _SoundFileAudioIO()

__all__ = ["audio_io"]
