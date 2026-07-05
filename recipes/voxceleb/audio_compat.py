"""Compatibility wrapper for SpeechBrain audio loading APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.runners.soundfile_audio import load_audio
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
        frames = -1 if num_frames is None else int(num_frames)
        tensor, sample_rate = load_audio(
            path,
            frame_offset=frame_offset,
            num_frames=frames,
            channels_first=True,
        )
        if frames > 0 and tensor.shape[-1] < frames:
            torch = _torch_module()
            pad_shape = (*tensor.shape[:-1], frames - tensor.shape[-1])
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=-1)
        return tensor, sample_rate


def _torch_module():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for the SR-agent SpeechBrain audio loader. "
            "Install project dependencies with: pip install -e .[speech]"
        ) from exc
    return torch


audio_io = _SoundFileAudioIO()

__all__ = ["audio_io"]
