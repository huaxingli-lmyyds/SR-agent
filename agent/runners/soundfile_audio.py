"""Soundfile-backed audio loading helpers used by SpeechBrain compatibility shims."""

from __future__ import annotations

from os import PathLike
from typing import Any


def load_audio(
    uri: str | PathLike[str],
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: str | None = None,
    buffer_size: int = 4096,
    backend: str | None = None,
    **_kwargs: Any,
):
    """Load audio with a torchaudio.load-compatible return shape.

    New torchaudio releases may route ``torchaudio.load`` through TorchCodec.
    SpeechBrain 1.0.3 still calls ``torchaudio.load`` inside augmentation
    datasets, so SR-agent uses soundfile for local waveform reads instead.
    """
    del format, buffer_size, backend
    np, sf, torch = _audio_dependencies()
    frames = -1 if num_frames is None or int(num_frames) < 0 else int(num_frames)
    dtype = "float32" if normalize else "int16"
    data, sample_rate = sf.read(
        str(uri),
        start=max(0, int(frame_offset)),
        frames=frames,
        dtype=dtype,
        always_2d=True,
    )
    array = data.T if channels_first else data
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor, sample_rate


def _audio_dependencies():
    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise ImportError(
            "numpy, soundfile, and torch are required for SR-agent audio "
            "loading. Install project dependencies with: pip install -e .[speech]"
        ) from exc
    return np, sf, torch


__all__ = ["load_audio"]
