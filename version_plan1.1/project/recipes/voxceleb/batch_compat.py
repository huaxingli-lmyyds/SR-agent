"""Batch compatibility helpers for local SpeechBrain VoxCeleb recipes."""

from __future__ import annotations

from typing import Any, Mapping

from agent.runners.speechbrain_dependency import patch_torchaudio_compatibility


def with_padded_batch(dataloader_options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return dataloader options that produce SpeechBrain batch objects."""
    patch_torchaudio_compatibility()
    from speechbrain.dataio.batch import PaddedBatch

    options = dict(dataloader_options or {})
    options.setdefault("collate_fn", PaddedBatch)
    return options


__all__ = ["with_padded_batch"]