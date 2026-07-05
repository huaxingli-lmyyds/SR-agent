from __future__ import annotations

from pathlib import Path

import pytest


def test_voxceleb_recipes_use_project_audio_compat_layer() -> None:
    project_root = Path(__file__).parents[2]
    for relative in (
        "recipes/voxceleb/train_speaker_embeddings.py",
        "recipes/voxceleb/speaker_verification_cosine.py",
        "recipes/voxceleb/voxceleb_prepare.py",
    ):
        source = (project_root / relative).read_text(encoding="utf-8")
        assert "from recipes.voxceleb.audio_compat import audio_io" in source
        assert "from speechbrain.dataio import audio_io" not in source


def test_audio_compat_exposes_load_function() -> None:
    pytest.importorskip("soundfile")
    from recipes.voxceleb.audio_compat import audio_io

    assert callable(audio_io.load)

def test_soundfile_fallback_loads_channel_first_tensor(tmp_path) -> None:
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf

    from recipes.voxceleb.audio_compat import audio_io

    wav_path = tmp_path / "sample.wav"
    samples = np.linspace(-0.5, 0.5, 160, dtype="float32")
    sf.write(wav_path, samples, 16000)

    signal, sample_rate = audio_io.load(str(wav_path), frame_offset=10, num_frames=20)

    assert sample_rate == 16000
    assert tuple(signal.shape) == (1, 20)
