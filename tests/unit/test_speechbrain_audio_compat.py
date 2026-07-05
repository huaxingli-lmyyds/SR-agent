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


def test_training_audio_pipeline_guards_short_random_chunks() -> None:
    source = Path("recipes/voxceleb/train_speaker_embeddings.py").read_text(encoding="utf-8")

    assert "duration_sample - snt_len_sample" in source
    assert "if duration_sample > snt_len_sample" in source
    assert "random.randint(0, duration_sample - snt_len_sample)" in source
    assert "num_frames = max(1, stop - start)" in source


def test_soundfile_fallback_pads_short_reads(tmp_path) -> None:
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf

    from recipes.voxceleb.audio_compat import audio_io

    wav_path = tmp_path / "short.wav"
    sf.write(wav_path, np.ones(8, dtype="float32"), 16000)

    signal, sample_rate = audio_io.load(str(wav_path), frame_offset=0, num_frames=20)

    assert sample_rate == 16000
    assert tuple(signal.shape) == (1, 20)
    assert signal[0, 8:].abs().sum().item() == 0
