from __future__ import annotations

from pathlib import Path


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
    from recipes.voxceleb.audio_compat import audio_io

    assert callable(audio_io.load)