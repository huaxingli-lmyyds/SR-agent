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


def test_voxceleb_recipes_use_project_batch_compat_layer() -> None:
    project_root = Path(__file__).parents[2]
    train_source = (project_root / "recipes/voxceleb/train_speaker_embeddings.py").read_text(
        encoding="utf-8"
    )
    verification_source = (
        project_root / "recipes/voxceleb/speaker_verification_cosine.py"
    ).read_text(encoding="utf-8")
    runner_source = (project_root / "agent/runners/speechbrain_backend.py").read_text(
        encoding="utf-8"
    )

    assert "from recipes.voxceleb.batch_compat import with_padded_batch" in train_source
    assert "from recipes.voxceleb.batch_compat import with_padded_batch" in verification_source
    assert "train_loader_kwargs=dataloader_options" in runner_source
    assert "valid_loader_kwargs=dataloader_options" in runner_source
    assert "recipe.with_padded_batch" in runner_source
    assert "train_loader_kwargs=hparams[\"dataloader_options\"]" not in runner_source
    assert "valid_loader_kwargs=hparams[\"dataloader_options\"]" not in runner_source


def test_audio_compat_uses_shared_soundfile_backend() -> None:
    audio_source = Path("recipes/voxceleb/audio_compat.py").read_text(encoding="utf-8")
    backend_source = Path("agent/runners/soundfile_audio.py").read_text(encoding="utf-8")

    assert "from agent.runners.soundfile_audio import load_audio" in audio_source
    assert "from speechbrain.dataio import audio_io" not in audio_source
    assert "import torchaudio" not in audio_source
    assert "def load_audio" in backend_source
    assert "import soundfile as sf" in backend_source
    assert "sf.read(" in backend_source


def test_training_configs_store_augmentation_annotations_with_augmentation_data() -> None:
    project_root = Path(__file__).parents[2]
    for relative in (
        "configs/train_ecapa_tdnn.yaml",
        "recipes/voxceleb/hparams/train_ecapa_tdnn.yaml",
        "recipes/voxceleb/hparams/train_resnet.yaml",
        "recipes/voxceleb/hparams/train_x_vectors.yaml",
        "recipes/voxceleb/hparams/train_ecapa_tdnn_mel_spec.yaml",
    ):
        source = (project_root / relative).read_text(encoding="utf-8")
        assert "noise_annotation: !ref <data_folder_noise>/noise.csv" in source
        assert "rir_annotation: !ref <data_folder_rir>/rir.csv" in source
        assert "noise_annotation: !ref <save_folder>/noise.csv" not in source
        assert "rir_annotation: !ref <save_folder>/rir.csv" not in source

def test_training_configs_use_project_augmentation_prepare() -> None:
    project_root = Path(__file__).parents[2]
    for relative in (
        "configs/train_ecapa_tdnn.yaml",
        "recipes/voxceleb/hparams/train_ecapa_tdnn.yaml",
        "recipes/voxceleb/hparams/train_resnet.yaml",
        "recipes/voxceleb/hparams/train_x_vectors.yaml",
        "recipes/voxceleb/hparams/train_ecapa_tdnn_mel_spec.yaml",
    ):
        source = (project_root / relative).read_text(encoding="utf-8")
        assert "recipes.voxceleb.augmentation_prepare.prepare_dataset_from_URL" in source
        assert "speechbrain.augment.preparation.prepare_dataset_from_URL" not in source


def test_project_augmentation_prepare_writes_speechbrain_csv(tmp_path) -> None:
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf

    from recipes.voxceleb.augmentation_prepare import prepare_dataset_from_URL

    wav_dir = tmp_path / "noise"
    wav_dir.mkdir()
    sf.write(wav_dir / "sample.wav", np.ones(160, dtype="float32"), 16000)
    csv_path = wav_dir / "noise.csv"

    prepare_dataset_from_URL("unused", str(wav_dir), ext="wav", csv_file=str(csv_path))

    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "ID,duration,wav,wav_format,wav_opts"
    assert len(rows) == 2
    assert "sample.wav" in rows[1]

def test_voxceleb_recipe_entrypoints_use_package_imports() -> None:
    project_root = Path(__file__).parents[2]
    for relative in (
        "recipes/voxceleb/train_speaker_embeddings.py",
        "recipes/voxceleb/speaker_verification_cosine.py",
    ):
        source = (project_root / relative).read_text(encoding="utf-8")
        assert "from recipes.voxceleb.voxceleb_prepare import prepare_voxceleb" in source
        assert "from voxceleb_prepare import prepare_voxceleb" not in source


def test_project_augmentation_prepare_delays_soundfile_import() -> None:
    source = Path("recipes/voxceleb/augmentation_prepare.py").read_text(encoding="utf-8")

    assert "def _soundfile_module" in source
    assert "import soundfile as sf" in source
    assert "soundfile as sf\nexcept" not in source
