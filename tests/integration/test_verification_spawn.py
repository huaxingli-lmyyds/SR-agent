"""Exercise the real SpeechBrain dataset, collate function and spawn workers."""

import csv
import pickle

import pytest


def test_verification_loader_reads_audio_with_spawn_workers(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from agent.runners.speechbrain_dependency import (
        patch_torchaudio_compatibility,
    )

    patch_torchaudio_compatibility()
    pytest.importorskip("speechbrain")
    from recipes.voxceleb.speaker_verification_cosine import dataio_prep

    wav = tmp_path / "sample.wav"
    sf.write(wav, np.linspace(-0.5, 0.5, 1600, dtype="float32"), 16000)
    manifest = tmp_path / "enrol.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "duration", "wav", "start", "stop"])
        writer.writerow(["a", 0.1, str(wav), 0, 1600])
        writer.writerow(["b", 0.05, str(wav), 0, 800])
    options = {
        "batch_size": 2,
        "num_workers": 1,
        "multiprocessing_context": "spawn",
    }
    _, enrol, _ = dataio_prep(
        {
            "data_folder": str(tmp_path),
            "score_norm": "none",
            "enrol_data": str(manifest),
            "test_data": str(manifest),
            "enrol_dataloader_opts": options,
            "test_dataloader_opts": options,
        }
    )
    pickle.dumps(enrol.dataset)
    batches = list(enrol)
    assert len(batches) == 1
    batch = batches[0]
    assert set(batch.id) == {"a", "b"}
    signals, lengths = batch.sig
    assert signals.shape == (2, 1600)
    assert sorted(lengths.tolist()) == [0.5, 1.0]


def test_plda_config_is_rejected_before_model_loading():
    from agent.runners.speechbrain_validation import (
        validate_verification_config,
    )

    with pytest.raises(ValueError, match="PLDA"):
        validate_verification_config("configs/verification_plda_xvector.yaml")


@pytest.mark.parametrize("random_chunk", [True, False])
def test_training_audio_and_labels_load_with_spawn_workers(
    tmp_path, random_chunk
):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from agent.runners.speechbrain_dependency import (
        patch_torchaudio_compatibility,
    )

    patch_torchaudio_compatibility()
    pytest.importorskip("speechbrain")
    from recipes.voxceleb.batch_compat import with_padded_batch
    from recipes.voxceleb.train_speaker_embeddings import dataio_prep
    from speechbrain.dataio.dataloader import make_dataloader

    manifest = tmp_path / "training.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "duration", "wav", "start", "stop", "spk_id"])
        for index, samples in enumerate((800, 3200)):
            wav = tmp_path / f"{index}.wav"
            sf.write(
                wav, np.linspace(-0.5, 0.5, samples, dtype="float32"), 16000
            )
            writer.writerow(
                [
                    str(index),
                    samples / 16000,
                    str(wav),
                    0,
                    samples,
                    f"speaker_{index}",
                ]
            )
    train, dev, encoder = dataio_prep(
        {
            "data_folder": str(tmp_path),
            "train_annotation": str(manifest),
            "valid_annotation": str(manifest),
            "save_folder": str(tmp_path),
            "sample_rate": 16000,
            "sentence_len": 0.1,
            "random_chunk": random_chunk,
            "validation_crop": "center",
        }
    )
    # Both audio and label callables, including their bound encoder/config,
    # must survive serialization when training runs in an isolated process.
    pickle.dumps(train)
    pickle.dumps(dev)
    options = with_padded_batch(
        {"batch_size": 2, "num_workers": 1, "multiprocessing_context": "spawn"}
    )
    (batch,) = list(make_dataloader(train, **options))
    signals, lengths = batch.sig
    assert tuple(signals.shape) == (2, 1600 if random_chunk else 3200)
    encoded, label_lengths = batch.spk_id_encoded
    assert encoded.flatten().tolist() == [
        encoder.encode_sequence_torch([f"speaker_{index}"]).item()
        for index in (0, 1)
    ]
    assert label_lengths.tolist() == [1.0, 1.0]
    if random_chunk:
        assert signals[0, 800:].abs().sum().item() == 0
    (dev_batch,) = list(make_dataloader(dev, **options))
    assert tuple(dev_batch.sig[0].shape) == (2, 1600)
    assert dev_batch.sig[0][0, 800:].abs().sum().item() == 0


def test_validation_crop_is_stable_while_training_crop_is_random(tmp_path):
    import random

    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from agent.runners.speechbrain_dependency import patch_torchaudio_compatibility

    patch_torchaudio_compatibility()
    pytest.importorskip("speechbrain")
    from recipes.voxceleb.train_speaker_embeddings import audio_pipeline

    audio = tmp_path / "long.wav"
    sf.write(audio, np.linspace(-0.5, 0.5, 32000, dtype="float32"), 16000)
    args = (str(audio), 0, 32000, 2.0)
    kwargs = {"sample_rate": 16000, "sentence_len": 0.1, "random_chunk": True}
    before = random.getstate()
    try:
        random.seed(1)
        dev1 = audio_pipeline(*args, **kwargs, crop_mode="center")
        train1 = audio_pipeline(*args, **kwargs)
        random.seed(2)
        dev2 = audio_pipeline(*args, **kwargs, crop_mode="center")
        train2 = audio_pipeline(*args, **kwargs)
    finally:
        random.setstate(before)
    assert dev1.equal(dev2)
    assert not train1.equal(train2)
    expected, _ = sf.read(audio, start=15200, stop=16800, dtype="float32")
    assert np.array_equal(dev1.numpy(), expected)
    with pytest.raises(ValueError, match="validation_crop"):
        audio_pipeline(*args, **kwargs, crop_mode="invalid")


def test_real_session_preparation_rebuilds_after_exclusion_protocol_changes(tmp_path):
    import random

    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from agent.runners.speechbrain_dependency import patch_torchaudio_compatibility

    patch_torchaudio_compatibility()
    pytest.importorskip("speechbrain")
    from recipes.voxceleb.voxceleb_prepare import prepare_voxceleb

    data = tmp_path / "corpus"
    for speaker in ("id10001", "id10002", "id10003", "id10004"):
        for session in ("a", "b", "c"):
            directory = data / "wav" / speaker / session
            directory.mkdir(parents=True)
            sf.write(directory / "1.wav", np.ones(3200) * 0.1, 16000)
    pairs = tmp_path / "exclude.txt"
    pairs.write_text("0 id10003/a/1.wav id10004/a/1.wav\n")
    save = tmp_path / "annotations"

    def prepare_and_read():
        before = random.getstate()
        try:
            random.seed(7)
            prepare_voxceleb(
                str(data), str(save), str(pairs),
                splits=["train", "dev"], split_ratio=[90, 10],
                split_unit="session", random_segment=True,
            )
        finally:
            random.setstate(before)
        rows = []
        for split in ("train", "dev"):
            with (save / f"{split}.csv").open() as stream:
                rows.append(list(csv.DictReader(stream)))
        return rows

    train, dev = prepare_and_read()
    assert len(train) == 4 and len(dev) == 2
    assert {row["spk_id"] for row in train + dev} == {"id10001", "id10002"}
    assert not {row["wav"].rsplit("/", 1)[0] for row in train} & {
        row["wav"].rsplit("/", 1)[0] for row in dev
    }
    pairs.write_text("0 id10002/a/1.wav id10004/a/1.wav\n")
    train, dev = prepare_and_read()
    assert {row["spk_id"] for row in train + dev} == {"id10001", "id10003"}


def test_xvector_default_declares_the_executed_scoring_protocol():
    from agent.models import get_model_adapter
    from agent.runners.speechbrain_validation import (
        validate_verification_config,
    )
    from agent.utils import ConfigParser

    config = get_model_adapter("xvector").default_evaluation_config
    values = ConfigParser(config).load_config()
    assert "compute_plda" not in values
    assert values["score_norm"] == "none"
    assert values["embedding_model"]["in_channels"] == 24
    validate_verification_config(config)


def test_real_ecapa_evaluation_with_local_weights_and_spawn_workers(
    tmp_path, monkeypatch
):
    """No downloads/GPU: exercise runner -> recipe -> workers -> EER scoring."""
    import math
    from pathlib import Path

    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    torch = pytest.importorskip("torch")
    from agent.runners.speechbrain_dependency import (
        patch_torchaudio_compatibility,
    )

    patch_torchaudio_compatibility()
    pytest.importorskip("speechbrain")
    from agent.runners.speechbrain import SpeechBrainRunnerAdapter
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    data = tmp_path / "dataset"
    for speaker, frequency in (("id10001", 200), ("id10002", 400)):
        folder = data / "wav" / speaker / "session"
        folder.mkdir(parents=True)
        time = np.arange(6400, dtype="float32") / 16000
        for index in (1, 2):
            samples = 0.2 * np.sin(2 * np.pi * (frequency + index) * time)
            sf.write(folder / f"{index}.wav", samples, 16000)
    pairs = tmp_path / "pairs.txt"
    pairs.write_text(
        "1 id10001/session/1.wav id10001/session/2.wav\n"
        "1 id10002/session/1.wav id10002/session/2.wav\n"
        "0 id10001/session/1.wav id10002/session/2.wav\n"
        "0 id10002/session/1.wav id10001/session/2.wav\n"
    )
    weights = tmp_path / "weights"
    weights.mkdir()
    torch.manual_seed(7)
    model = ECAPA_TDNN(
        input_size=8,
        channels=[16, 16, 16, 16, 48],
        attention_channels=8,
        lin_neurons=8,
    )
    torch.save(model.state_dict(), weights / "embedding_model.ckpt")
    config = tmp_path / "verification.yaml"
    config.write_text(f"""
data_folder: {data}
output_folder: {tmp_path / "evaluation"}
save_folder: !ref <output_folder>/save
verification_file: {pairs}
pretrain_path: {weights}
prep_cache_root: {tmp_path / "cache"}
enrol_data: !ref <save_folder>/enrol.csv
test_data: !ref <save_folder>/test.csv
split_ratio: [90, 10]
skip_prep: false
score_norm: none
enrol_dataloader_opts:
    batch_size: 2
    num_workers: 1
test_dataloader_opts:
    batch_size: 2
    num_workers: 1
compute_features: !new:speechbrain.lobes.features.Fbank
    n_mels: 8
mean_var_norm: !new:speechbrain.processing.features.InputNormalization
    norm_type: sentence
    std_norm: false
embedding_model: !new:speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN
    input_size: 8
    channels: [16, 16, 16, 16, 48]
    attention_channels: 8
    lin_neurons: 8
pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
    collect_in: !ref <save_folder>
    loadables:
        embedding_model: !ref <embedding_model>
    paths:
        embedding_model: !ref <pretrain_path>/embedding_model.ckpt
""")
    result = SpeechBrainRunnerAdapter().run_evaluation(
        str(config),
        str(weights),
        str(data),
        {"_run_opts": {"device": "cpu"}, "_hpo_max_duration_seconds": 45},
    )
    assert result["status"] == "success", result.get("error")
    assert math.isfinite(result["eer"]) and 0 <= result["eer"] <= 1
    assert math.isfinite(result["min_dcf"])
    assert len(Path(result["scores_path"]).read_text().splitlines()) == 4
