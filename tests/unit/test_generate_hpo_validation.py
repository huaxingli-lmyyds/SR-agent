import json
from pathlib import Path

import pytest

from scripts.tools.generate_hpo_validation import generate_protocol


def _audio(root: Path, speaker: str, sessions: int = 2, per_session: int = 3):
    for session in range(sessions):
        folder = root / "wav" / speaker / f"video{session}"
        folder.mkdir(parents=True, exist_ok=True)
        for utterance in range(per_session):
            (folder / f"{utterance:05d}.wav").write_bytes(b"fake")


def _dataset(tmp_path: Path):
    data = tmp_path / "vox1"
    for speaker in ("id10001", "id10002", "id10003", "id10004"):
        _audio(data, speaker)
    for speaker in ("id10270", "id10271"):
        _audio(data, speaker)
    test = tmp_path / "veri_test2.txt"
    test.write_text(
        "1 id10270/video0/00000.wav id10270/video1/00000.wav\n"
        "0 id10270/video0/00001.wav id10271/video0/00000.wav\n",
        encoding="utf-8",
    )
    return data, test


def test_generates_balanced_deterministic_speaker_disjoint_protocol(tmp_path):
    data, test = _dataset(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    arguments = {
        "data_folder": data,
        "test_pairs": test,
        "validation_speakers": 3,
        "positive_pairs": 6,
        "negative_pairs": 6,
        "max_utterances_per_speaker": 6,
        "min_sessions": 2,
        "speaker_prefix": "id1",
        "seed": 17,
    }
    manifest = generate_protocol(output_dir=first, **arguments)
    generate_protocol(output_dir=second, **arguments)

    pairs = (first / "hpo_validation.txt").read_text().splitlines()
    assert pairs == (second / "hpo_validation.txt").read_text().splitlines()
    assert len(pairs) == 12
    assert sum(line.startswith("1 ") for line in pairs) == 6
    assert sum(line.startswith("0 ") for line in pairs) == 6
    validation_speakers = set(
        (first / "validation_speakers.txt").read_text().splitlines()
    )
    assert len(validation_speakers) == 3
    assert not validation_speakers & {"id10270", "id10271"}
    for row in pairs:
        label, left, right = row.split()
        assert left.split("/")[0] in validation_speakers
        assert right.split("/")[0] in validation_speakers
        if label == "1":
            assert left.split("/")[0] == right.split("/")[0]
            assert left.split("/")[1] != right.split("/")[1]
        else:
            assert left.split("/")[0] != right.split("/")[0]
    assert manifest["integrity"]["validation_test_speaker_overlap"] == 0
    exclusions = (first / "training_exclusions.txt").read_text().splitlines()
    assert exclusions == pairs + test.read_text().splitlines()
    assert exclusions == (
        second / "training_exclusions.txt"
    ).read_text().splitlines()
    saved = json.loads(
        (first / "hpo_validation_manifest.json").read_text()
    )
    assert saved["counts"]["validation_pairs"] == 12
    assert saved["counts"]["training_exclusion_pairs"] == 14
    assert saved["usage"]["training_exclusion_pairs"].endswith(
        "training_exclusions.txt"
    )
    assert saved["integrity"]["training_exclusions_sha256"]


def test_rejects_missing_test_audio_before_writing(tmp_path):
    data, test = _dataset(tmp_path)
    test.write_text(
        test.read_text()
        + "0 id10270/video0/missing.wav id10271/video0/00000.wav\n"
    )
    output = tmp_path / "protocol"
    with pytest.raises(ValueError, match="test-pair audio files are absent"):
        generate_protocol(
            data_folder=data,
            test_pairs=test,
            output_dir=output,
            validation_speakers=2,
            positive_pairs=2,
            negative_pairs=2,
        )
    assert not output.exists()


def test_refuses_to_overwrite_frozen_protocol(tmp_path):
    data, test = _dataset(tmp_path)
    output = tmp_path / "protocol"
    output.mkdir()
    (output / "hpo_validation.txt").write_text("frozen\n")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        generate_protocol(
            data_folder=data,
            test_pairs=test,
            output_dir=output,
            validation_speakers=2,
            positive_pairs=2,
            negative_pairs=2,
        )
    assert (output / "hpo_validation.txt").read_text() == "frozen\n"
