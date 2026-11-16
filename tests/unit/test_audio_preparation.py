import csv
from hashlib import sha256
from pathlib import Path
import shutil

import numpy as np
import pytest
import soundfile as sf

from agent.data_processing.preparation import (
    prepare_audio,
    validate_prepared_data,
)
from agent.data_processing.workflow import DataProcessingWorkflow


def raw_audio(root, speakers=8):
    for speaker in range(speakers):
        for session in range(2):
            for utterance in range(2):
                path = (
                    root
                    / f"person{speaker}"
                    / f"session{session}"
                    / f"{utterance}.wav"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                t = np.arange(4000) / 8000
                signal = 0.2 * np.sin(
                    2
                    * np.pi
                    * (200 + speaker * 21 + session * 5 + utterance)
                    * t
                )
                sf.write(path, np.column_stack((signal, signal)), 8000)
    return root


def read_manifest(root, role):
    with (root / f"{role}.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_raw_audio_becomes_portable_reproducible_leak_free_protocol(tmp_path):
    source = raw_audio(tmp_path / "raw")
    before = {
        str(p): sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*.wav")
    }
    first = prepare_audio(source, tmp_path / "prepared", {"seed": 17})
    second = prepare_audio(source, tmp_path / "repeated", {"seed": 17})
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert before == {
        str(p): sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*.wav")
    }
    root = Path(first["root"])
    roles = {
        role: read_manifest(root, role)
        for role in ("train", "dev", "hpo_validation", "final_test")
    }
    speakers = {
        role: {r["spk_id"] for r in rows} for role, rows in roles.items()
    }
    assert speakers["dev"] <= speakers["train"]
    assert not speakers["train"] & (
        speakers["hpo_validation"] | speakers["final_test"]
    )
    assert not speakers["hpo_validation"] & speakers["final_test"]
    groups = {
        role: {Path(r["wav"]).parent.as_posix() for r in rows}
        for role, rows in roles.items()
    }
    assert not groups["train"] & groups["dev"]
    for rows in roles.values():
        for row in rows:
            path = Path(row["wav"].replace("$data_root", str(root)))
            info = sf.info(path)
            assert info.samplerate == 16000 and info.channels == 1
    for name in ("validation_pairs.txt", "final_test_pairs.txt"):
        pairs = [
            line.split() for line in (root / name).read_text().splitlines()
        ]
        assert {row[0] for row in pairs} == {"0", "1"}
        for label, left, right in pairs:
            assert (root / "wav" / left).is_file() and (
                root / "wav" / right
            ).is_file()
            assert left != right
            assert (left.split("/")[0] == right.split("/")[0]) == (label == "1")
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    validate_prepared_data({**first, "root": str(moved)}, verify_audio=True)
    (moved / "train.csv").write_text("changed")
    with pytest.raises(ValueError, match="manifest changed"):
        validate_prepared_data({**first, "root": str(moved)})


def test_raw_preparation_reaches_published_quality_gate(tmp_path):
    source = raw_audio(tmp_path / "raw")
    result = DataProcessingWorkflow().run(
        {
            "dataset_uri": str(source),
            "dataset_type": "audio",
            "task_type": "speaker_verification",
            "output_path": str(tmp_path / "records/version.json"),
            "requested_operations": [{"operation": "prepare_speaker_dataset"}],
        }
    )
    assert result["status"] == "success", result.get("error")
    version = result["published_version"]
    assert version["quality_decision"]["training_allowed"]
    assert version["prepared_data"]["root"] == version["consumer_uri"]
    validate_prepared_data(version["prepared_data"], verify_audio=True)
    reused = DataProcessingWorkflow().run(
        {
            "dataset_uri": version["consumer_uri"],
            "task_type": "speaker_verification",
            "output_path": str(tmp_path / "reuse/version.json"),
        }
    )
    assert reused["status"] == "success"
    assert (
        reused["published_version"]["prepared_data"] == version["prepared_data"]
    )
    with pytest.raises(ValueError, match="protected evaluation speakers"):
        DataProcessingWorkflow().run(
            {
                "dataset_uri": version["consumer_uri"],
                "task_type": "speaker_verification",
                "output_path": str(tmp_path / "resplit/version.json"),
                "requested_operations": [
                    {"operation": "prepare_speaker_dataset"}
                ],
            }
        )


def test_advisor_cannot_break_frozen_preparation_or_reuse_stale_feedback(
    tmp_path,
):
    from agent.data_processing.feedback import normalize_feedback

    source = raw_audio(tmp_path / "raw")
    workflow = DataProcessingWorkflow(
        strategy_advisor=lambda state: {
            "suggested_operations": [{"operation": "filter_by_duration"}],
        }
    )
    result = workflow.run(
        {
            "dataset_uri": str(source),
            "dataset_type": "audio",
            "task_type": "speaker_verification",
            "output_path": str(tmp_path / "records/version.json"),
            "requested_operations": [{"operation": "prepare_speaker_dataset"}],
        }
    )
    assert result["status"] == "success", result.get("error")
    assert (
        "frozen dataset preparation"
        in result["plan"]["rejected_operations"][0]["reason"]
    )
    prepared = result["published_version"]["prepared_data"]
    feedback = {
        "dataset_uri": prepared["root"],
        "protocol_sha256": prepared["protocol_sha256"],
        "study_id": "study1",
        "trial_count": 1,
        "evaluation_role": "hpo_validation",
    }
    assert (
        normalize_feedback(feedback, prepared["root"])["status"] == "accepted"
    )
    protocol = Path(prepared["root"]) / "prepared_protocol.json"
    protocol.write_text(
        protocol.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    assert (
        normalize_feedback(feedback, prepared["root"])["reason"]
        == "dataset_protocol_mismatch"
    )


def test_unlabelled_audio_and_conflicting_duplicate_labels_are_rejected(
    tmp_path,
):
    source = raw_audio(tmp_path / "raw")
    shutil.copyfile(
        source / "person0/session0/0.wav", source / "person1/session0/0.wav"
    )
    with pytest.raises(ValueError, match="conflicting speaker"):
        prepare_audio(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    flat = tmp_path / "flat"
    flat.mkdir()
    shutil.copyfile(source / "person0/session0/0.wav", flat / "audio.wav")
    with pytest.raises(ValueError, match="identities cannot be inferred"):
        prepare_audio(flat, tmp_path / "unlabelled")


def test_label_manifest_cannot_escape_audio_root(tmp_path):
    source = raw_audio(tmp_path / "raw")
    labels = tmp_path / "labels.csv"
    labels.write_text("path,speaker_id\n../outside.wav,speaker1\n")
    with pytest.raises(ValueError, match="escapes raw audio"):
        prepare_audio(source, tmp_path / "bad", {"labels_csv": str(labels)})


def test_flat_audio_can_use_explicit_speaker_and_session_labels(tmp_path):
    original = raw_audio(tmp_path / "source")
    flat = tmp_path / "flat"
    flat.mkdir()
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "speaker_id", "session_id"])
        for index, path in enumerate(sorted(original.rglob("*.wav"))):
            name = f"recording{index}.wav"
            shutil.copyfile(path, flat / name)
            writer.writerow([name, path.parent.parent.name, path.parent.name])
    prepared = prepare_audio(
        flat, tmp_path / "prepared", {"labels_csv": str(labels)}
    )
    assert prepared["training_speaker_count"] == 4
    validate_prepared_data(prepared, verify_audio=True)
