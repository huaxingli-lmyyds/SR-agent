"""Regression coverage for speaker roles, quality errors and immutable holdouts."""

import csv
import random
import wave
from pathlib import Path

import pytest

from agent.data_processing.audio import _profile_manifests, _speaker
from agent.data_processing.contracts import (
    DataOperation,
    DataProcessingPlan,
    QualityPolicy,
)
from agent.data_processing.quality import DefaultQualityGate
from agent.data_processing.scope import bind_evaluation_scope
from agent.data_processing.service import (
    build_processing_plan,
    execute_plan,
    infer_dataset_spec,
    profile_dataset,
)
from recipes.voxceleb.splits import split_sessions


def write_audio(path, seconds=0.2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x01\x01" * int(16000 * seconds))
    return path


def write_manifest(root, split, entries):
    with (root / f"{split}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "wav", "spk_id"])
        for index, (path, speaker) in enumerate(entries):
            writer.writerow([str(index), str(path), speaker])


@pytest.mark.parametrize("with_trials", [False, True])
def test_manifest_errors_survive_trial_profile_and_block_gate(
    tmp_path, with_trials
):
    write_audio(tmp_path / "valid.wav")
    write_manifest(tmp_path, "train", [("missing.wav", "id10001")])
    if with_trials:
        (tmp_path / "pairs.txt").write_text("malformed trial\n")
    dataset = infer_dataset_spec(str(tmp_path), dataset_type="audio")
    profile = profile_dataset(dataset)
    assert profile.quality_metrics["missing_audio_count"] == 1
    assert "manifest_audio_missing" in {item.code for item in profile.issues}
    assert not DefaultQualityGate().evaluate(profile).training_allowed
    result = execute_plan(DataProcessingPlan(dataset=dataset))[-1]
    assert result.status == "failed"
    assert "manifest_audio_missing" in result.details["decision"]["blockers"]


def test_voxceleb_identity_uses_speaker_not_video(tmp_path):
    paths = [
        write_audio(tmp_path / "wav" / "id10001" / session / "1.wav")
        for session in ("video_a", "video_b")
    ]
    for root in (tmp_path, tmp_path / "wav"):
        assert {_speaker(root, path) for path in paths} == {"id10001"}
    profile = profile_dataset(
        infer_dataset_spec(str(tmp_path), dataset_type="audio")
    )
    assert profile.distributions["samples_per_speaker"]["count"] == 1
    assert profile.distributions["samples_per_speaker"]["max"] == 2


def test_internal_dev_identity_overlap_allowed_but_file_leakage_blocked(
    tmp_path,
):
    first = write_audio(tmp_path / "id10001" / "a.wav")
    second = write_audio(tmp_path / "id10001" / "b.wav")
    write_manifest(tmp_path, "train", [(first, "id10001")])
    write_manifest(tmp_path, "dev", [(second, "id10001")])
    policy = QualityPolicy.for_task("speaker_verification")
    assert _profile_manifests(tmp_path, policy)["speaker_overlap_count"] == 0
    assert _profile_manifests(tmp_path)["speaker_overlap_count"] == 1
    write_manifest(tmp_path, "dev", [(first, "id10001")])
    assert _profile_manifests(tmp_path, policy)["file_overlap_count"] == 1


def test_endpoint_lists_share_population_but_holdout_remains_disjoint(tmp_path):
    audio = write_audio(tmp_path / "id10001" / "a.wav")
    for split in ("enrol", "test"):
        write_manifest(tmp_path, split, [(audio, "id10001")])
    policy = QualityPolicy.for_task("speaker_verification")
    result = _profile_manifests(tmp_path, policy)
    assert result["speaker_overlap_count"] == result["file_overlap_count"] == 0
    assert result["duplicate_sample_count"] == 0
    dataset = infer_dataset_spec(
        str(tmp_path), dataset_type="audio", task_type="speaker_verification"
    )
    assert not build_processing_plan(profile_dataset(dataset)).operations
    write_manifest(tmp_path, "validation", [(audio, "id10001")])
    result = _profile_manifests(tmp_path, policy)
    assert result["speaker_overlap_count"] == result["file_overlap_count"] == 2


def test_explicit_split_roles_support_custom_manifest_names(tmp_path):
    audio = write_audio(tmp_path / "sample.wav")
    for name in ("fit", "monitor"):
        write_manifest(tmp_path, name, [(audio, "id10001")])
    policy = QualityPolicy(
        split_roles={"fit": "train", "monitor": "internal_dev"}
    )
    result = _profile_manifests(tmp_path, policy)
    assert result["speaker_overlap_count"] == 0
    assert result["file_overlap_count"] == 1
    with pytest.raises(ValueError, match="split_roles"):
        QualityPolicy(split_roles={"fit": "mystery"})


def test_voxceleb_profile_ignores_augmentation_assets(tmp_path):
    write_audio(tmp_path / "wav" / "id10001" / "video" / "1.wav")
    noise = tmp_path / "noise"
    noise.mkdir()
    (noise / "broken.wav").write_bytes(b"not audio")
    (noise / "noise.csv").write_text("ID,wav\nx,missing.wav\n")
    profile = profile_dataset(
        infer_dataset_spec(str(tmp_path), dataset_type="audio")
    )
    assert profile.sample_count == 1
    assert profile.quality_metrics["unreadable_audio_count"] == 0
    assert profile.quality_metrics["missing_speaker_count"] == 0


def test_duration_warning_does_not_automatically_delete_speech(tmp_path):
    write_audio(tmp_path / "id10001" / "long.wav", seconds=0.3)
    policy = QualityPolicy(max_duration_seconds=0.2)
    profile = profile_dataset(
        infer_dataset_spec(str(tmp_path), dataset_type="audio"), policy
    )
    assert "duration_out_of_range" in {issue.code for issue in profile.issues}
    plan = build_processing_plan(profile)
    assert "filter_by_duration" not in {op.operation for op in plan.operations}
    assert DefaultQualityGate().evaluate(profile).training_allowed


def test_mixed_corpus_protection_covers_planning_and_direct_execution(tmp_path):
    source = tmp_path / "corpus"
    write_audio(source / "wav" / "id10001" / "video" / "a.wav")
    protocol = tmp_path / "exclude.txt"
    protocol.write_text("1 id10001/video/a.wav id10001/video/b.wav\n")
    dataset = infer_dataset_spec(
        str(source), dataset_type="audio", task_type="speaker_verification"
    )
    bind_evaluation_scope(dataset, str(protocol))
    assert dataset.metadata["protected_speakers"] == ["id10001"]
    assert len(dataset.metadata["exclusion_protocol_sha256"]) == 64
    profile = profile_dataset(dataset)
    requested = {"operation": "build_debug_subset"}
    with pytest.raises(ValueError, match="training-only view"):
        build_processing_plan(profile, requested_operations=[requested])
    plan = build_processing_plan(
        profile, requested_operations=[{**requested, "_advisory": True}]
    )
    assert not plan.operations
    assert (
        "protected evaluation speakers" in plan.rejected_operations[0]["reason"]
    )
    direct = DataProcessingPlan(
        dataset, operations=[DataOperation("build_debug_subset")]
    )
    with pytest.raises(ValueError, match="training-only view"):
        execute_plan(direct, output_root=tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert execute_plan(DataProcessingPlan(dataset))[-1].status == "success"


def test_training_only_view_can_be_processed(tmp_path):
    source = tmp_path / "corpus"
    write_audio(source / "wav" / "id10002" / "video" / "a.wav")
    protocol = tmp_path / "exclude.txt"
    protocol.write_text("1 id10001/video/a.wav id10001/video/b.wav\n")
    dataset = infer_dataset_spec(
        str(source), dataset_type="audio", task_type="speaker_verification"
    )
    bind_evaluation_scope(dataset, str(protocol))
    assert not dataset.metadata["contains_protected_speakers"]
    plan = build_processing_plan(
        profile_dataset(dataset),
        requested_operations=[{"operation": "build_debug_subset"}],
    )
    assert plan.operations[0].operation == "build_debug_subset"


@pytest.mark.parametrize("missing_reference", [False, True])
def test_workflow_binds_protocol_before_advice_and_enforces_gate(
    tmp_path, missing_reference
):
    pytest.importorskip("langgraph")
    from agent.data_processing.workflow import DataProcessingWorkflow

    source = tmp_path / "corpus"
    write_audio(source / "wav" / "id10001" / "video" / "a.wav")
    if missing_reference:
        write_manifest(source, "train", [("missing.wav", "id10002")])
    protocol = tmp_path / "exclude.txt"
    protocol.write_text("1 id10001/video/a.wav id10001/video/b.wav\n")

    def advisor(state):
        assert state["profile"]["dataset"]["metadata"][
            "contains_protected_speakers"
        ]
        return {"suggested_operations": [{"operation": "build_debug_subset"}]}

    result = DataProcessingWorkflow(strategy_advisor=advisor).run(
        {
            "dataset_uri": str(source),
            "dataset_type": "audio",
            "task_type": "speaker_verification",
            "training_exclusion_pairs": str(protocol),
            "output_path": str(tmp_path / "version.json"),
        }
    )
    assert not result["plan"]["operations"]
    rejected = result["plan"]["rejected_operations"]
    assert rejected[0]["source"] == "llm"
    if missing_reference:
        assert result["status"] == "failed"
        assert rejected[1]["source"] == "quality_policy"
        assert not (tmp_path / "version.json").exists()
    else:
        assert result["status"] == "success"
        assert (
            result["published_version"]["quality_policy"]["split_roles"]["dev"]
            == "internal_dev"
        )


def test_session_split_preserves_identities_and_isolates_recordings():
    files = [
        f"/dataset/wav/{speaker}/{session}/{index}.wav"
        for speaker in ("id10001", "id10002")
        for session in ("a", "b", "c")
        for index in range(3)
    ] + ["/dataset/wav/id10003/only/1.wav"]
    before = random.getstate()
    try:
        random.seed(42)
        train, dev = split_sessions(files, 90)
        random.seed(42)
        assert split_sessions(list(reversed(files)), 90) == (train, dev)
    finally:
        random.setstate(before)
    assert set(train) | set(dev) == set(files)
    assert not set(train) & set(dev)
    assert {Path(path).parent.parent.name for path in train} == {
        "id10001",
        "id10002",
        "id10003",
    }
    assert not {Path(path).parent for path in train} & {
        Path(path).parent for path in dev
    }
    assert all("id10003" not in path for path in dev)
    with pytest.raises(ValueError, match="training percentage"):
        split_sessions(files, 100)
