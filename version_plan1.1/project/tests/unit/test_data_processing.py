import csv
import wave
from pathlib import Path

import pytest

from agent.data_processing.audio import _AudioSubsetProcessor
from agent.data_processing.contracts import (
    DataOperation,
    DataOperationResult,
    DataProcessingPlan,
    QualityPolicy,
)
from agent.data_processing.materialization import materialize_file
from agent.data_processing.registry import register_processor
from agent.data_processing.service import (
    _execution_operations,
    _hash_dataset,
    build_processing_plan,
    execute_plan,
    infer_dataset_spec,
    profile_dataset,
    publish_dataset_version,
)


def test_generic_dataset_profile_plan_and_publish(
    tmp_path, dataset_dir
) -> None:
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="text")
    profile = profile_dataset(dataset)
    plan = build_processing_plan(profile, target_goal="validate")
    results = execute_plan(plan)
    output = tmp_path / "versions" / "dataset.json"
    version = publish_dataset_version(dataset, results, output)

    assert profile.sample_count == 1
    assert plan.operations == []
    assert len(results) == 1
    assert results[0].operation == "quality_gate"
    assert results[-1].status == "success"
    assert version.dataset_id == dataset.dataset_id
    assert output.exists()


def test_manifest_quality_issue_creates_and_executes_optimization(
    tmp_path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    manifest = dataset_dir / "train.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "wav"])
        writer.writeheader()
        writer.writerows(
            [
                {"id": "a", "wav": "a.wav"},
                {"id": "a", "wav": "a.wav"},
                {"id": "b", "wav": ""},
            ]
        )

    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="tabular")
    profile = profile_dataset(dataset)
    plan = build_processing_plan(profile, target_goal="clean manifests")
    results = execute_plan(plan, output_root=tmp_path / "processed")
    version = publish_dataset_version(
        dataset, results, tmp_path / "version.json"
    )

    assert plan.operations[0].operation == "filter_manifest_rows"
    assert results[0].after_metrics["dropped_row_count"] == 2
    assert version.output_uri == results[0].output_dataset_uri
    assert version.consumer_uri is None
    assert version.consumption_status == "not_ready"
    assert results[0].artifacts[0]["name"] == "train.csv"
    assert Path(version.output_uri, "train.csv").exists()


def test_complete_derived_dataset_can_be_consumed_downstream(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    with wave.open(str(dataset_dir / "sample.wav"), "wb") as stream:
        stream.setparams((1, 2, 16000, 1600, "NONE", "not compressed"))
        stream.writeframes(b"\x01\x00" * 1600)
    (dataset_dir / "train.csv").write_text(
        "id,wav,spk_id\na,sample.wav,speaker_a\n", encoding="utf-8"
    )
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="audio")
    profile = profile_dataset(dataset)
    plan = build_processing_plan(
        profile,
        requested_operations=[
            {
                "operation": "filter_manifest_rows",
                "parameters": {"materialize_complete_dataset": True},
            }
        ],
    )

    results = execute_plan(plan, output_root=tmp_path / "processed")
    version = publish_dataset_version(
        dataset, results, tmp_path / "version.json"
    )

    assert version.consumer_uri == results[0].output_dataset_uri
    assert version.consumption_status == "ready"
    assert Path(version.consumer_uri, "sample.wav").exists()


def test_requested_operation_parameters_are_validated(dataset_dir) -> None:
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="text")
    profile = profile_dataset(dataset)

    with pytest.raises(ValueError, match="drop_empty_rows"):
        build_processing_plan(
            profile,
            requested_operations=[
                {
                    "operation": "filter_manifest_rows",
                    "parameters": {"drop_empty_rows": "yes"},
                }
            ],
        )

def test_llm_cannot_enable_protected_materialization(dataset_dir) -> None:
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="text")
    profile = profile_dataset(dataset)

    plan = build_processing_plan(
        profile,
        requested_operations=[
            {
                "operation": "filter_manifest_rows",
                "parameters": {"materialize_complete_dataset": True},
                "_advisory": True,
            }
        ],
    )

    assert all(
        item.operation != "filter_manifest_rows" for item in plan.operations
    )
    assert plan.rejected_operations[0]["source"] == "llm"


def test_other_domains_can_register_parameterized_processors(tmp_path) -> None:
    class ImageMetadataProcessor:
        operation_name = "test_normalize_image_metadata"
        supported_data_types = {"image"}
        parameter_schema = {"color_mode": {"type": "string", "default": "RGB"}}

        def validate(self, dataset, parameters):
            if parameters.get("color_mode", "RGB") not in {"RGB", "L"}:
                raise ValueError("unsupported color mode")

        def execute(self, dataset, parameters):
            return DataOperationResult(
                status="success",
                operation=self.operation_name,
                before_metrics={"normalized": 0},
                after_metrics={"normalized": 1, "error_count": 0},
                details={"color_mode": parameters.get("color_mode", "RGB")},
            )

    register_processor(ImageMetadataProcessor())
    dataset_dir = tmp_path / "images"
    dataset_dir.mkdir()
    (dataset_dir / "sample.jpg").write_bytes(b"fake")
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="image")
    profile = profile_dataset(dataset)
    plan = build_processing_plan(
        profile,
        requested_operations=[
            {
                "operation": "test_normalize_image_metadata",
                "parameters": {"color_mode": "L"},
            }
        ],
    )

    results = execute_plan(plan, output_root=tmp_path / "processed")

    assert results[0].details["color_mode"] == "L"


def test_dataset_scan_limit_must_be_positive(dataset_dir) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        infer_dataset_spec(str(dataset_dir), max_files=0)


def test_validation_preview_profile_is_reused(monkeypatch, dataset_dir) -> None:
    from agent.data_processing import service

    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="text")
    profile = profile_dataset(dataset)
    plan = build_processing_plan(
        profile,
        requested_operations=[{"operation": "validate_dataset"}],
    )
    calls = 0
    original = service.profile_dataset

    def counting_profile(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "profile_dataset", counting_profile)
    execute_plan(plan, initial_profile=profile)

    assert calls == 1


def test_quality_policy_rejects_unsafe_bounds() -> None:
    with pytest.raises(ValueError, match="max_clipping_ratio"):
        QualityPolicy(max_clipping_ratio=1.1)


def test_manifest_output_cannot_modify_source_tree(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.csv").write_text(
        "id,wav\na,a.wav\n", encoding="utf-8"
    )
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="tabular")
    plan = build_processing_plan(
        profile_dataset(dataset),
        requested_operations=[{"operation": "filter_manifest_rows"}],
    )

    results = execute_plan(plan, output_root=dataset_dir / "processed")

    assert results[0].status == "failed"
    assert "outside the source dataset" in results[0].error


def test_recursive_manifests_keep_relative_paths(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    for split in ("train", "dev"):
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True)
        (split_dir / "manifest.csv").write_text(
            "id,wav\na,a.wav\n", encoding="utf-8"
        )
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="tabular")
    plan = build_processing_plan(
        profile_dataset(dataset),
        requested_operations=[
            {
                "operation": "filter_manifest_rows",
                "parameters": {"csv_glob": "**/*.csv"},
            }
        ],
    )

    results = execute_plan(plan, output_root=tmp_path / "processed")

    assert results[0].status == "success"
    assert (
        Path(results[0].output_dataset_uri) / "train" / "manifest.csv"
    ).exists()
    assert (
        Path(results[0].output_dataset_uri) / "dev" / "manifest.csv"
    ).exists()


def _write_test_wav(path: Path, frames: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, frames, "NONE", "not compressed"))
        stream.writeframes(b"\x01\x00" * frames)


def test_audio_profile_detects_split_leakage_and_invalid_trials(
    tmp_path,
) -> None:
    dataset_dir = tmp_path / "audio"
    _write_test_wav(dataset_dir / "train" / "speaker_a" / "a.wav")
    _write_test_wav(dataset_dir / "dev" / "speaker_a" / "b.wav")
    (dataset_dir / "train.csv").write_text(
        "spk_id,wav\nspeaker_a,train/speaker_a/a.wav\n", encoding="utf-8"
    )
    (dataset_dir / "dev.csv").write_text(
        "spk_id,wav\nspeaker_a,dev/speaker_a/b.wav\n", encoding="utf-8"
    )
    pair = "1 train/speaker_a/a.wav dev/speaker_a/b.wav\n"
    (dataset_dir / "verification_trials.txt").write_text(
        pair + pair, encoding="utf-8"
    )

    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="audio")
    profile = profile_dataset(dataset)
    plan = build_processing_plan(
        profile,
        requested_operations=[{"operation": "check_speaker_split"}],
    )
    results = execute_plan(plan, output_root=tmp_path / "processed")

    assert profile.quality_metrics["speaker_overlap_count"] == 1
    assert profile.quality_metrics["duplicate_trial_count"] == 1
    assert results[0].operation == "check_speaker_split"
    relaxed_policy = QualityPolicy(
        require_disjoint_speakers=False,
        require_disjoint_files=False,
        require_valid_trials=False,
    )
    relaxed = profile_dataset(dataset, relaxed_policy)
    assert relaxed.quality_metrics["error_count"] == 0
    relaxed_plan = build_processing_plan(relaxed, policy=relaxed_policy)
    relaxed_results = execute_plan(
        relaxed_plan, output_root=tmp_path / "relaxed"
    )
    assert relaxed_results[-1].status == "success"
    assert results[0].status == "failed"
    assert results[0].impact["affected_samples"] >= 1


def test_debug_subset_preview_version_hashes_and_quality_gate(tmp_path) -> None:
    dataset_dir = tmp_path / "audio"
    _write_test_wav(dataset_dir / "speaker_a" / "a.wav")
    _write_test_wav(dataset_dir / "speaker_a" / "b.wav")
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="audio")
    dataset.version = "source-v1"
    plan = build_processing_plan(
        profile_dataset(dataset),
        requested_operations=[
            {
                "operation": "build_debug_subset",
                "parameters": {"max_samples": 1, "max_per_speaker": 1},
            }
        ],
    )

    results = execute_plan(plan, output_root=tmp_path / "processed")
    version = publish_dataset_version(
        dataset, results, tmp_path / "version.json"
    )

    assert results[0].impact["affected_samples"] == 1
    assert "operation" not in results[0].impact
    assert results[0].impact["estimated_output_bytes"] > 0
    assert results[-1].operation == "quality_gate"
    assert version.quality_decision["training_allowed"] is True
    assert all(
        item["operation"] != "quality_gate" for item in version.operations
    )
    assert version.quality_policy["max_audio_files"] == 1000
    assert version.parent_version == "source-v1"
    assert "manifest.csv" in version.file_hashes
    assert any(name.endswith("a.wav") for name in version.file_hashes)
    assert version.operations[0]["parameters"] == {
        "max_samples": 1,
        "max_per_speaker": 1,
    }


def test_adjacent_audio_filters_share_one_duration_scan(tmp_path) -> None:
    dataset_dir = tmp_path / "audio"
    dataset_dir.mkdir()
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="audio")
    plan = DataProcessingPlan(
        dataset=dataset,
        operations=[
            DataOperation(operation="filter_unreadable_audio"),
            DataOperation(
                operation="filter_by_duration",
                parameters={"min_seconds": 0.2, "max_seconds": 10.0},
            ),
        ],
    )

    operations = _execution_operations(plan.operations)

    assert len(operations) == 1
    assert operations[0].operation == "filter_by_duration"
    assert operations[0].parameters["_combined_operations"] == [
        "filter_unreadable_audio",
        "filter_by_duration",
    ]


def test_execute_reuses_preview_selection(tmp_path) -> None:
    class CountingSubsetProcessor(_AudioSubsetProcessor):
        operation_name = "test_counting_subset"

        def __init__(self) -> None:
            self.scan_count = 0

        def _accept(self, path, parameters):
            self.scan_count += 1
            return True

    dataset_dir = tmp_path / "audio"
    dataset_dir.mkdir()
    (dataset_dir / "a.wav").write_bytes(b"a")
    (dataset_dir / "b.wav").write_bytes(b"b")
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="audio")
    parameters = {"_output_uri": str(tmp_path / "derived")}
    processor = CountingSubsetProcessor()

    impact = processor.preview(dataset, parameters)
    result = processor.execute(dataset, parameters)

    assert impact.scanned_samples == 2
    assert processor.scan_count == 2
    assert result.details["selection_reused"] is True
    assert result.details["materialization"]["generated"] == 1
    assert "selection_cache" not in impact.details


def test_hash_dataset_stops_at_limit(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for index in range(3):
        (dataset_dir / f"{index}.txt").write_text(
            str(index), encoding="utf-8"
        )

    hashes, complete = _hash_dataset(dataset_dir, 2)

    assert len(hashes) == 2
    assert complete is False


def test_hardlink_materialization_preserves_source_on_manifest_rewrite(
    tmp_path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    source_audio = dataset_dir / "sample.wav"
    _write_test_wav(source_audio)
    source_manifest = dataset_dir / "train.csv"
    original_manifest = "id,wav\na,sample.wav\na,sample.wav\n"
    source_manifest.write_text(original_manifest, encoding="utf-8")
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="tabular")
    plan = build_processing_plan(
        profile_dataset(dataset),
        requested_operations=[
            {
                "operation": "filter_manifest_rows",
                "parameters": {"materialize_complete_dataset": True},
            }
        ],
    )

    results = execute_plan(plan, output_root=tmp_path / "processed")
    output = Path(results[0].output_dataset_uri)

    assert source_manifest.read_text(encoding="utf-8") == original_manifest
    assert (output / "train.csv").read_text(encoding="utf-8").count("a,") == 1
    assert (output / "sample.wav").exists()
    materialization = results[0].details["materialization"]
    assert materialization["hardlink"] + materialization["copy"] == 1
    assert materialization["generated"] == 1


def test_materialize_file_hardlinks_or_safely_falls_back(tmp_path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "derived" / "source.bin"
    source.write_bytes(b"shared audio bytes")

    method = materialize_file(source, destination, "hardlink")

    assert method in {"hardlink", "copy"}
    assert destination.read_bytes() == source.read_bytes()
    if method == "hardlink":
        assert source.samefile(destination)
