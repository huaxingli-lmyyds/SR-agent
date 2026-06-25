from pathlib import Path

import csv
import pytest

from agent.data_processing.service import (
    build_processing_plan,
    execute_plan,
    infer_dataset_spec,
    profile_dataset,
    publish_dataset_version,
)
from agent.data_processing.contracts import DataOperationResult
from agent.data_processing.registry import register_processor


def test_generic_dataset_profile_plan_and_publish(tmp_path, dataset_dir) -> None:
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="text")
    profile = profile_dataset(dataset)
    plan = build_processing_plan(profile, target_goal="validate")
    results = execute_plan(plan)
    output = tmp_path / "versions" / "dataset.json"
    version = publish_dataset_version(dataset, results, output)

    assert profile.sample_count == 1
    assert results[-1].status == "success"
    assert version.dataset_id == dataset.dataset_id
    assert output.exists()


def test_manifest_quality_issue_creates_and_executes_optimization(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    manifest = dataset_dir / "train.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "wav"])
        writer.writeheader()
        writer.writerows([
            {"id": "a", "wav": "a.wav"},
            {"id": "a", "wav": "a.wav"},
            {"id": "b", "wav": ""},
        ])

    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="tabular")
    profile = profile_dataset(dataset)
    plan = build_processing_plan(profile, target_goal="clean manifests")
    results = execute_plan(plan, output_root=tmp_path / "processed")
    version = publish_dataset_version(dataset, results, tmp_path / "version.json")

    assert plan.operations[0].operation == "filter_manifest_rows"
    assert results[0].after_metrics["dropped_row_count"] == 2
    assert version.output_uri == results[0].output_dataset_uri
    assert version.consumer_uri is None
    assert version.consumption_status == "not_ready"
    assert version.artifacts[0]["name"] == "train.csv"
    assert Path(version.output_uri, "train.csv").exists()


def test_complete_derived_dataset_can_be_consumed_downstream(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "sample.wav").write_bytes(b"audio")
    (dataset_dir / "train.csv").write_text("id,wav\na,sample.wav\n", encoding="utf-8")
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="audio")
    profile = profile_dataset(dataset)
    plan = build_processing_plan(profile, requested_operations=[{
        "operation": "filter_manifest_rows",
        "parameters": {"materialize_complete_dataset": True},
    }])

    results = execute_plan(plan, output_root=tmp_path / "processed")
    version = publish_dataset_version(dataset, results, tmp_path / "version.json")

    assert version.consumer_uri == results[0].output_dataset_uri
    assert version.consumption_status == "ready"
    assert Path(version.consumer_uri, "sample.wav").exists()


def test_requested_operation_parameters_are_validated(dataset_dir) -> None:
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="text")
    profile = profile_dataset(dataset)

    with pytest.raises(ValueError, match="drop_empty_rows"):
        build_processing_plan(profile, requested_operations=[{
            "operation": "filter_manifest_rows",
            "parameters": {"drop_empty_rows": "yes"},
        }])


def test_advisor_cannot_enable_protected_materialization(dataset_dir) -> None:
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="text")
    profile = profile_dataset(dataset)

    plan = build_processing_plan(profile, requested_operations=[{
        "operation": "filter_manifest_rows",
        "parameters": {"materialize_complete_dataset": True},
        "_advisory": True,
    }])

    assert all(item.operation != "filter_manifest_rows" for item in plan.operations)
    assert plan.rejected_operations[0]["source"] == "advisor"


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
    plan = build_processing_plan(profile, requested_operations=[{
        "operation": "test_normalize_image_metadata",
        "parameters": {"color_mode": "L"},
    }])

    results = execute_plan(plan, output_root=tmp_path / "processed")

    assert results[0].details["color_mode"] == "L"


def test_dataset_scan_limit_must_be_positive(dataset_dir) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        infer_dataset_spec(str(dataset_dir), max_files=0)


def test_manifest_output_cannot_modify_source_tree(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.csv").write_text("id,wav\na,a.wav\n", encoding="utf-8")
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
        (split_dir / "manifest.csv").write_text("id,wav\na,a.wav\n", encoding="utf-8")
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="tabular")
    plan = build_processing_plan(
        profile_dataset(dataset),
        requested_operations=[{
            "operation": "filter_manifest_rows",
            "parameters": {"csv_glob": "**/*.csv"},
        }],
    )

    results = execute_plan(plan, output_root=tmp_path / "processed")

    assert results[0].status == "success"
    assert (Path(results[0].output_dataset_uri) / "train" / "manifest.csv").exists()
    assert (Path(results[0].output_dataset_uri) / "dev" / "manifest.csv").exists()
