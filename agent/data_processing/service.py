"""Dataset profiling, planning, validation, and lineage services."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .contracts import (
    DataIssue,
    DataOperation,
    DataOperationResult,
    DataProcessingPlan,
    DataProfile,
    DatasetSpec,
    DatasetVersion,
)
from .registry import PROCESSORS, register_processor


TYPE_EXTENSIONS = {
    "audio": {".wav", ".flac", ".mp3", ".ogg", ".m4a"},
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"},
    "text": {".txt", ".jsonl", ".json", ".csv", ".tsv", ".md"},
    "tabular": {".csv", ".tsv", ".parquet", ".xlsx", ".xls"},
}


def infer_dataset_spec(
    source_uri: str,
    *,
    dataset_id: Optional[str] = None,
    dataset_type: str = "auto",
    format_name: str = "auto",
    task_type: str = "generic",
    max_files: int = 10000,
) -> DatasetSpec:
    source = Path(source_uri).resolve()
    if dataset_type == "auto":
        extensions = Counter(
            path.suffix.lower()
            for path in _iter_files(source, max_files)
            if path.suffix
        )
        scores = {
            name: sum(count for ext, count in extensions.items() if ext in supported)
            for name, supported in TYPE_EXTENSIONS.items()
        }
        dataset_type = max(scores, key=scores.get) if scores and max(scores.values()) else "generic"
    if format_name == "auto":
        format_name = source.suffix.lower().lstrip(".") if source.is_file() else "directory"
    return DatasetSpec(
        dataset_id=dataset_id or source.stem or source.name,
        dataset_type=dataset_type,
        source_uri=str(source),
        format=format_name,
        task_type=task_type,
        metadata={"profile_scan_limit": max_files},
    )


def profile_dataset(dataset: DatasetSpec) -> DataProfile:
    source = Path(dataset.source_uri)
    scan_limit = int(dataset.metadata.get("profile_scan_limit", 10000))
    files = list(_iter_files(source, scan_limit))
    extensions = Counter(path.suffix.lower() or "<none>" for path in files)
    sizes = [path.stat().st_size for path in files if path.exists()]
    empty_files = [str(path) for path in files if path.exists() and path.stat().st_size == 0]
    issues: List[DataIssue] = []

    if not source.exists():
        issues.append(DataIssue(
            code="source_missing",
            severity="error",
            message="Dataset source does not exist.",
            evidence={"source_uri": dataset.source_uri},
            suggested_operation="validate_dataset",
        ))
    if not files:
        issues.append(DataIssue(
            code="dataset_empty",
            severity="error",
            message="No files were found in the dataset source.",
            suggested_operation="validate_dataset",
        ))
    if empty_files:
        issues.append(DataIssue(
            code="empty_files",
            severity="warning",
            message="Empty files were found.",
            evidence={"count": len(empty_files), "examples": empty_files[:20]},
            suggested_operation="validate_dataset",
        ))

    manifest_profile = _profile_manifest(source)
    issues.extend(manifest_profile["issues"])
    duplicate_count = _count_duplicate_paths(files)
    if duplicate_count:
        issues.append(DataIssue(
            code="duplicate_file_names",
            severity="warning",
            message="Duplicate file names were found in different directories.",
            evidence={"count": duplicate_count},
            suggested_operation="validate_dataset",
        ))

    total_size = sum(sizes)
    quality_metrics = {
        "missing_source": not source.exists(),
        "empty_file_count": len(empty_files),
        "empty_file_ratio": len(empty_files) / len(files) if files else 0.0,
        "duplicate_file_name_count": duplicate_count,
        "manifest_invalid_row_count": manifest_profile["invalid_row_count"],
        "issue_count": len(issues),
        "error_count": sum(issue.severity == "error" for issue in issues),
        "warning_count": sum(issue.severity == "warning" for issue in issues),
    }
    return DataProfile(
        dataset=dataset,
        sample_count=manifest_profile["row_count"] or len(files),
        schema=manifest_profile["schema"],
        distributions={
            "file_extensions": dict(extensions),
            "file_size_bytes": {
                "total": total_size,
                "min": min(sizes) if sizes else 0,
                "max": max(sizes) if sizes else 0,
                "average": total_size / len(sizes) if sizes else 0,
            },
        },
        quality_metrics=quality_metrics,
        issues=issues,
        extensions={
            "filesystem": {
                "scanned_file_count": len(files),
                "scan_limit": scan_limit,
                "scan_limited": len(files) >= scan_limit,
            },
            "manifest": manifest_profile["details"],
        },
    )


def build_processing_plan(profile: DataProfile, target_goal: str = "") -> DataProcessingPlan:
    operations: List[DataOperation] = []
    seen = set()
    for issue in profile.issues:
        operation_name = issue.suggested_operation or "validate_dataset"
        if operation_name in seen:
            continue
        seen.add(operation_name)
        operations.append(DataOperation(
            operation=operation_name,
            reason=issue.message,
            expected_effect={"resolve_issue": issue.code},
        ))
    if not operations:
        operations.append(DataOperation(
            operation="validate_dataset",
            reason="Confirm the dataset remains valid before publishing a version.",
            expected_effect={"error_count": 0},
        ))
    return DataProcessingPlan(
        dataset=profile.dataset,
        operations=operations,
        validation_rules=[{"metric": "error_count", "operator": "eq", "value": 0}],
        target_goal=target_goal,
    )


class ValidateDatasetProcessor:
    operation_name = "validate_dataset"
    supported_data_types = {"*"}

    def validate(self, dataset: DatasetSpec, parameters: Dict[str, Any]) -> None:
        if not dataset.source_uri:
            raise ValueError("dataset source_uri is required")

    def execute(self, dataset: DatasetSpec, parameters: Dict[str, Any]) -> DataOperationResult:
        self.validate(dataset, parameters)
        profile = profile_dataset(dataset)
        return DataOperationResult(
            status="failed" if profile.quality_metrics["error_count"] else "success",
            operation=self.operation_name,
            input_dataset_version=dataset.version,
            before_metrics=profile.quality_metrics,
            after_metrics=profile.quality_metrics,
            details={"profile": profile.to_dict()},
            error="dataset validation failed" if profile.quality_metrics["error_count"] else None,
        )


def execute_plan(plan: DataProcessingPlan) -> List[DataOperationResult]:
    results = []
    for operation in plan.operations:
        processor = PROCESSORS.get(operation.operation, plan.dataset.dataset_type)
        processor.validate(plan.dataset, operation.parameters)
        result = processor.execute(plan.dataset, operation.parameters)
        results.append(result)
        if result.status == "failed":
            break
    validation_error = _validate_plan_results(plan, results)
    if validation_error:
        metrics = results[-1].after_metrics if results else {}
        results.append(DataOperationResult(
            status="failed",
            operation="validate_plan",
            before_metrics=metrics,
            after_metrics=metrics,
            error=validation_error,
        ))
    return results


def publish_dataset_version(
    dataset: DatasetSpec,
    results: Iterable[DataOperationResult],
    output_path: Path,
    *,
    parent_version: Optional[str] = None,
) -> DatasetVersion:
    result_list = list(results)
    if not result_list:
        raise ValueError("cannot publish a dataset version without operation results")
    failed = [result for result in result_list if result.status == "failed"]
    if failed:
        raise ValueError("cannot publish a dataset version with failed operations")
    version = _version_id(dataset, result_list)
    record = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version=version,
        source_uri=dataset.source_uri,
        parent_version=parent_version or dataset.version,
        operations=[result.to_dict() for result in result_list],
        quality_metrics=result_list[-1].after_metrics if result_list else {},
        created_at=datetime.now().isoformat(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _validate_plan_results(
    plan: DataProcessingPlan,
    results: List[DataOperationResult],
) -> Optional[str]:
    if not results or results[-1].status == "failed":
        return None
    metrics = results[-1].after_metrics
    for rule in plan.validation_rules:
        metric = rule.get("metric")
        operator = rule.get("operator")
        expected = rule.get("value")
        actual = metrics.get(metric)
        passed = {
            "eq": actual == expected,
            "lte": actual is not None and actual <= expected,
            "gte": actual is not None and actual >= expected,
        }.get(operator)
        if passed is None:
            return f"unsupported validation operator: {operator}"
        if not passed:
            return f"validation failed: {metric} {operator} {expected}; actual={actual}"
    return None


def dataset_spec_from_dict(data: Dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(**data)


def profile_from_dict(data: Dict[str, Any]) -> DataProfile:
    return DataProfile(
        dataset=dataset_spec_from_dict(data["dataset"]),
        sample_count=data.get("sample_count", 0),
        schema=data.get("schema") or {},
        distributions=data.get("distributions") or {},
        quality_metrics=data.get("quality_metrics") or {},
        issues=[DataIssue(**item) for item in data.get("issues") or []],
        extensions=data.get("extensions") or {},
    )


def plan_from_dict(data: Dict[str, Any]) -> DataProcessingPlan:
    return DataProcessingPlan(
        dataset=dataset_spec_from_dict(data["dataset"]),
        operations=[DataOperation(**item) for item in data.get("operations") or []],
        validation_rules=data.get("validation_rules") or [],
        target_goal=data.get("target_goal") or "",
    )


def result_from_dict(data: Dict[str, Any]) -> DataOperationResult:
    return DataOperationResult(**data)


def _iter_files(source: Path, limit: Optional[int] = None) -> Iterable[Path]:
    if source.is_file():
        yield source
    elif source.is_dir():
        count = 0
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            yield path
            count += 1
            if limit is not None and count >= limit:
                break


def _count_duplicate_paths(files: List[Path]) -> int:
    names = Counter(path.name.lower() for path in files)
    return sum(count - 1 for count in names.values() if count > 1)


def _profile_manifest(source: Path) -> Dict[str, Any]:
    candidates = [source] if source.is_file() else list(source.glob("*.csv")) if source.is_dir() else []
    schema: Dict[str, Any] = {}
    row_count = 0
    invalid_row_count = 0
    issues: List[DataIssue] = []
    details: Dict[str, Any] = {"files": []}
    for path in candidates[:20]:
        if path.suffix.lower() != ".csv":
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames or []
                schema[path.name] = fields
                local_rows = 0
                local_invalid = 0
                for row in reader:
                    local_rows += 1
                    if any(value is None or str(value).strip() == "" for value in row.values()):
                        local_invalid += 1
                row_count += local_rows
                invalid_row_count += local_invalid
                details["files"].append({
                    "path": str(path),
                    "rows": local_rows,
                    "invalid_rows": local_invalid,
                })
        except (OSError, csv.Error, UnicodeError) as exc:
            issues.append(DataIssue(
                code="manifest_unreadable",
                severity="error",
                message=f"Manifest cannot be read: {path.name}",
                evidence={"path": str(path), "error": str(exc)},
                suggested_operation="validate_dataset",
            ))
    if invalid_row_count:
        issues.append(DataIssue(
            code="manifest_invalid_rows",
            severity="warning",
            message="Manifest rows with missing values were found.",
            evidence={"count": invalid_row_count},
            suggested_operation="validate_dataset",
        ))
    return {
        "schema": schema,
        "row_count": row_count,
        "invalid_row_count": invalid_row_count,
        "issues": issues,
        "details": details,
    }


def _version_id(dataset: DatasetSpec, results: List[DataOperationResult]) -> str:
    payload = json.dumps(
        {"dataset": dataset.to_dict(), "results": [result.to_dict() for result in results]},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return f"{dataset.dataset_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{digest}"


register_processor(ValidateDatasetProcessor())
