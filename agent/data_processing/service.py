"""Dataset profiling, planning, validation, and lineage services."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .audio import register_audio_components
from .contracts import (
    DataIssue,
    DataOperation,
    DataOperationResult,
    DataProcessingPlan,
    DataProfile,
    DatasetSpec,
    DatasetVersion,
    OperationImpact,
    QualityPolicy,
)
from .materialization import (
    materialize_tree,
    validate_materialization_mode,
)
from .quality import DefaultQualityGate
from .registry import PROCESSORS, PROFILERS, register_processor

TYPE_EXTENSIONS = {
    "audio": {".wav", ".flac", ".mp3", ".ogg", ".m4a"},
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"},
    "text": {".txt", ".jsonl", ".json", ".csv", ".tsv", ".md"},
    "tabular": {".csv", ".tsv", ".parquet", ".xlsx", ".xls"},
}

AUDIT_OPERATIONS = {
    "validate_dataset",
    "validate_audio_files",
    "check_speaker_split",
    "validate_verification_trials",
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
    if (
        isinstance(max_files, bool)
        or not isinstance(max_files, int)
        or max_files <= 0
    ):
        raise ValueError("max_files must be a positive integer")
    source = Path(source_uri).resolve()
    if dataset_type == "auto":
        extensions = Counter(
            path.suffix.lower()
            for path in _iter_files(source, max_files)
            if path.suffix
        )
        scores = {
            name: sum(
                count for ext, count in extensions.items() if ext in supported
            )
            for name, supported in TYPE_EXTENSIONS.items()
        }
        dataset_type = (
            max(scores, key=scores.get)
            if scores and max(scores.values())
            else "generic"
        )
    if format_name == "auto":
        format_name = (
            source.suffix.lower().lstrip(".")
            if source.is_file()
            else "directory"
        )
    return DatasetSpec(
        dataset_id=dataset_id or source.stem or source.name,
        dataset_type=dataset_type,
        source_uri=str(source),
        format=format_name,
        task_type=task_type,
        metadata={"profile_scan_limit": max_files},
    )


def profile_dataset(
    dataset: DatasetSpec,
    policy: Optional[QualityPolicy] = None,
) -> DataProfile:
    active_policy = policy or QualityPolicy()
    profiler = PROFILERS.get(dataset.dataset_type)
    return (
        profiler.profile(dataset, active_policy)
        if profiler
        else _profile_generic_dataset(dataset)
    )


def _operation_profile(
    dataset: DatasetSpec, parameters: Dict[str, Any]
) -> DataProfile:
    cached = parameters.get("_preview_profile")
    if isinstance(cached, DataProfile):
        return cached
    profile = profile_dataset(
        dataset, QualityPolicy.from_dict(parameters.get("_quality_policy"))
    )
    parameters["_preview_profile"] = profile
    return profile


def _profile_generic_dataset(dataset: DatasetSpec) -> DataProfile:
    source = Path(dataset.source_uri)
    scan_limit = int(dataset.metadata.get("profile_scan_limit", 10000))
    files = list(_iter_files(source, scan_limit))
    extensions = Counter(path.suffix.lower() or "<none>" for path in files)
    sizes = [path.stat().st_size for path in files if path.exists()]
    empty_files = [
        str(path)
        for path in files
        if path.exists() and path.stat().st_size == 0
    ]
    issues: List[DataIssue] = []

    if not source.exists():
        issues.append(
            DataIssue(
                code="source_missing",
                severity="error",
                message="Dataset source does not exist.",
                evidence={"source_uri": dataset.source_uri},
                suggested_operation="validate_dataset",
            )
        )
    if not files:
        issues.append(
            DataIssue(
                code="dataset_empty",
                severity="error",
                message="No files were found in the dataset source.",
                suggested_operation="validate_dataset",
            )
        )
    if empty_files:
        issues.append(
            DataIssue(
                code="empty_files",
                severity="warning",
                message="Empty files were found.",
                evidence={
                    "count": len(empty_files),
                    "examples": empty_files[:20],
                },
                suggested_operation="validate_dataset",
            )
        )

    manifest_profile = _profile_manifest(source)
    issues.extend(manifest_profile["issues"])
    duplicate_count = _count_duplicate_paths(files)
    if duplicate_count:
        issues.append(
            DataIssue(
                code="duplicate_file_names",
                severity="warning",
                message="Duplicate file names were found in different directories.",
                evidence={"count": duplicate_count},
                suggested_operation="validate_dataset",
            )
        )

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


def build_processing_plan(
    profile: DataProfile,
    target_goal: str = "",
    requested_operations: Optional[List[Dict[str, Any]]] = None,
    policy: Optional[QualityPolicy] = None,
) -> DataProcessingPlan:
    operations: List[DataOperation] = []
    rejected_operations: List[Dict[str, Any]] = []
    seen = set()
    for requested in requested_operations or []:
        operation_name = str(requested.get("operation") or "").strip()
        if not operation_name or operation_name in seen:
            continue
        advisory = bool(requested.get("_advisory"))
        try:
            processor = PROCESSORS.get(
                operation_name, profile.dataset.dataset_type
            )
            parameters = dict(requested.get("parameters") or {})
            if advisory:
                for name, rule in getattr(
                    processor, "parameter_schema", {}
                ).items():
                    if rule.get("advisor_allowed", True) is False:
                        value = parameters.get(name, rule.get("default"))
                        if value != rule.get("default"):
                            raise ValueError(
                                f"advisor cannot change protected parameter: {name}"
                            )
            validation_parameters = dict(parameters)
            validation_parameters["_output_uri"] = str(
                Path(profile.dataset.source_uri).resolve().parent
                / ".data-processing-preview"
            )
            processor.validate(profile.dataset, validation_parameters)
        except (KeyError, ValueError) as exc:
            if not advisory:
                raise
            rejected_operations.append(
                {
                    "operation": operation_name,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "source": "llm",
                }
            )
            continue
        seen.add(operation_name)
        operations.append(
            DataOperation(
                operation=operation_name,
                parameters=parameters,
                reason=str(
                    requested.get("reason")
                    or "Requested by the data processing policy."
                ),
            )
        )
    for issue in profile.issues:
        operation_name = issue.suggested_operation
        if not operation_name or operation_name in AUDIT_OPERATIONS:
            continue
        if operation_name in seen:
            continue
        seen.add(operation_name)
        operations.append(
            DataOperation(
                operation=operation_name,
                reason=issue.message,
            )
        )
    return DataProcessingPlan(
        dataset=profile.dataset,
        operations=operations,
        rejected_operations=rejected_operations,
        quality_policy=(policy or QualityPolicy()).to_dict(),
        target_goal=target_goal,
    )


class ValidateDatasetProcessor:
    operation_name = "validate_dataset"
    supported_data_types = {"*"}
    parameter_schema: Dict[str, Any] = {}

    def validate(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> None:
        if not dataset.source_uri:
            raise ValueError("dataset source_uri is required")

    def preview(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> OperationImpact:
        profile = _operation_profile(dataset, parameters)
        affected = int(profile.quality_metrics.get("error_count", 0))
        return OperationImpact(
            self.operation_name, profile.sample_count, affected
        )

    def execute(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> DataOperationResult:
        self.validate(dataset, parameters)
        profile = _operation_profile(dataset, parameters)
        return DataOperationResult(
            status="failed"
            if profile.quality_metrics["error_count"]
            else "success",
            operation=self.operation_name,
            after_metrics=profile.quality_metrics,
            error="dataset validation failed"
            if profile.quality_metrics["error_count"]
            else None,
        )


class FilterManifestRowsProcessor:
    """Create a derived CSV dataset while preserving the original source."""

    operation_name = "filter_manifest_rows"
    supported_data_types = {"*"}
    parameter_schema = {
        "drop_empty_rows": {"type": "boolean", "default": True},
        "deduplicate_rows": {"type": "boolean", "default": True},
        "csv_glob": {"type": "string", "default": "*.csv"},
        "materialization_mode": {
            "type": "string",
            "default": "hardlink",
            "description": (
                "Reuse immutable dataset bytes; fall back to copy when linking "
                "is unavailable."
            ),
        },
        "materialize_complete_dataset": {
            "type": "boolean",
            "default": False,
            "advisor_allowed": False,
        },
    }

    def validate(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> None:
        for name in (
            "drop_empty_rows",
            "deduplicate_rows",
            "materialize_complete_dataset",
        ):
            if name in parameters and not isinstance(parameters[name], bool):
                raise ValueError(f"{name} must be a boolean")
        if "csv_glob" in parameters and not isinstance(
            parameters["csv_glob"], str
        ):
            raise ValueError("csv_glob must be a string")
        validate_materialization_mode(
            str(parameters.get("materialization_mode", "hardlink"))
        )
        if ".." in str(parameters.get("csv_glob") or ""):
            raise ValueError("csv_glob cannot traverse outside the dataset")

    def preview(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> OperationImpact:
        self.validate(dataset, parameters)
        source = Path(dataset.source_uri)
        pattern = str(parameters.get("csv_glob") or "*.csv")
        candidates = (
            [source] if source.is_file() else list(source.glob(pattern))
        )
        rows = 0
        for path in candidates:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                rows += sum(1 for _ in reader)
        estimated = sum(path.stat().st_size for path in candidates)
        mode = validate_materialization_mode(
            str(parameters.get("materialization_mode", "hardlink"))
        )
        if (
            parameters.get("materialize_complete_dataset")
            and source.is_dir()
            and mode == "copy"
        ):
            estimated = sum(
                path.stat().st_size
                for path in source.rglob("*")
                if path.is_file()
            )
        return OperationImpact(
            self.operation_name,
            rows,
            rows,
            estimated,
            True,
            {"materialization_mode": mode},
        )

    def execute(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> DataOperationResult:
        self.validate(dataset, parameters)
        source = Path(dataset.source_uri)
        if not parameters.get("_output_uri"):
            return DataOperationResult(
                status="failed",
                operation=self.operation_name,
                error="filter_manifest_rows requires an execution output directory",
            )
        output = Path(str(parameters["_output_uri"]))
        materialize = bool(
            parameters.get("materialize_complete_dataset", False)
        )
        mode = validate_materialization_mode(
            str(parameters.get("materialization_mode", "hardlink"))
        )
        resolved_source = source.resolve()
        resolved_output = output.resolve()
        if resolved_output == resolved_source or (
            source.is_dir() and resolved_source in resolved_output.parents
        ):
            return DataOperationResult(
                status="failed",
                operation=self.operation_name,
                error="output directory must be outside the source dataset",
            )
        output.mkdir(parents=True, exist_ok=True)
        materialization = {"hardlink": 0, "copy": 0}
        if materialize and source.is_dir():
            materialization = materialize_tree(source, output, mode)
        materialization["generated"] = 0
        pattern = str(parameters.get("csv_glob") or "*.csv")
        candidates = (
            [source]
            if source.is_file() and source.suffix.lower() == ".csv"
            else list(source.glob(pattern))
        )
        before_rows = after_rows = dropped_rows = 0
        artifacts: List[Dict[str, Any]] = []
        for path in candidates:
            relative_path = (
                path.name if source.is_file() else path.relative_to(source)
            )
            destination = output / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if materialize and destination.exists():
                method = "copy"
                try:
                    if path.samefile(destination):
                        method = "hardlink"
                except OSError:
                    pass
                destination.unlink()
                materialization[method] = max(
                    0, materialization[method] - 1
                )
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames or []
                rows = list(reader)
            before_rows += len(rows)
            cleaned = []
            fingerprints = set()
            for row in rows:
                if parameters.get("drop_empty_rows", True) and any(
                    value is None or str(value).strip() == ""
                    for value in row.values()
                ):
                    dropped_rows += 1
                    continue
                fingerprint = json.dumps(
                    row, sort_keys=True, ensure_ascii=False
                )
                if (
                    parameters.get("deduplicate_rows", True)
                    and fingerprint in fingerprints
                ):
                    dropped_rows += 1
                    continue
                fingerprints.add(fingerprint)
                cleaned.append(row)
            with destination.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(cleaned)
            materialization["generated"] += 1
            after_rows += len(cleaned)
            artifacts.append(
                {
                    "type": "manifest",
                    "name": relative_path.as_posix()
                    if isinstance(relative_path, Path)
                    else relative_path,
                    "path": str(destination),
                }
            )
        if not candidates:
            return DataOperationResult(
                status="failed",
                operation=self.operation_name,
                error=f"no CSV manifests matched {pattern}",
            )
        return DataOperationResult(
            status="success",
            operation=self.operation_name,
            output_dataset_uri=str(output),
            consumer_ready=materialize,
            before_metrics={"manifest_row_count": before_rows},
            after_metrics={
                "manifest_row_count": after_rows,
                "dropped_row_count": dropped_rows,
                "error_count": 0,
            },
            artifacts=artifacts,
            details={
                "materialization_mode": mode,
                "materialization": materialization,
            },
        )


def _execution_operations(
    operations: List[DataOperation],
) -> List[DataOperation]:
    """Combine adjacent audio filters that share the same metadata scan."""

    optimized: List[DataOperation] = []
    index = 0
    pair = {"filter_unreadable_audio", "filter_by_duration"}
    while index < len(operations):
        current = operations[index]
        following = (
            operations[index + 1] if index + 1 < len(operations) else None
        )
        if following and {current.operation, following.operation} == pair:
            duration = (
                current
                if current.operation == "filter_by_duration"
                else following
            )
            unreadable = following if duration is current else current
            parameters = dict(duration.parameters)
            if (
                "materialization_mode" not in parameters
                and "materialization_mode" in unreadable.parameters
            ):
                parameters["materialization_mode"] = unreadable.parameters[
                    "materialization_mode"
                ]
            parameters["_combined_operations"] = [
                current.operation,
                following.operation,
            ]
            optimized.append(
                DataOperation(
                    operation="filter_by_duration",
                    parameters=parameters,
                    reason="Combined readability and duration filtering.",
                )
            )
            index += 2
            continue
        optimized.append(current)
        index += 1
    return optimized


def execute_plan(
    plan: DataProcessingPlan,
    *,
    output_root: Optional[Path] = None,
    initial_profile: Optional[DataProfile] = None,
) -> List[DataOperationResult]:
    results: List[DataOperationResult] = []
    current_dataset = plan.dataset
    dataset_changed = False
    policy = QualityPolicy.from_dict(plan.quality_policy)
    operations = _execution_operations(plan.operations)
    for index, operation in enumerate(operations):
        processor = PROCESSORS.get(
            operation.operation, current_dataset.dataset_type
        )
        parameters = dict(operation.parameters)
        parameters["_quality_policy"] = policy.to_dict()
        if output_root is not None:
            parameters["_output_uri"] = str(
                output_root / f"{index:02d}-{operation.operation}"
            )
        processor.validate(current_dataset, parameters)
        preview_method = getattr(processor, "preview", None)
        impact = (
            preview_method(current_dataset, parameters)
            if callable(preview_method)
            else OperationImpact(operation.operation)
        )
        result = processor.execute(current_dataset, parameters)
        result.impact = impact.to_dict()
        result.impact.pop("operation", None)
        result.parameters = {
            key: value
            for key, value in parameters.items()
            if not key.startswith("_")
        }
        combined = parameters.get("_combined_operations")
        if combined:
            result.details["combined_operations"] = list(combined)
        results.append(result)
        if result.status == "failed":
            break
        if result.output_dataset_uri:
            next_dataset = DatasetSpec(
                **{
                    **current_dataset.to_dict(),
                    "source_uri": result.output_dataset_uri,
                    "version": current_dataset.version,
                }
            )
            if result.consumer_ready:
                current_dataset = next_dataset
                dataset_changed = True

    if all(result.status != "failed" for result in results):
        final_profile = (
            initial_profile
            if initial_profile is not None and not dataset_changed
            else profile_dataset(current_dataset, policy)
        )
        decision = DefaultQualityGate(policy).evaluate(final_profile)
        results.append(
            DataOperationResult(
                status="success" if decision.training_allowed else "failed",
                operation="quality_gate",
                after_metrics=final_profile.quality_metrics,
                details={
                    "decision": decision.to_dict(),
                    "quality_policy": policy.to_dict(),
                },
                error=None
                if decision.training_allowed
                else "quality gate blocked downstream training",
            )
        )

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
        raise ValueError(
            "cannot publish a dataset version without operation results"
        )
    failed = [result for result in result_list if result.status == "failed"]
    if failed:
        raise ValueError(
            "cannot publish a dataset version with failed operations"
        )
    version = _version_id(dataset, result_list)
    output_uri = next(
        (
            result.output_dataset_uri
            for result in reversed(result_list)
            if result.output_dataset_uri
        ),
        dataset.source_uri,
    )
    output_results = [
        result for result in result_list if result.output_dataset_uri
    ]
    final_output = output_results[-1] if output_results else None
    if final_output is None:
        consumer_uri = dataset.source_uri
        consumption_status = "source_unchanged"
        consumption_reason = "No operation produced a derived dataset; use the validated source dataset."
    elif final_output.consumer_ready:
        consumer_uri = final_output.output_dataset_uri
        consumption_status = "ready"
        consumption_reason = (
            "The final derived dataset is marked consumer-ready."
        )
    else:
        consumer_uri = None
        consumption_status = "not_ready"
        consumption_reason = (
            "Data processing produced derived artifacts, but the final output is not a complete "
            "dataset that downstream training can consume."
        )
    gate_result = next(
        (
            item
            for item in reversed(result_list)
            if item.operation == "quality_gate"
        ),
        None,
    )
    quality_decision = dict(
        (gate_result.details if gate_result else {}).get("decision") or {}
    )
    policy_data = (gate_result.details if gate_result else {}).get(
        "quality_policy"
    ) or {}
    hash_limit = QualityPolicy.from_dict(policy_data).hash_file_limit
    file_hashes, hashes_complete = _hash_dataset(Path(output_uri), hash_limit)
    record = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version=version,
        source_uri=dataset.source_uri,
        output_uri=output_uri,
        consumer_uri=consumer_uri,
        consumption_status=consumption_status,
        consumption_reason=consumption_reason,
        parent_version=parent_version or dataset.version,
        operations=[
            result.to_dict()
            for result in result_list
            if result.operation != "quality_gate"
        ],
        quality_policy=dict(policy_data),
        quality_metrics=dict(gate_result.after_metrics) if gate_result else {},
        quality_decision=quality_decision,
        file_hashes=file_hashes,
        hashes_complete=hashes_complete,
        created_at=datetime.now().isoformat(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return record


def dataset_spec_from_dict(data: Dict[str, Any]) -> DatasetSpec:
    fields = DatasetSpec.__dataclass_fields__
    return DatasetSpec(
        **{key: value for key, value in data.items() if key in fields}
    )


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
        operations=[
            DataOperation(
                operation=item["operation"],
                parameters=item.get("parameters") or {},
                reason=item.get("reason") or "",
            )
            for item in data.get("operations") or []
        ],
        rejected_operations=data.get("rejected_operations") or [],
        quality_policy=data.get("quality_policy") or {},
        target_goal=data.get("target_goal") or "",
    )


def result_from_dict(data: Dict[str, Any]) -> DataOperationResult:
    fields = DataOperationResult.__dataclass_fields__
    return DataOperationResult(
        **{key: value for key, value in data.items() if key in fields}
    )


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
    candidates = (
        [source]
        if source.is_file()
        else list(source.glob("*.csv"))
        if source.is_dir()
        else []
    )
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
                    if any(
                        value is None or str(value).strip() == ""
                        for value in row.values()
                    ):
                        local_invalid += 1
                row_count += local_rows
                invalid_row_count += local_invalid
                details["files"].append(
                    {
                        "path": str(path),
                        "rows": local_rows,
                        "invalid_rows": local_invalid,
                    }
                )
        except (OSError, csv.Error, UnicodeError) as exc:
            issues.append(
                DataIssue(
                    code="manifest_unreadable",
                    severity="error",
                    message=f"Manifest cannot be read: {path.name}",
                    evidence={"path": str(path), "error": str(exc)},
                    suggested_operation="validate_dataset",
                )
            )
    if invalid_row_count:
        issues.append(
            DataIssue(
                code="manifest_invalid_rows",
                severity="warning",
                message="Manifest rows with missing values were found.",
                evidence={"count": invalid_row_count},
                suggested_operation="filter_manifest_rows",
            )
        )
    return {
        "schema": schema,
        "row_count": row_count,
        "invalid_row_count": invalid_row_count,
        "issues": issues,
        "details": details,
    }


def _hash_dataset(source: Path, limit: int) -> tuple[Dict[str, str], bool]:
    if not source.exists():
        return {}, False
    hashes: Dict[str, str] = {}
    complete = True
    for index, path in enumerate(_iter_files(source)):
        if index >= limit:
            complete = False
            break
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        name = (
            source.name
            if source.is_file()
            else path.relative_to(source).as_posix()
        )
        hashes[name] = digest.hexdigest()
    return hashes, complete


def _version_id(
    dataset: DatasetSpec, results: List[DataOperationResult]
) -> str:
    payload = json.dumps(
        {
            "dataset": dataset.to_dict(),
            "results": [result.to_dict() for result in results],
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return f"{dataset.dataset_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{digest}"


register_processor(ValidateDatasetProcessor())
register_processor(FilterManifestRowsProcessor())
register_audio_components()
