"""Model-agnostic dataset inspection, planning, validation, and version tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from agent.data_processing import (
    PROCESSORS,
    build_processing_plan,
    execute_plan,
    infer_dataset_spec,
    plan_from_dict,
    profile_dataset,
    profile_from_dict,
    publish_dataset_version,
)
from agent.utils import ExperimentTracker, get_experiment_artifact_dir


def _payload(value: str) -> dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("JSON payload must be an object")
    return parsed


def _update_lifecycle(
    experiment_id: Optional[str],
    *,
    status: Optional[str] = None,
    metrics: Optional[dict] = None,
    artifacts: Optional[list] = None,
    lifecycle: Optional[dict] = None,
) -> None:
    if not experiment_id:
        return
    tracker = ExperimentTracker()
    record = tracker.get_experiment(experiment_id) or {}
    current_lifecycle = dict(
        ((record.get("extensions") or {}).get("data_lifecycle") or {})
    )
    current_lifecycle.update(lifecycle or {})
    tracker.update_experiment(
        experiment_id=experiment_id,
        experiment_type="data_processing",
        stage="data_preparation",
        status=status,
        metrics=metrics,
        artifacts=artifacts,
        extensions={"data_lifecycle": current_lifecycle},
    )


@tool
def InspectDataset(
    dataset_uri: str,
    dataset_id: Optional[str] = None,
    dataset_type: str = "auto",
    format_name: str = "auto",
    task_type: str = "generic",
    max_files: int = 10000,
    experiment_id: Optional[str] = None,
) -> str:
    """Inspect any local dataset and return a model-agnostic quality profile."""
    try:
        dataset = infer_dataset_spec(
            dataset_uri,
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            format_name=format_name,
            task_type=task_type,
            max_files=max_files,
        )
        profile = profile_dataset(dataset)
        profile_data = profile.to_dict()
        _update_lifecycle(
            experiment_id,
            status="running",
            metrics={"quality_before": profile.quality_metrics},
            lifecycle={"dataset": dataset.to_dict(), "profile_before": profile_data},
        )
        return json.dumps(profile_data, ensure_ascii=False, default=str)
    except Exception as exc:
        _update_lifecycle(experiment_id, status="failed", lifecycle={"inspect_error": str(exc)})
        return json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)


@tool
def BuildDataProcessingPlan(
    profile_json: str,
    target_goal: str = "",
    experiment_id: Optional[str] = None,
) -> str:
    """Build an auditable processing plan from a dataset quality profile."""
    try:
        profile = profile_from_dict(_payload(profile_json))
        plan = build_processing_plan(profile, target_goal)
        plan_data = plan.to_dict()
        _update_lifecycle(experiment_id, lifecycle={"plan": plan_data})
        return json.dumps(plan_data, ensure_ascii=False, default=str)
    except Exception as exc:
        _update_lifecycle(experiment_id, status="failed", lifecycle={"plan_error": str(exc)})
        return json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)


@tool
def ExecuteDataProcessingPlan(
    plan_json: str,
    experiment_id: Optional[str] = None,
) -> str:
    """Execute registered operations in a processing plan and validate results."""
    try:
        plan = plan_from_dict(_payload(plan_json))
        results = execute_plan(plan)
        result_data = [result.to_dict() for result in results]
        final_metrics = results[-1].after_metrics if results else {}
        failed = next((result for result in results if result.status == "failed"), None)
        _update_lifecycle(
            experiment_id,
            status="failed" if failed else "running",
            metrics={"quality_after": final_metrics},
            lifecycle={"operation_results": result_data},
        )
        return json.dumps({
            "status": "failed" if failed else "success",
            "results": result_data,
            "error": failed.error if failed else None,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        _update_lifecycle(experiment_id, status="failed", lifecycle={"execution_error": str(exc)})
        return json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)


@tool
def PublishDatasetVersion(
    plan_json: str,
    operation_results_json: str,
    experiment_id: str,
    parent_version: Optional[str] = None,
) -> str:
    """Publish validated dataset lineage metadata without overwriting source data."""
    try:
        plan = plan_from_dict(_payload(plan_json))
        result_payload = json.loads(operation_results_json)
        result_items = (
            result_payload.get("results", [])
            if isinstance(result_payload, dict)
            else result_payload
        )
        if not isinstance(result_items, list):
            raise ValueError("operation_results_json must contain a result list")
        from agent.data_processing.service import result_from_dict

        results = [result_from_dict(item) for item in result_items]
        output_dir = get_experiment_artifact_dir(
            experiment_id, "dataset_versions", "data_processing", create=True
        )
        pending_path = Path(output_dir) / "dataset_version.json"
        version = publish_dataset_version(
            plan.dataset,
            results,
            pending_path,
            parent_version=parent_version,
        )
        final_path = Path(output_dir) / f"{version.version}.json"
        pending_path.replace(final_path)
        artifact = {
            "type": "dataset_version",
            "name": version.version,
            "path": str(final_path),
            "metadata": {"dataset_id": version.dataset_id},
        }
        _update_lifecycle(
            experiment_id,
            status="success",
            artifacts=[artifact],
            lifecycle={"published_version": version.to_dict()},
        )
        return json.dumps(version.to_dict(), ensure_ascii=False, default=str)
    except Exception as exc:
        _update_lifecycle(experiment_id, status="failed", lifecycle={"publish_error": str(exc)})
        return json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)


@tool
def ListDataProcessors() -> str:
    """List registered data processing operations and supported dataset types."""
    return json.dumps(PROCESSORS.describe(), ensure_ascii=False)


__all__ = [
    "InspectDataset",
    "BuildDataProcessingPlan",
    "ExecuteDataProcessingPlan",
    "PublishDatasetVersion",
    "ListDataProcessors",
]
