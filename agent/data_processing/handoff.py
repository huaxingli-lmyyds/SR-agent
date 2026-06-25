"""Validated dataset handoff from data processing to downstream agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agent.utils.path_tool import is_remote_path, resolve_data_path


def build_data_handoff(version: Dict[str, Any], experiment_id: Any = None) -> Dict[str, Any]:
    """Build the stable handoff payload exposed by data-processing agents."""

    return {
        "dataset_id": version.get("dataset_id"),
        "dataset_version": version.get("version"),
        "source_uri": version.get("source_uri"),
        "output_uri": version.get("output_uri"),
        "consumer_uri": version.get("consumer_uri"),
        "consumption_status": version.get("consumption_status"),
        "consumption_reason": version.get("consumption_reason"),
        "data_processing_experiment_id": str(experiment_id) if experiment_id else None,
    }


def resolve_data_handoff(context: Dict[str, Any], config_data_folder: Any) -> Dict[str, Any]:
    """Resolve the exact dataset downstream agents must consume."""

    previous = context.get("previous_results") or {}
    data_result = previous.get("data_processing_agent")
    explicit_handoff = context.get("data_handoff")
    if data_result is not None and data_result.get("status") != "success":
        raise ValueError("data processing agent did not complete successfully")

    summary = (data_result or {}).get("summary") or {}
    handoff = explicit_handoff or summary.get("data_handoff") or {}
    version = summary.get("dataset_version") or {}
    if data_result is not None and not handoff and not version:
        raise ValueError("data processing result does not provide a dataset handoff")

    if handoff or version:
        handoff = handoff or build_data_handoff(
            version,
            ((data_result or {}).get("experiment_ids") or {}).get("data_processing"),
        )
        status = str(handoff.get("consumption_status") or "source_unchanged")
        if status not in {"ready", "source_unchanged"}:
            raise ValueError(
                "data processing output is not ready for downstream consumption: "
                f"{handoff.get('consumption_reason') or status}"
            )
        consumer_uri = handoff.get("consumer_uri")
        if not consumer_uri:
            raise ValueError("data processing handoff does not provide consumer_uri")
        resolved = str(consumer_uri) if is_remote_path(consumer_uri) else str(resolve_data_path(consumer_uri))
        if not is_remote_path(resolved) and not Path(resolved).exists():
            raise ValueError(f"data processing consumer_uri does not exist: {resolved}")
        return {
            "source": "data_processing_agent" if data_result is not None else "context",
            "dataset_id": handoff.get("dataset_id"),
            "consumer_uri": resolved,
            "consumption_status": status,
            "dataset_version": handoff.get("dataset_version"),
            "data_processing_experiment_id": handoff.get("data_processing_experiment_id"),
        }

    return {
        "source": "config",
        "dataset_id": None,
        "consumer_uri": str(resolve_data_path(config_data_folder)),
        "consumption_status": "source_unchanged",
        "dataset_version": None,
        "data_processing_experiment_id": None,
    }


__all__ = ["build_data_handoff", "resolve_data_handoff"]
