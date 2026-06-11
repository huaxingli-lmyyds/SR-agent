"""Generic experiment history tools."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from agent.utils import ExperimentTracker
from agent.utils.path_tool import (
    get_data_processing_experiments_dir,
    get_hpo_experiments_dir,
    get_manage_experiments_dir,
)


def _metric(record: Dict[str, Any], name: str) -> Optional[Any]:
    for split in ("test", "validation", "train", "summary"):
        value = (record.get("metrics") or {}).get(split, {}).get(name)
        if value is not None:
            return value
    return None


def _summary(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment_id": record.get("experiment_id"),
        "schema_version": record.get("schema_version"),
        "stage": record.get("stage"),
        "status": record.get("status"),
        "task": record.get("task"),
        "model": record.get("model"),
        "actor": record.get("actor"),
        "execution": record.get("execution"),
        "metrics": record.get("metrics"),
        "artifacts": record.get("artifacts"),
        "parameters": record.get("parameters"),
        "extensions": record.get("extensions"),
        "linked_experiments": record.get("linked_experiments"),
        "agent_message_count": len(record.get("agent_messages") or []),
    }


def _record(tracker: ExperimentTracker, experiment_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if experiment_id:
        return tracker.get_experiment(experiment_id)
    recent = tracker.list_experiments(limit=1)
    return recent[0] if recent else None


def _compare(tracker: ExperimentTracker, ids: Optional[List[str]], metric: str) -> str:
    records = (
        [tracker.get_experiment(item) for item in ids]
        if ids else tracker.list_experiments(limit=5)
    )
    values = {
        record["experiment_id"]: _metric(record, metric)
        for record in records if record and _metric(record, metric) is not None
    }
    return json.dumps({"metric": metric, "values": values}, ensure_ascii=False, default=str)


@tool
def CompareHPOExperiments(experiment_ids: Optional[List[str]] = None, metric: str = "eer") -> str:
    return _compare(ExperimentTracker(get_hpo_experiments_dir()), experiment_ids, metric)


@tool
def GetHPOExperimentResults(experiment_id: Optional[str] = None) -> str:
    record = _record(ExperimentTracker(get_hpo_experiments_dir()), experiment_id)
    return json.dumps(_summary(record) if record else {"error": "experiment not found"}, ensure_ascii=False, default=str)


@tool
def ListHPOExperiments(n: int = 10) -> str:
    records = ExperimentTracker(get_hpo_experiments_dir()).list_experiments(limit=n)
    return json.dumps([_summary(record) for record in records], ensure_ascii=False, default=str)


@tool
def CompareDataProcessingExperiments(experiment_ids: Optional[List[str]] = None, metric: str = "train") -> str:
    return _compare(ExperimentTracker(get_data_processing_experiments_dir()), experiment_ids, metric)


@tool
def GetDataProcessingExperimentResults(experiment_id: Optional[str] = None) -> str:
    record = _record(ExperimentTracker(get_data_processing_experiments_dir()), experiment_id)
    return json.dumps(_summary(record) if record else {"error": "experiment not found"}, ensure_ascii=False, default=str)


@tool
def ListDataProcessingExperiments(n: int = 10) -> str:
    records = ExperimentTracker(get_data_processing_experiments_dir()).list_experiments(limit=n)
    return json.dumps([_summary(record) for record in records], ensure_ascii=False, default=str)


@tool
def CompareOrchestrationExperiments(experiment_ids: Optional[List[str]] = None, metric: str = "rounds") -> str:
    tracker = ExperimentTracker(get_manage_experiments_dir())
    if metric == "messages":
        records = [tracker.get_experiment(item) for item in experiment_ids] if experiment_ids else tracker.list_experiments(limit=5)
        return json.dumps({
            record["experiment_id"]: len(record.get("agent_messages") or [])
            for record in records if record
        }, ensure_ascii=False)
    return _compare(tracker, experiment_ids, metric)


@tool
def GetOrchestrationExperimentResults(experiment_id: Optional[str] = None) -> str:
    record = _record(ExperimentTracker(get_manage_experiments_dir()), experiment_id)
    return json.dumps(_summary(record) if record else {"error": "experiment not found"}, ensure_ascii=False, default=str)


@tool
def ListOrchestrationExperiments(n: int = 10) -> str:
    records = ExperimentTracker(get_manage_experiments_dir()).list_experiments(limit=n)
    return json.dumps([_summary(record) for record in records], ensure_ascii=False, default=str)


__all__ = [
    "CompareHPOExperiments", "GetHPOExperimentResults", "ListHPOExperiments",
    "CompareDataProcessingExperiments", "GetDataProcessingExperimentResults", "ListDataProcessingExperiments",
    "CompareOrchestrationExperiments", "GetOrchestrationExperimentResults", "ListOrchestrationExperiments",
]
