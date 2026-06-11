"""Model-agnostic evaluation tool backed by registered adapters."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from agent.core.adapters import get_runner_adapter, get_task_adapter
from agent.core.contracts import OperationResult
from agent.core.experiment_service import ExperimentService
from agent.utils import (
    ExperimentTracker,
    MetricsCalculator,
    extract_scores_data,
    get_experiment_artifact_dir,
    resolve_config_path,
    resolve_data_path,
    resolve_optional_project_path,
)


def _checkpoint_path(record: dict) -> Optional[str]:
    for artifact in record.get("artifacts") or []:
        if artifact.get("type") == "checkpoint":
            return artifact.get("path")
    return None


def _run_evaluation(
    model_path: Optional[str] = None,
    verification_config: Optional[str] = None,
    data_folder: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> str:
    """Evaluate a recorded model and return a structured operation result."""
    tracker = ExperimentTracker()
    if experiment_id is None:
        recent = tracker.list_experiments(limit=1)
        if not recent:
            return OperationResult(
                status="failed",
                stage="evaluation",
                error="no experiment record",
            ).to_json()
        experiment_id = recent[0]["experiment_id"]

    record = tracker.get_experiment(experiment_id)
    if not record:
        return OperationResult(
            status="failed",
            stage="evaluation",
            error=f"experiment not found: {experiment_id}",
            experiment_id=experiment_id,
        ).to_json()

    model_path = model_path or _checkpoint_path(record)
    resolved_model = resolve_optional_project_path(model_path)
    model_path = str(resolved_model) if resolved_model is not None else None
    if model_path is None:
        return OperationResult(
            status="failed",
            stage="evaluation",
            error="no checkpoint artifact or model_path was provided",
            experiment_id=experiment_id,
        ).to_json()
    data_folder = str(resolve_data_path(data_folder or (record.get("task") or {}).get("dataset")))
    output_folder = get_experiment_artifact_dir(
        experiment_id,
        "evaluation",
        record.get("experiment_type") or "hpo",
        create=True,
    )
    task = record.get("task") or {}
    execution = record.get("execution") or {}
    runner_name = execution.get("runner") or (record.get("model") or {}).get("implementation") or "speechbrain"
    runner_adapter = get_runner_adapter(runner_name)
    task_adapter = get_task_adapter(task.get("type") or "speaker_verification")
    config_candidate = verification_config or execution.get("evaluation_config_path")
    if config_candidate is None and runner_name != "speechbrain":
        config_candidate = record.get("config_path")
    config_path = str(resolve_config_path(config_candidate, default_name="verification_ecapa.yaml"))

    started_at = datetime.now()
    raw = runner_adapter.run_evaluation(
        config_path,
        model_path=model_path,
        data_path=data_folder,
        overrides={"output_folder": str(output_folder)},
    )

    metrics = dict(raw.get("metrics") or {})
    for key in ("eer", "min_dcf"):
        if raw.get(key) is not None:
            metrics[key] = raw[key]
    scores_path = raw.get("scores_path")
    if scores_path and Path(scores_path).exists():
        scores = extract_scores_data(scores_path)
        if not scores.get("error"):
            metrics.update(MetricsCalculator.compute_all_metrics(
                scores.get("genuine_scores", []),
                scores.get("impostor_scores", []),
            ))

    task_adapter.validate_metrics(metrics)
    result = runner_adapter.normalize_evaluation_result({
        "status": raw.get("status", "failed"),
        "error": raw.get("error"),
        "metrics": metrics,
        "evaluation_log_path": str(output_folder / "log.txt"),
        "scores_path": scores_path,
        "output_folder": raw.get("output_folder") or str(output_folder),
    })
    result.task = record.get("task") or {}
    result.model = record.get("model") or {}
    result.execution.update({
        "runner": runner_name,
        "output_folder": raw.get("output_folder") or str(output_folder),
    })
    result.parameters = {"model_path": model_path, "data_path": data_folder}
    ExperimentService(tracker).record_result(
        experiment_id,
        result,
        duration_seconds=(datetime.now() - started_at).total_seconds(),
        actor={"type": "hpo_agent", "name": "model_evaluator"},
    )
    return result.to_json()


@tool
def RunEvaluation(
    model_path: Optional[str] = None,
    verification_config: Optional[str] = None,
    data_folder: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> str:
    """Evaluate through a registered runner and always return OperationResult JSON."""
    try:
        return _run_evaluation(
            model_path=model_path,
            verification_config=verification_config,
            data_folder=data_folder,
            experiment_id=experiment_id,
        )
    except Exception as exc:
        return OperationResult(
            status="failed",
            stage="evaluation",
            error=str(exc),
            experiment_id=experiment_id,
        ).to_json()


__all__ = ["RunEvaluation"]
