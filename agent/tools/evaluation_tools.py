"""Model-agnostic evaluation tool backed by registered adapters."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from agent.core.adapters import RUNNER_ADAPTERS
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
from agent.utils import runner


def _checkpoint_path(record: dict) -> Optional[str]:
    for artifact in record.get("artifacts") or []:
        if artifact.get("type") == "checkpoint":
            return artifact.get("path")
    return None


@tool
def RunEvaluation(
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
    data_folder = str(resolve_data_path(data_folder or (record.get("task") or {}).get("dataset")))
    output_folder = get_experiment_artifact_dir(experiment_id, "evaluation", "hpo", create=True)
    config_path = str(resolve_config_path(verification_config, default_name="verification_ecapa.yaml"))

    started_at = datetime.now()
    raw = runner.run_evaluation(
        config_path=config_path,
        model_path=model_path,
        data_folder=data_folder,
        overrides={"output_folder": str(output_folder)},
    )

    metrics = {"eer": raw.get("eer"), "min_dcf": raw.get("min_dcf")}
    scores_path = raw.get("scores_path")
    if scores_path and Path(scores_path).exists():
        scores = extract_scores_data(scores_path)
        if not scores.get("error"):
            metrics.update(MetricsCalculator.compute_all_metrics(
                scores.get("genuine_scores", []),
                scores.get("impostor_scores", []),
            ))

    result = RUNNER_ADAPTERS["speechbrain"].normalize_evaluation_result({
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
        "runner": "speechbrain",
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


__all__ = ["RunEvaluation"]
