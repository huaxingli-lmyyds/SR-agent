"""Model-agnostic evaluation tool backed by registered adapters."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from agent.core.adapters import resolve_adapter_bundle
from agent.core.contracts import OperationResult
from agent.core.experiment_service import ExperimentService
from agent.hpo import HPOService
from agent.utils import (
    ExperimentTracker,
    MetricsCalculator,
    extract_scores_data,
    get_experiment_artifact_dir,
    resolve_config_path,
    resolve_data_path,
    resolve_optional_project_path,
)


def _checkpoint_path(record: dict, trial_id: Optional[str] = None) -> Optional[str]:
    for artifact in record.get("artifacts") or []:
        metadata = artifact.get("metadata") or {}
        path = str(artifact.get("path") or "")
        matches_trial = (
            trial_id is None
            or metadata.get("trial_id") == trial_id
            or f"/trials/{trial_id}/" in path.replace("\\", "/")
        )
        if artifact.get("type") == "checkpoint" and matches_trial:
            return artifact.get("path")
    return None


def _run_evaluation(
    model_path: Optional[str] = None,
    verification_config: Optional[str] = None,
    data_folder: Optional[str] = None,
    experiment_id: Optional[str] = None,
    trial_id: Optional[str] = None,
    experiments_dir: Optional[str] = None,
    runner: Optional[str] = None,
    task_type: Optional[str] = None,
    model_family: Optional[str] = None,
    implementation: Optional[str] = None,
    device: Optional[str] = None,
    precision: Optional[str] = None,
    eval_precision: Optional[str] = None,
) -> str:
    """Evaluate a recorded model and return a structured operation result."""
    tracker = ExperimentTracker(experiments_dir)
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

    model_path = model_path or _checkpoint_path(record, trial_id)
    resolved_model = resolve_optional_project_path(model_path)
    model_path = str(resolved_model) if resolved_model is not None else None
    if model_path is None:
        return OperationResult(
            status="failed",
            stage="evaluation",
            error=(
                f"no checkpoint artifact found for trial_id={trial_id}"
                if trial_id else "no checkpoint artifact or model_path was provided"
            ),
            experiment_id=experiment_id,
        ).to_json()
    data_folder = str(resolve_data_path(data_folder or (record.get("task") or {}).get("dataset")))
    output_folder = get_experiment_artifact_dir(
        experiment_id,
        "evaluation",
        record.get("experiment_type") or "hpo",
        create=True,
    )
    if trial_id:
        output_folder = output_folder / trial_id
        output_folder.mkdir(parents=True, exist_ok=True)
    task = record.get("task") or {}
    execution = record.get("execution") or {}
    model = record.get("model") or {}
    task_type = task_type or task.get("type") or "speaker_verification"
    model_family = model_family or model.get("family") or "ecapa_tdnn"
    implementation = implementation or model.get("implementation") or "speechbrain"
    runner_name = runner or execution.get("runner") or implementation
    adapters = resolve_adapter_bundle(task_type, model_family, implementation, runner_name)
    task_adapter, model_adapter, runner_adapter = adapters.task, adapters.model, adapters.runner
    config_candidate = (
        verification_config
        or execution.get("evaluation_config_path")
        or getattr(model_adapter, "default_evaluation_config", None)
        or getattr(runner_adapter, "default_evaluation_config", None)
        or record.get("config_path")
    )
    config_path = str(resolve_config_path(config_candidate))

    started_at = datetime.now()
    run_opts = {}
    if device:
        run_opts["device"] = device
    if precision:
        run_opts["precision"] = precision
    if eval_precision:
        run_opts["eval_precision"] = eval_precision
    overrides = {"output_folder": str(output_folder)}
    if run_opts:
        overrides["_run_opts"] = run_opts
    raw = runner_adapter.run_evaluation(
        config_path,
        model_path=model_path,
        data_path=data_folder,
        overrides=overrides,
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
    result.task = {
        **task,
        "type": task_type,
        "dataset": data_folder,
        "primary_metric": task_adapter.primary_metric,
        "metric_mode": task_adapter.metric_mode,
    }
    result.model = {
        **model,
        "family": model_family,
        "implementation": implementation,
    }
    result.execution.update({
        "runner": runner_name,
        "output_folder": raw.get("output_folder") or str(output_folder),
        "trial_id": trial_id,
        "runtime_options": run_opts,
    })
    for artifact in result.artifacts:
        if trial_id:
            artifact.metadata["trial_id"] = trial_id
    result.parameters = {"model_path": model_path, "data_path": data_folder}
    ExperimentService(tracker).record_result(
        experiment_id,
        result,
        duration_seconds=(datetime.now() - started_at).total_seconds(),
        actor={"type": "hpo_agent", "name": "model_evaluator"},
        update_status=trial_id is None,
    )
    if trial_id:
        service = HPOService(tracker)
        study = service.load_study(experiment_id)
        trial_metrics = {}
        for split_metrics in result.metrics.values():
            trial_metrics.update(split_metrics or {})
        service.record_trial(
            study,
            trial_id,
            status="completed" if result.status == "success" else "failed",
            metrics=trial_metrics,
            artifacts=[artifact.to_dict() for artifact in result.artifacts],
            stop_reason=result.error,
        )
    return result.to_json()


@tool
def RunEvaluation(
    model_path: Optional[str] = None,
    verification_config: Optional[str] = None,
    data_folder: Optional[str] = None,
    experiment_id: Optional[str] = None,
    trial_id: Optional[str] = None,
    experiments_dir: Optional[str] = None,
    runner: Optional[str] = None,
    task_type: Optional[str] = None,
    model_family: Optional[str] = None,
    implementation: Optional[str] = None,
    device: Optional[str] = None,
    precision: Optional[str] = None,
    eval_precision: Optional[str] = None,
) -> str:
    """Evaluate through a registered runner and always return OperationResult JSON."""
    try:
        return _run_evaluation(
            model_path=model_path,
            verification_config=verification_config,
            data_folder=data_folder,
            experiment_id=experiment_id,
            trial_id=trial_id,
            experiments_dir=experiments_dir,
            runner=runner,
            task_type=task_type,
            model_family=model_family,
            implementation=implementation,
            device=device,
            precision=precision,
            eval_precision=eval_precision,
        )
    except Exception as exc:
        return OperationResult(
            status="failed",
            stage="evaluation",
            error=str(exc),
            experiment_id=experiment_id,
        ).to_json()


__all__ = ["RunEvaluation"]
