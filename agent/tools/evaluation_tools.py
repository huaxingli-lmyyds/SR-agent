"""Model-agnostic evaluation tool backed by registered adapters."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from agent.core.adapters import resolve_adapter_bundle
from agent.core.contracts import OperationResult
from agent.core.metrics import require_finite_metric
from agent.core.experiment_service import ExperimentService
from agent.hpo import HPOService
from agent.hpo.protocol import (
    METRIC_PROTOCOL_ID,
    METRIC_UNITS,
    file_sha256,
    resolve_hpo_validation_protocol,
)
from agent.utils import (
    ExperimentTracker,
    resolve_evaluation_metrics,
    get_experiment_artifact_dir,
    resolve_config_path,
    resolve_data_path,
    resolve_optional_project_path,
)


def _checkpoint_path(record: dict, trial_id: Optional[str] = None) -> Optional[str]:
    def matches(artifact: dict) -> bool:
        metadata = artifact.get("metadata") or {}
        path = str(artifact.get("path") or "")
        return (
            trial_id is None
            or metadata.get("trial_id") == trial_id
            or f"/trials/{trial_id}/" in path.replace("\\", "/")
        )

    for artifact in record.get("artifacts") or []:
        if artifact.get("type") == "checkpoint" and matches(artifact):
            return artifact.get("path")

    optimization = (record.get("extensions") or {}).get("optimization") or {}
    for trial in optimization.get("trial_summary") or []:
        if trial_id is not None and trial.get("trial_id") != trial_id:
            continue
        for artifact in trial.get("artifacts") or []:
            if artifact.get("type") == "checkpoint" and matches(artifact):
                return artifact.get("path")
    return None


def _run_evaluation(
    model_path: Optional[str] = None,
    verification_config: Optional[str] = None,
    verification_pairs: Optional[str] = None,
    evaluation_split: Optional[str] = None,
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
    if experiments_dir:
        output_folder = Path(experiments_dir).resolve() / experiment_id / "evaluation"
        output_folder.mkdir(parents=True, exist_ok=True)
    else:
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
    split = str(evaluation_split or ("validation" if trial_id else "test")).lower()
    if split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be 'validation' or 'test'")
    if trial_id and split != "validation":
        raise ValueError("HPO Trials may only be evaluated on the validation split")

    study = None
    if trial_id:
        study = HPOService(tracker).load_study(experiment_id)
        objective = study.objectives[0]
        primary_metric, metric_mode = objective.metric, objective.mode
    else:
        primary_metric = task.get("primary_metric") or task_adapter.primary_metric
        metric_mode = task.get("metric_mode") or task_adapter.metric_mode

    pairs_path = None
    if record.get("experiment_type") == "hpo" and trial_id:
        protocol_runtime = resolve_hpo_validation_protocol(
            {
                "verification_config": verification_config,
                "validation_pairs": verification_pairs,
            },
            persisted_execution=execution,
            require_explicit=False,
        )
        config_candidate = protocol_runtime["verification_config"]
        pairs_path = protocol_runtime["validation_pairs"]
    elif record.get("experiment_type") == "hpo" and split == "test":
        # Never let final held-out evaluation reuse the Study's validation
        # config by fallback. Both test inputs must be deliberately supplied.
        if not verification_config or not verification_pairs:
            raise ValueError(
                "final HPO test evaluation requires explicit verification_config "
                "and verification_pairs after model lock"
            )
        config_candidate = verification_config
        resolved_pairs = resolve_optional_project_path(verification_pairs)
        if resolved_pairs is None or not resolved_pairs.is_file():
            raise ValueError(f"verification_pairs file not found: {verification_pairs}")
        pairs_path = str(resolved_pairs)
        optimization = (record.get("extensions") or {}).get("optimization") or {}
        recorded_study_id = optimization.get("study_id") or (
            optimization.get("study") or {}
        ).get("study_id")
        if recorded_study_id:
            locked_study = HPOService(tracker).load_study(experiment_id)
            if (
                locked_study.status != "completed"
                or not locked_study.best_trial_id
            ):
                raise ValueError(
                    "held-out test evaluation is allowed only after the HPO Study "
                    "has completed and locked a best Trial"
                )
            locked_checkpoint = _checkpoint_path(
                record, str(locked_study.best_trial_id)
            )
            if locked_checkpoint is not None and (
                resolve_optional_project_path(locked_checkpoint) != resolved_model
            ):
                raise ValueError(
                    "held-out test model_path is not the locked best Trial checkpoint"
                )
    else:
        config_candidate = (
            verification_config
            or execution.get("evaluation_config_path")
            or getattr(model_adapter, "default_evaluation_config", None)
            or getattr(runner_adapter, "default_evaluation_config", None)
            or record.get("config_path")
        )
        if verification_pairs:
            resolved_pairs = resolve_optional_project_path(verification_pairs)
            if resolved_pairs is None or not resolved_pairs.is_file():
                raise ValueError(f"verification_pairs file not found: {verification_pairs}")
            pairs_path = str(resolved_pairs)
    config_path = str(resolve_config_path(config_candidate))
    if not Path(config_path).is_file():
        raise ValueError(f"verification_config file not found: {config_path}")
    config_sha256 = file_sha256(Path(config_path))
    pairs_sha256 = file_sha256(Path(pairs_path)) if pairs_path else None

    started_at = datetime.now()
    run_opts = {}
    if device:
        run_opts["device"] = device
    if precision:
        run_opts["precision"] = precision
    if eval_precision:
        run_opts["eval_precision"] = eval_precision
    overrides = {"output_folder": str(output_folder)}
    if pairs_path:
        overrides["verification_file"] = pairs_path
    if run_opts:
        overrides["_run_opts"] = run_opts
    raw = runner_adapter.run_evaluation(
        config_path,
        model_path=model_path,
        data_path=data_folder,
        overrides=overrides,
    )

    metrics = resolve_evaluation_metrics(raw)
    scores_path = raw.get("scores_path")

    # A failed runner may have no metrics; retain its real error for retry policy.
    if raw.get("status") == "success":
        task_adapter.validate_metrics(metrics)
        if trial_id:
            require_finite_metric(metrics, primary_metric)
    result = runner_adapter.normalize_evaluation_result({
        "status": raw.get("status", "failed"),
        "error": raw.get("error"),
        "metrics": metrics,
        "evaluation_log_path": str(output_folder / "log.txt"),
        "scores_path": scores_path,
        "output_folder": raw.get("output_folder") or str(output_folder),
    })
    normalized_metrics = {}
    for values in result.metrics.values():
        normalized_metrics.update(values or {})
    result.metrics = {split: normalized_metrics}
    result.task = {
        **task,
        "type": task_type,
        "dataset": data_folder,
        "primary_metric": primary_metric,
        "metric_mode": metric_mode,
        "metric_protocol": task.get("metric_protocol") or METRIC_PROTOCOL_ID,
        "metric_units": task.get("metric_units") or dict(METRIC_UNITS),
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
        "evaluation_split": split,
        "verification_config_path": config_path,
        "verification_pairs_path": pairs_path,
        "verification_config_sha256": config_sha256,
        "verification_pairs_sha256": pairs_sha256,
        "metric_protocol": task.get("metric_protocol") or METRIC_PROTOCOL_ID,
        "runtime_options": run_opts,
    })
    for artifact in result.artifacts:
        if trial_id:
            artifact.metadata["trial_id"] = trial_id
    result.parameters = {
        "model_path": model_path,
        "data_path": data_folder,
        "evaluation_split": split,
        "verification_config": config_path,
        "verification_pairs": pairs_path,
    }
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
            cost={
                "evaluated_checkpoint": model_path,
                "evaluation": {
                    "duration_seconds": (datetime.now() - started_at).total_seconds(),
                    "status": result.status,
                    "metrics": trial_metrics,
                    "split": split,
                    "verification_config_sha256": config_sha256,
                    "verification_pairs_sha256": pairs_sha256,
                    "metric_protocol": task.get("metric_protocol")
                    or METRIC_PROTOCOL_ID,
                },
            },
            artifacts=[artifact.to_dict() for artifact in result.artifacts],
            stop_reason=result.error,
        )
        tracker.update_hpo_experiment(
            experiment_id,
            extensions={"optimization": {"latest_trial": {
                "trial_id": trial_id,
                "phase": "completed" if result.status == "success" else "evaluation_failed",
                "status": result.status,
                "updated_at": datetime.now().isoformat(),
            }}},
        )
    return result.to_json()


@tool
def RunEvaluation(
    model_path: Optional[str] = None,
    verification_config: Optional[str] = None,
    verification_pairs: Optional[str] = None,
    evaluation_split: Optional[str] = None,
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
            verification_pairs=verification_pairs,
            evaluation_split=evaluation_split,
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
