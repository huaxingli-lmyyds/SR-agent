"""Training/evaluation tool boundary for a single HPO Trial."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from agent.hpo.resources import CampaignResourceBudget, stage_gpu_count


@dataclass
class TrialExecutor:
    experiment_id: str
    data_folder: str
    experiments_dir: Path
    runtime_options: dict
    task_type: str
    model_family: str
    implementation: str
    runner: str
    resource_budget: CampaignResourceBudget | None = None

    def _invoke_stage(self, trial, attempt, tool, payload, stage, gpu_count):
        resources = self.resource_budget
        experiment_id = self.experiment_id
        if resources:
            deadline = resources.start_stage(
                experiment_id=experiment_id,
                trial_id=trial.trial_id,
                attempt=attempt,
                stage=stage,
                gpu_count=gpu_count,
                max_seconds=trial.budget.max_duration_seconds
                if stage == "training"
                else None,
            )
            if deadline is not None:
                payload["runtime_deadline_epoch"] = deadline
        try:
            return json.loads(tool.invoke(payload))
        except Exception as exc:
            return {
                "status": "failed",
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if resources:
                resources.finish_stage()

    def __call__(self, trial: Any, attempt: int) -> dict[str, Any]:
        runtime_options = self.runtime_options
        experiment_id = self.experiment_id
        data_folder = self.data_folder
        task_type = self.task_type
        default_model_family = self.model_family
        default_implementation = self.implementation
        default_runner = self.runner
        from agent.tools.evaluation_tools import RunEvaluation
        from agent.tools.training_tools import TrainModel

        parameters = dict(trial.parameters)
        model_family = str(parameters.pop("model_family", default_model_family))
        implementation = str(
            parameters.pop("implementation", default_implementation)
        )
        runner = str(parameters.pop("runner", default_runner))
        training_gpu_count = stage_gpu_count(runtime_options, runner)
        evaluation_gpu_count = stage_gpu_count(
            runtime_options, runner, evaluation=True
        )

        training_started = perf_counter()
        if trial.cost.get("evaluation_only"):
            checkpoint = trial.cost.get("evaluated_checkpoint")
            if not checkpoint or not Path(checkpoint).exists():
                return {
                    "status": "failed",
                    "stage": "evaluation",
                    "error": "FileNotFoundError: missing checkpoint for evaluation recovery",
                }
            train_result = {
                "status": "success",
                "artifacts": [
                    {"type": "checkpoint", "path": checkpoint},
                ],
            }
        else:
            train_result = self._invoke_stage(
                trial,
                attempt,
                TrainModel,
                {
                    "experiment_id": experiment_id,
                    "trial_id": trial.trial_id,
                    "data_folder": data_folder,
                    "checkpoint_selection": runtime_options.get("checkpoint_selection", "legacy"),
                    "parameters_json": json.dumps(
                        parameters, ensure_ascii=False
                    ),
                    "budget_json": json.dumps(
                        trial.budget.to_dict(), ensure_ascii=False
                    ),
                    "training_exclusion_pairs": runtime_options.get(
                        "training_exclusion_pairs"
                    ),
                    "task_type": task_type,
                    "model_family": model_family,
                    "implementation": implementation,
                    "runner": runner,
                    "experiments_dir": str(self.experiments_dir),
                    "device": runtime_options.get("device"),
                    "ddp_devices": runtime_options.get("ddp_devices"),
                    "distributed_backend": runtime_options.get(
                        "distributed_backend"
                    ),
                    "batch_size_semantics": runtime_options.get(
                        "batch_size_semantics"
                    ),
                    "precision": runtime_options.get("precision"),
                    "eval_precision": runtime_options.get("eval_precision"),
                },
                "training",
                training_gpu_count,
            )
        training_seconds = (
            0.0
            if trial.cost.get("evaluation_only")
            else perf_counter() - training_started
        )
        training_runtime = dict(
            (
                (train_result.get("extensions") or {}).get("speechbrain") or {}
            ).get("runtime")
            or {}
        )
        ddp_runtime = dict(training_runtime.get("ddp") or {})
        training_world_size = max(
            int(ddp_runtime.get("world_size") or training_gpu_count or 1), 1
        )
        if train_result.get("status") != "success":
            return {
                **train_result,
                "cost": {
                    "attempt_training_seconds": training_seconds,
                    "attempt_training_gpu_seconds": (
                        training_seconds * training_gpu_count
                    ),
                    "training_world_size": training_world_size,
                    "attempt_evaluation_seconds": 0.0,
                    "attempt_evaluation_gpu_seconds": 0.0,
                },
            }
        # Bind evaluation to this attempt, not the first historical artifact.
        current_checkpoint = next(
            (
                item.get("path")
                for item in train_result.get("artifacts") or []
                if item.get("type") in {"checkpoint", "checkpoint_candidate"} and item.get("path")
            ),
            None,
        )
        if current_checkpoint is None:
            return {
                "status": "failed",
                "stage": "training",
                "error": "missing checkpoint artifact from current training attempt",
                "cost": {
                    "attempt_training_seconds": training_seconds,
                    "attempt_training_gpu_seconds": (
                        training_seconds * training_gpu_count
                    ),
                    "training_world_size": training_world_size,
                    "attempt_evaluation_seconds": 0.0,
                    "attempt_evaluation_gpu_seconds": 0.0,
                },
            }
        if runtime_options.get("checkpoint_selection") == "validation_topk":
            return self._evaluate_candidates(trial, attempt, train_result, training_seconds,
                training_gpu_count, evaluation_gpu_count, training_world_size, training_runtime)
        evaluation_started = perf_counter()
        evaluation = self._invoke_stage(
            trial,
            attempt,
            RunEvaluation,
            {
                "model_path": current_checkpoint,
                "verification_config": runtime_options.get(
                    "verification_config"
                ),
                "verification_pairs": runtime_options.get("validation_pairs"),
                "evaluation_split": "validation",
                "experiment_id": experiment_id,
                "trial_id": trial.trial_id,
                "data_folder": data_folder,
                "experiments_dir": str(self.experiments_dir),
                "runner": runner,
                "task_type": task_type,
                "model_family": model_family,
                "implementation": implementation,
                "device": runtime_options.get("device"),
                "precision": runtime_options.get("precision"),
                "eval_precision": runtime_options.get("eval_precision"),
            },
            "evaluation",
            evaluation_gpu_count,
        )
        metrics: dict[str, Any] = {}
        for values in (evaluation.get("metrics") or {}).values():
            metrics.update(values or {})
        evaluation_seconds = perf_counter() - evaluation_started
        return {
            "status": evaluation.get("status", "failed"),
            "stage": "evaluation",
            "error": evaluation.get("error"),
            "metrics": metrics,
            "artifacts": evaluation.get("artifacts") or [],
            "cost": {
                "attempt": attempt,
                "evaluated_checkpoint": current_checkpoint,
                "training_observation": {
                    "selected_checkpoint": ((train_result.get("extensions") or {}).get("speechbrain") or {}).get("selected_checkpoint") or {},
                    "epoch_data": (((train_result.get("extensions") or {}).get("speechbrain") or {}).get("epoch_data") or [])[-12:],
                    "final_metrics": ((train_result.get("extensions") or {}).get("speechbrain") or {}).get("final_metrics") or {},
                    "final_learning_rate_schedule": training_runtime.get("final_learning_rate_schedule") or {},
                } if not trial.cost.get("evaluation_only") else trial.cost.get("training_observation", {}),
                "attempt_training_seconds": training_seconds,
                "attempt_training_gpu_seconds": (
                    training_seconds * training_gpu_count
                ),
                "training_world_size": training_world_size,
                "attempt_evaluation_seconds": evaluation_seconds,
                "attempt_evaluation_gpu_seconds": evaluation_seconds
                * evaluation_gpu_count,
            },
        }


    def _evaluate_candidates(self, trial, attempt, train_result, training_seconds,
                             training_gpu_count, evaluation_gpu_count, world_size, training_runtime):
        from hashlib import sha256
        from agent.hpo.service import HPOService
        from agent.utils import ExperimentTracker
        from agent.tools.evaluation_tools import RunEvaluation
        from agent.runners.checkpoint_selection import choose
        from agent.core.metrics import is_finite_metric
        service = HPOService(ExperimentTracker(self.experiments_dir))
        study = service.load_study(self.experiment_id)
        objective = study.objectives[0]
        observation = (train_result.get("extensions") or {}).get("speechbrain") or {}
        persisted = service.load_trial(self.experiment_id, trial.trial_id)
        rows = persisted.cost.get("checkpoint_evaluations") if trial.cost.get("evaluation_only") else None
        rows = [dict(r) for r in (rows or observation.get("checkpoint_candidates") or [])]
        cost = {"checkpoint_evaluations": rows, "checkpoint_selection": "validation_topk",
                "attempt_training_seconds": training_seconds,
                "attempt_training_gpu_seconds": training_seconds * training_gpu_count,
                "training_world_size": world_size,
                "attempt_evaluation_seconds": 0.0, "attempt_evaluation_gpu_seconds": 0.0,
                "training_observation": persisted.cost.get("training_observation", {}) if trial.cost.get("evaluation_only") else {
                    "epoch_data": observation.get("epoch_data", []),
                    "final_metrics": observation.get("final_metrics", {}),
                    "final_learning_rate_schedule": training_runtime.get("final_learning_rate_schedule", {})}}
        if not rows:
            return {"status": "failed", "stage": "evaluation", "error": "missing checkpoint candidates", "cost": cost}
        cost["evaluated_checkpoint"] = rows[0]["path"]
        # Persist the candidate manifest before evaluation so recovery never retrains.
        cost["evaluation"] = {"status": "failed", "selection_complete": False}
        service.record_trial(study, trial.trial_id, status="running", cost=cost)
        for row in rows:
            if row.get("status") == "success" and is_finite_metric(row.get("metrics", {}).get(objective.metric)):
                continue
            key = sha256(row["path"].encode()).hexdigest()[:20]
            started = perf_counter()
            result = self._invoke_stage(trial, attempt, RunEvaluation, {
                "model_path": row["path"], "checkpoint_id": key, "defer_commit": True,
                "verification_config": self.runtime_options.get("verification_config"),
                "verification_pairs": self.runtime_options.get("validation_pairs"),
                "evaluation_split": "validation", "experiment_id": self.experiment_id,
                "trial_id": trial.trial_id, "data_folder": self.data_folder,
                "experiments_dir": str(self.experiments_dir), "runner": self.runner,
                "task_type": self.task_type, "model_family": self.model_family,
                "implementation": self.implementation, "device": self.runtime_options.get("device"),
                "precision": self.runtime_options.get("precision"),
                "eval_precision": self.runtime_options.get("eval_precision"),
            }, "evaluation", evaluation_gpu_count)
            duration = perf_counter() - started
            cost["attempt_evaluation_seconds"] += duration
            cost["attempt_evaluation_gpu_seconds"] += duration * evaluation_gpu_count
            metrics = {k:v for group in (result.get("metrics") or {}).values() for k,v in (group or {}).items()}
            status = result.get("status", "failed")
            error = result.get("error")
            if status == "success" and not is_finite_metric(metrics.get(objective.metric)):
                status, error = "failed", "invalid primary metric in checkpoint evaluation"
            row.update(status=status, metrics=metrics, error=error, artifacts=result.get("artifacts") or [],
                       seconds=duration, attempts=int(row.get("attempts", 0))+1)
            service.record_trial(study, trial.trial_id, status="running", cost=cost)
            if status != "success":
                return {"status": "failed", "stage": "evaluation", "error": error, "cost": cost}
        best = choose(rows, objective.metric, objective.mode)
        cost["evaluated_checkpoint"] = best["path"]
        cost["evaluation"] = {"status": "success", "selection_complete": True}
        cost["training_observation"]["selected_checkpoint"] = best.get("observation", {})
        cost["selection_reason"] = f"best_validation_{objective.metric};earliest_epoch_tiebreak"
        return {"status": "success", "stage": "evaluation", "metrics": best["metrics"],
                "artifacts": [{"type": "checkpoint", "name": Path(best["path"]).name,
                    "path": best["path"], "metadata": {"trial_id": trial.trial_id,
                    "checkpoint_selection": "validation_topk", "epoch": best["epoch"]}},
                    *best.get("artifacts", [])], "cost": cost}
