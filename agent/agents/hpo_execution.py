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
                if item.get("type") == "checkpoint" and item.get("path")
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
