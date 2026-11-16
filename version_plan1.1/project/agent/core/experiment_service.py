"""Single model-agnostic write interface for experiment records."""

from __future__ import annotations

from typing import Any, Dict, Optional

from agent.utils.experiment_tracker import ExperimentTracker
from .contracts import OperationResult


class ExperimentService:
    def __init__(self, tracker: Optional[ExperimentTracker] = None) -> None:
        self.tracker = tracker or ExperimentTracker()

    def record_result(
        self,
        experiment_id: str,
        result: OperationResult,
        *,
        experiment_type: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        actor: Optional[Dict[str, Any]] = None,
        update_status: bool = True,
    ) -> bool:
        """Record an operation result without letting trial steps pollute HPO top level.

        HPO experiments contain many trial-level training/evaluation operations. When
        update_status is False, those operation details are returned to the caller and
        stored on the Trial object, while the top-level HPO record keeps only study-level
        summaries written by HPOService/HPOAgent.
        """
        result.experiment_id = experiment_id
        if not update_status:
            stable_updates: Dict[str, Any] = {}
            if result.task:
                stable_updates["task"] = result.task
            if result.model:
                stable_updates["model"] = result.model
            return self.tracker.update_experiment(
                experiment_id,
                experiment_type=experiment_type,
                **stable_updates,
            )

        return self.tracker.update_experiment(
            experiment_id,
            experiment_type=experiment_type,
            status=result.status,
            error=result.error,
            duration=duration_seconds,
            stage=result.stage,
            actor=actor,
            task=result.task,
            model=result.model,
            execution=result.execution,
            metrics=result.metrics,
            artifacts=[artifact.to_dict() for artifact in result.artifacts],
            parameters=result.parameters,
            extensions=result.extensions,
        )