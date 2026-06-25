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
        result.experiment_id = experiment_id
        return self.tracker.update_experiment(
            experiment_id,
            experiment_type=experiment_type,
            status=result.status if update_status else None,
            error=result.error if update_status else None,
            duration=duration_seconds if update_status else None,
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
