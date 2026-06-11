"""Tools for model-agnostic HPO study and trial lifecycle management."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

from agent.hpo import HPOService, Objective, TrialBudget, search_space_from_dict


def _object(value: str) -> dict:
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("JSON payload must be an object")
    return data


def _list(value: str) -> list:
    data = json.loads(value)
    if not isinstance(data, list):
        raise ValueError("JSON payload must be a list")
    return data


@tool
def CreateHPOStudy(
    experiment_id: str,
    search_space_json: str,
    objectives_json: str,
    budgets_json: str,
    strategy: str = "successive_halving",
    reduction_factor: int = 3,
    random_seed: int = 0,
    max_trials: Optional[int] = None,
) -> str:
    """Create a structured HPO study before running any training trial."""
    try:
        study = HPOService().create_study(
            experiment_id,
            search_space_from_dict(_object(search_space_json)),
            [Objective(**item) for item in _list(objectives_json)],
            [TrialBudget(**item) for item in _list(budgets_json)],
            strategy=strategy,
            reduction_factor=reduction_factor,
            random_seed=random_seed,
            max_trials=max_trials,
        )
        return json.dumps(study.to_dict(), ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)


@tool
def SuggestHPOTrials(experiment_id: str, count: int = 3) -> str:
    """Suggest unique random-search candidates using the study search space."""
    try:
        service = HPOService()
        study = service.load_study(experiment_id)
        trials = service.suggest_trials(study, count)
        return json.dumps([trial.to_dict() for trial in trials], ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)


@tool
def RecordHPOTrialResult(
    experiment_id: str,
    trial_id: str,
    status: str,
    metrics_json: str = "{}",
    intermediate_metrics_json: str = "[]",
    cost_json: str = "{}",
    artifacts_json: str = "[]",
    stop_reason: Optional[str] = None,
) -> str:
    """Record one independent trial result and refresh the study best trial."""
    try:
        service = HPOService()
        study = service.load_study(experiment_id)
        trial = service.record_trial(
            study,
            trial_id,
            status=status,
            metrics=_object(metrics_json),
            intermediate_metrics=_list(intermediate_metrics_json),
            cost=_object(cost_json),
            artifacts=_list(artifacts_json),
            stop_reason=stop_reason,
        )
        return json.dumps(trial.to_dict(), ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)


@tool
def CheckTrialEarlyStopping(
    experiment_id: str,
    trial_id: str,
    patience: int = 3,
    min_improvement: float = 0.0,
) -> str:
    """Check whether a trial should stop based on intermediate objective metrics."""
    try:
        service = HPOService()
        study = service.load_study(experiment_id)
        decision = service.early_stop(
            study,
            trial_id,
            patience=patience,
            min_improvement=min_improvement,
        )
        return json.dumps({
            "should_stop": decision.should_stop,
            "reason": decision.reason,
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)


@tool
def PromoteHPOTrials(experiment_id: str) -> str:
    """Promote the best completed trials to the next budget rung."""
    try:
        service = HPOService()
        study = service.load_study(experiment_id)
        trials = service.promote_trials(study)
        return json.dumps([trial.to_dict() for trial in trials], ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)


@tool
def GetHPOStudy(experiment_id: str) -> str:
    """Return the study definition and all independent trial records."""
    try:
        service = HPOService()
        study = service.load_study(experiment_id)
        return json.dumps({
            "study": study.to_dict(),
            "trials": [trial.to_dict() for trial in service.list_trials(experiment_id)],
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)


__all__ = [
    "CreateHPOStudy",
    "SuggestHPOTrials",
    "RecordHPOTrialResult",
    "CheckTrialEarlyStopping",
    "PromoteHPOTrials",
    "GetHPOStudy",
]
