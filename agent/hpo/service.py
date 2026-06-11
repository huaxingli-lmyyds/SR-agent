"""Persistent HPO study and trial lifecycle service."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agent.utils import ExperimentTracker, get_experiment_artifact_dir

from .contracts import HPOStudy, Objective, SearchParameter, SearchSpace, Trial, TrialBudget
from .policies import EarlyStoppingPolicy, StopDecision
from .strategies import RandomSearchStrategy, SuccessiveHalvingStrategy

TRIAL_STATUSES = {"suggested", "running", "completed", "promoted", "stopped", "failed"}


class HPOService:
    def __init__(self, tracker: Optional[ExperimentTracker] = None) -> None:
        self.tracker = tracker or ExperimentTracker()
        self.random_strategy = RandomSearchStrategy()
        self.halving_strategy = SuccessiveHalvingStrategy()

    def create_study(
        self,
        experiment_id: str,
        search_space: SearchSpace,
        objectives: List[Objective],
        budgets: List[TrialBudget],
        *,
        strategy: str = "successive_halving",
        reduction_factor: int = 3,
        random_seed: int = 0,
        max_trials: Optional[int] = None,
    ) -> HPOStudy:
        if self.tracker.get_experiment(experiment_id) is None:
            raise ValueError(f"HPO experiment not found: {experiment_id}")
        if not search_space.parameters:
            raise ValueError("search space must contain at least one parameter")
        if not objectives:
            raise ValueError("study must contain at least one objective")
        if not budgets:
            raise ValueError("study must contain at least one budget rung")
        if strategy not in {"random_search", "successive_halving"}:
            raise ValueError(f"unsupported HPO strategy: {strategy}")
        if reduction_factor < 2:
            raise ValueError("reduction_factor must be at least 2")
        if max_trials is not None and max_trials <= 0:
            raise ValueError("max_trials must be positive")
        for objective in objectives:
            if objective.mode not in {"min", "max"}:
                raise ValueError(f"unsupported objective mode: {objective.mode}")
        for budget in budgets:
            if budget.epochs is not None and budget.epochs <= 0:
                raise ValueError("budget epochs must be positive")
            if budget.data_fraction is not None and not 0 < budget.data_fraction <= 1:
                raise ValueError("budget data_fraction must be in (0, 1]")
        now = datetime.now().isoformat()
        study = HPOStudy(
            study_id=f"study_{uuid4().hex[:10]}",
            experiment_id=experiment_id,
            strategy=strategy,
            search_space=search_space,
            objectives=objectives,
            budgets=budgets,
            reduction_factor=reduction_factor,
            max_trials=max_trials,
            random_seed=random_seed,
            created_at=now,
            updated_at=now,
        )
        self._save_study(study)
        return study

    def suggest_trials(self, study: HPOStudy, count: int) -> List[Trial]:
        existing_trials = self.list_trials(study.experiment_id)
        if study.max_trials is not None:
            count = min(count, max(study.max_trials - len(existing_trials), 0))
        if count <= 0:
            return []
        candidates = self.random_strategy.suggest(
            study.search_space,
            count,
            seed=study.random_seed + len(existing_trials),
            existing=[trial.parameters for trial in existing_trials],
        )
        now = datetime.now().isoformat()
        budget = study.budgets[0]
        trials = [
            Trial(
                trial_id=f"trial_{uuid4().hex[:10]}",
                parameters=candidate,
                budget=budget,
                created_at=now,
                updated_at=now,
            )
            for candidate in candidates
        ]
        for trial in trials:
            self.save_trial(study.experiment_id, trial)
            study.trial_ids.append(trial.trial_id)
        study.status = "running"
        study.updated_at = now
        self._save_study(study)
        return trials

    def record_trial(
        self,
        study: HPOStudy,
        trial_id: str,
        *,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
        intermediate_metrics: Optional[List[Dict[str, Any]]] = None,
        cost: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None,
        stop_reason: Optional[str] = None,
    ) -> Trial:
        if status not in TRIAL_STATUSES:
            raise ValueError(f"unsupported trial status: {status}")
        trial = self.load_trial(study.experiment_id, trial_id)
        trial.status = status
        trial.metrics.update(metrics or {})
        trial.intermediate_metrics = intermediate_metrics or trial.intermediate_metrics
        trial.cost.update(cost or {})
        trial.artifacts.extend(artifacts or [])
        trial.stop_reason = stop_reason
        trial.updated_at = datetime.now().isoformat()
        self.save_trial(study.experiment_id, trial)
        self._refresh_study(study)
        return trial

    def finish_study(self, study: HPOStudy, status: str, stop_reason: Optional[str] = None) -> HPOStudy:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"unsupported study terminal status: {status}")
        study.status = status
        study.stop_reason = stop_reason
        study.updated_at = datetime.now().isoformat()
        self._refresh_study(study)
        return study

    def promote_trials(self, study: HPOStudy) -> List[Trial]:
        objective = study.objectives[0]
        candidates = self.halving_strategy.promote(
            self.list_trials(study.experiment_id),
            objective,
            study.reduction_factor,
        )
        promoted: List[Trial] = []
        remaining = None
        if study.max_trials is not None:
            remaining = max(study.max_trials - len(self.list_trials(study.experiment_id)), 0)
        for candidate in candidates:
            if remaining is not None and len(promoted) >= remaining:
                break
            next_rung = candidate.rung + 1
            if next_rung >= len(study.budgets):
                continue
            now = datetime.now().isoformat()
            trial = Trial(
                trial_id=f"trial_{uuid4().hex[:10]}",
                parameters=dict(candidate.parameters),
                budget=study.budgets[next_rung],
                parent_trial_id=candidate.trial_id,
                rung=next_rung,
                created_at=now,
                updated_at=now,
            )
            candidate.status = "promoted"
            candidate.updated_at = now
            self.save_trial(study.experiment_id, candidate)
            self.save_trial(study.experiment_id, trial)
            study.trial_ids.append(trial.trial_id)
            promoted.append(trial)
        self._save_study(study)
        return promoted

    def early_stop(
        self,
        study: HPOStudy,
        trial_id: str,
        *,
        patience: int = 3,
        min_improvement: float = 0.0,
    ) -> StopDecision:
        trial = self.load_trial(study.experiment_id, trial_id)
        objective = study.objectives[0]
        best_value = self.best_metric_value(study, exclude_trial_id=trial_id)
        return EarlyStoppingPolicy(patience, min_improvement).evaluate(
            trial.intermediate_metrics,
            metric=objective.metric,
            mode=objective.mode,
            best_known=best_value,
        )

    def load_study(self, experiment_id: str) -> HPOStudy:
        data = json.loads(self._study_path(experiment_id).read_text(encoding="utf-8"))
        return study_from_dict(data)

    def list_trials(self, experiment_id: str) -> List[Trial]:
        trial_dir = self._trial_dir(experiment_id)
        if not trial_dir.exists():
            return []
        return [
            trial_from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(trial_dir.glob("trial_*.json"))
        ]

    def load_trial(self, experiment_id: str, trial_id: str) -> Trial:
        path = self._trial_dir(experiment_id) / f"{trial_id}.json"
        return trial_from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_trial(self, experiment_id: str, trial: Trial) -> None:
        path = self._trial_dir(experiment_id) / f"{trial.trial_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trial.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def best_metric_value(self, study: HPOStudy, exclude_trial_id: Optional[str] = None) -> Optional[float]:
        objective = study.objectives[0]
        values = [
            trial.metrics[objective.metric]
            for trial in self.list_trials(study.experiment_id)
            if trial.trial_id != exclude_trial_id
            and isinstance(trial.metrics.get(objective.metric), (int, float))
        ]
        if not values:
            return None
        return min(values) if objective.mode == "min" else max(values)

    def _refresh_study(self, study: HPOStudy) -> None:
        objective = study.objectives[0]
        trials = self.list_trials(study.experiment_id)
        candidates = [
            trial for trial in trials
            if trial.status in {"completed", "promoted"}
            and isinstance(trial.metrics.get(objective.metric), (int, float))
        ]
        if candidates:
            reverse = objective.mode == "max"
            candidates.sort(key=lambda trial: trial.metrics[objective.metric], reverse=reverse)
            study.best_trial_id = candidates[0].trial_id
        study.updated_at = datetime.now().isoformat()
        self._save_study(study)

    def _save_study(self, study: HPOStudy) -> None:
        path = self._study_path(study.experiment_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(study.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        trials = self.list_trials(study.experiment_id)
        self.tracker.update_hpo_experiment(
            study.experiment_id,
            extensions={"optimization": {
                "study": study.to_dict(),
                "trial_summary": [
                    {
                        "trial_id": trial.trial_id,
                        "status": trial.status,
                        "rung": trial.rung,
                        "parameters": trial.parameters,
                        "budget": trial.budget.to_dict(),
                        "metrics": trial.metrics,
                        "stop_reason": trial.stop_reason,
                    }
                    for trial in trials
                ],
            }},
            artifacts=[{
                "type": "hpo_study",
                "name": study.study_id,
                "path": str(path),
            }],
        )

    @staticmethod
    def _study_path(experiment_id: str) -> Path:
        return get_experiment_artifact_dir(experiment_id, "hpo_study", "hpo", create=True) / "study.json"

    @staticmethod
    def _trial_dir(experiment_id: str) -> Path:
        return get_experiment_artifact_dir(experiment_id, "hpo_study", "hpo", create=True) / "trials"


def search_space_from_dict(data: Dict[str, Any]) -> SearchSpace:
    return SearchSpace(
        parameters=[SearchParameter(**item) for item in data.get("parameters") or []],
        constraints=data.get("constraints") or [],
    )


def study_from_dict(data: Dict[str, Any]) -> HPOStudy:
    return HPOStudy(
        study_id=data["study_id"],
        experiment_id=data["experiment_id"],
        strategy=data["strategy"],
        search_space=search_space_from_dict(data["search_space"]),
        objectives=[Objective(**item) for item in data.get("objectives") or []],
        budgets=[TrialBudget(**item) for item in data.get("budgets") or []],
        reduction_factor=data.get("reduction_factor", 3),
        max_trials=data.get("max_trials"),
        constraints=data.get("constraints") or [],
        trial_ids=data.get("trial_ids") or [],
        best_trial_id=data.get("best_trial_id"),
        status=data.get("status", "created"),
        stop_reason=data.get("stop_reason"),
        random_seed=data.get("random_seed", 0),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )


def trial_from_dict(data: Dict[str, Any]) -> Trial:
    return Trial(
        trial_id=data["trial_id"],
        parameters=data.get("parameters") or {},
        budget=TrialBudget(**data["budget"]),
        status=data.get("status", "suggested"),
        parent_trial_id=data.get("parent_trial_id"),
        rung=data.get("rung", 0),
        metrics=data.get("metrics") or {},
        intermediate_metrics=data.get("intermediate_metrics") or [],
        cost=data.get("cost") or {},
        artifacts=data.get("artifacts") or [],
        stop_reason=data.get("stop_reason"),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )
