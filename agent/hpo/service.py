"""Persistent HPO study and trial lifecycle service."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agent.utils import ExperimentTracker, get_experiment_artifact_dir

from .contracts import HPOStudy, Objective, SearchParameter, SearchSpace, StrategyProposal, Trial, TrialBudget
from .feedback import HPOFeedbackAnalyzer
from .policies import EarlyStoppingPolicy, StopDecision, StrategyDecisionPolicy
from .strategies import STRATEGIES, CandidateStrategy, SuccessiveHalvingStrategy

TRIAL_STATUSES = {"suggested", "running", "completed", "promoted", "stopped", "failed"}
TERMINAL_TRIAL_STATUSES = {"completed", "promoted", "stopped", "failed"}
TRIAL_TRANSITIONS = {
    "suggested": {"running", "failed", "stopped"},
    "running": {"completed", "failed", "stopped"},
    "completed": {"promoted", "failed"},
    "promoted": set(),
    "stopped": set(),
    "failed": set(),
}


class HPOService:
    def __init__(self, tracker: Optional[ExperimentTracker] = None) -> None:
        self.tracker = tracker or ExperimentTracker()
        self.halving_strategy = SuccessiveHalvingStrategy()

    @staticmethod
    def register_strategy(strategy: CandidateStrategy) -> None:
        """Register a candidate generator without changing scheduler code."""
        STRATEGIES.register(strategy)

    @staticmethod
    def available_strategies() -> List[str]:
        return STRATEGIES.names()

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
        initial_trial_count: Optional[int] = None,
        promotion_limits: Optional[List[int]] = None,
        max_training_runs: Optional[int] = None,
        min_completed_per_rung: int = 1,
    ) -> HPOStudy:
        if self.tracker.get_experiment(experiment_id) is None:
            raise ValueError(f"HPO experiment not found: {experiment_id}")
        self.validate_study_plan(
            search_space,
            objectives,
            budgets,
            strategy=strategy,
            reduction_factor=reduction_factor,
            max_trials=max_trials,
            initial_trial_count=initial_trial_count,
            promotion_limits=promotion_limits,
            max_training_runs=max_training_runs,
            min_completed_per_rung=min_completed_per_rung,
        )
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
            initial_trial_count=initial_trial_count,
            promotion_limits=list(promotion_limits or []),
            max_training_runs=max_training_runs or max_trials,
            min_completed_per_rung=min_completed_per_rung,
            random_seed=random_seed,
            created_at=now,
            updated_at=now,
        )
        self._save_study(study)
        return study

    def validate_study_plan(
        self,
        search_space: SearchSpace,
        objectives: List[Objective],
        budgets: List[TrialBudget],
        *,
        strategy: str = "successive_halving",
        reduction_factor: int = 3,
        max_trials: Optional[int] = None,
        initial_trial_count: Optional[int] = None,
        promotion_limits: Optional[List[int]] = None,
        max_training_runs: Optional[int] = None,
        min_completed_per_rung: int = 1,
    ) -> None:
        """Validate a proposed Study plan without creating records or artifacts."""
        if not search_space.parameters:
            raise ValueError("search space must contain at least one parameter")
        parameter_names = [parameter.name for parameter in search_space.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("search space parameter names must be unique")
        if not objectives:
            raise ValueError("study must contain at least one objective")
        if not budgets:
            raise ValueError("study must contain at least one budget rung")
        candidate_strategy = STRATEGIES.get(strategy)
        candidate_strategy.validate(search_space)
        if reduction_factor < 2:
            raise ValueError("reduction_factor must be at least 2")
        if max_trials is not None and max_trials <= 0:
            raise ValueError("max_trials must be positive")
        if initial_trial_count is not None and initial_trial_count <= 0:
            raise ValueError("initial_trial_count must be positive")
        if max_training_runs is not None and max_training_runs <= 0:
            raise ValueError("max_training_runs must be positive")
        if min_completed_per_rung <= 0:
            raise ValueError("min_completed_per_rung must be positive")
        if any(limit < 0 for limit in (promotion_limits or [])):
            raise ValueError("promotion_limits must be non-negative")
        effective_run_limit = max_training_runs or max_trials
        if initial_trial_count is not None and effective_run_limit is not None:
            if initial_trial_count > effective_run_limit:
                raise ValueError("initial_trial_count cannot exceed max_training_runs")
        if len(promotion_limits or []) > max(len(budgets) - 1, 0):
            raise ValueError("promotion_limits cannot exceed the number of promotion rungs")
        for objective in objectives:
            if objective.mode not in {"min", "max"}:
                raise ValueError(f"unsupported objective mode: {objective.mode}")
        for budget in budgets:
            if budget.epochs is not None and budget.epochs <= 0:
                raise ValueError("budget epochs must be positive")
            if budget.data_fraction is not None and not 0 < budget.data_fraction <= 1:
                raise ValueError("budget data_fraction must be in (0, 1]")
            if budget.max_duration_seconds is not None and budget.max_duration_seconds <= 0:
                raise ValueError("budget max_duration_seconds must be positive")

    def suggest_trials(self, study: HPOStudy, count: int) -> List[Trial]:
        existing_trials = self.list_trials(study.experiment_id)
        initial_trials = [trial for trial in existing_trials if trial.rung == 0]
        initial_limit = (
            study.initial_trial_count or study.max_trials
            if study.strategy == "successive_halving"
            else study.max_training_runs or study.max_trials
        )
        if initial_limit is not None:
            count = min(count, max(initial_limit - len(initial_trials), 0))
        count = min(count, self.remaining_training_runs(study))
        if count <= 0:
            return []
        candidate_strategy = study.candidate_strategy or study.strategy
        candidates = STRATEGIES.get(candidate_strategy).suggest(
            study.search_space,
            count,
            seed=study.random_seed + len(existing_trials),
            existing=[trial.parameters for trial in existing_trials],
            history=existing_trials,
            objective=study.objectives[0],
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

    def review_strategy(
        self,
        study: HPOStudy,
        proposal: Optional[StrategyProposal] = None,
        *,
        trigger: str = "periodic",
    ) -> Dict[str, Any]:
        """Analyze feedback and safely update only subsequent candidate generation."""
        trials = self.list_trials(study.experiment_id)
        analyzer = HPOFeedbackAnalyzer()
        feedback = analyzer.analyze(study, trials)
        if proposal is None:
            proposal = analyzer.propose(study, feedback, self.available_strategies())
        original_proposal = proposal
        blocked_fields = []
        if proposal is not None and (proposal.max_training_runs is not None or proposal.budgets is not None):
            if proposal.max_training_runs is not None:
                blocked_fields.append({"field": "max_training_runs", "reason": "runtime reviews cannot change Study run quota"})
            if proposal.budgets is not None:
                blocked_fields.append({"field": "budgets", "reason": "runtime reviews cannot change active Study rung budgets"})
            proposal = StrategyProposal(
                action=proposal.action,
                requested_strategy=proposal.requested_strategy,
                search_space=proposal.search_space,
                reason_codes=proposal.reason_codes,
                evidence=proposal.evidence,
                expected_effect=proposal.expected_effect,
                confidence=proposal.confidence,
            )
        decision = StrategyDecisionPolicy().review(
            proposal,
            base_strategy=study.candidate_strategy or study.strategy,
            base_search_space=study.search_space,
            base_budgets=study.budgets,
            hard_max_training_runs=int(study.max_training_runs or study.max_trials or 1),
            objectives=study.objectives,
            available_strategies=self.available_strategies(),
            validate_plan=self.validate_study_plan,
        )
        if blocked_fields:
            decision.proposal_id = original_proposal.proposal_id
            decision.proposal = original_proposal.to_dict()
            decision.rejected_fields.extend(blocked_fields)
            decision.decision = "approved_with_changes" if decision.accepted_fields else "rejected"
            decision.reason_codes = ["runtime_proposal_partially_approved" if decision.accepted_fields else "runtime_proposal_rejected"]
        study.candidate_strategy = decision.adopted_strategy
        study.search_space = search_space_from_dict(decision.adopted_search_space)
        review = {
            "trigger": trigger,
            "feedback": feedback,
            "proposal": original_proposal.to_dict() if original_proposal else None,
            "decision": decision.to_dict(),
            "applied_candidate_strategy": study.candidate_strategy,
            "applied_search_space": study.search_space.to_dict(),
            "created_at": datetime.now().isoformat(),
        }
        study.strategy_reviews.append(review)
        study.updated_at = datetime.now().isoformat()
        self._save_study(study)
        return review

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
        if status != trial.status and status not in TRIAL_TRANSITIONS.get(trial.status, set()):
            raise ValueError(f"invalid trial transition: {trial.status} -> {status}")
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

    def retry_trial(self, study: HPOStudy, trial_id: str, reason: str) -> Trial:
        """Explicitly reopen a failed trial for a bounded scheduler retry."""
        if self.remaining_training_runs(study) <= 0:
            raise ValueError("max_training_runs exhausted")
        trial = self.load_trial(study.experiment_id, trial_id)
        if trial.status != "failed":
            raise ValueError(f"only failed trials can be retried, got: {trial.status}")
        trial.status = "suggested"
        trial.stop_reason = reason
        trial.cost["retry_count"] = int(trial.cost.get("retry_count", 0)) + 1
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

    def completion_errors(self, study: HPOStudy) -> List[str]:
        """Return reasons why a study cannot be considered successfully complete."""
        objective = study.objectives[0]
        trials = self.list_trials(study.experiment_id)
        completed = [
            trial for trial in trials
            if trial.status in {"completed", "promoted"}
            and isinstance(trial.metrics.get(objective.metric), (int, float))
        ]
        errors: List[str] = []
        if not completed:
            errors.append("no completed trial with a valid primary metric")
        active = [trial.trial_id for trial in trials if trial.status in {"suggested", "running"}]
        if active:
            errors.append(f"trials without terminal status: {', '.join(active)}")
        if study.best_trial_id is None:
            errors.append("best_trial_id is missing")
        return errors

    def complete_study(self, study: HPOStudy, stop_reason: Optional[str] = None) -> HPOStudy:
        errors = self.completion_errors(study)
        if errors:
            raise ValueError("; ".join(errors))
        return self.finish_study(study, "completed", stop_reason)

    def promote_trials(self, study: HPOStudy) -> List[Trial]:
        objective = study.objectives[0]
        trials = self.list_trials(study.experiment_id)
        completed_by_rung: Dict[int, List[Trial]] = {}
        for trial in trials:
            if trial.status == "completed":
                completed_by_rung.setdefault(trial.rung, []).append(trial)
        eligible_rungs = []
        for rung, items in completed_by_rung.items():
            if rung + 1 >= len(study.budgets) or len(items) < study.min_completed_per_rung:
                continue
            limit = study.promotion_limits[rung] if rung < len(study.promotion_limits) else None
            destination_count = len([trial for trial in trials if trial.rung == rung + 1])
            if limit is None or destination_count < limit:
                eligible_rungs.append(rung)
        if not eligible_rungs:
            return []
        source_rung = min(eligible_rungs)
        destination_rung = source_rung + 1
        already_at_destination = len([trial for trial in trials if trial.rung == destination_rung])
        promotion_limit = (
            study.promotion_limits[source_rung]
            if source_rung < len(study.promotion_limits)
            else None
        )
        remaining_for_rung = (
            max(promotion_limit - already_at_destination, 0)
            if promotion_limit is not None else None
        )
        candidates = self.halving_strategy.promote(
            trials,
            objective,
            study.reduction_factor,
            rung=source_rung,
            limit=remaining_for_rung,
        )
        promoted: List[Trial] = []
        remaining = self.remaining_training_runs(study)
        for candidate in candidates:
            if remaining is not None and len(promoted) >= remaining:
                break
            next_rung = destination_rung
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

    def remaining_training_runs(self, study: HPOStudy) -> int:
        limit = study.max_training_runs or study.max_trials
        if limit is None:
            return 2**31 - 1
        return max(limit - self.training_runs_used(study), 0)

    def training_runs_used(self, study: HPOStudy) -> int:
        trials = self.list_trials(study.experiment_id)
        retry_count = sum(int(trial.cost.get("retry_count", 0)) for trial in trials)
        return len(trials) + retry_count

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
        objective = study.objectives[0] if study.objectives else None
        best_trial = (
            self.load_trial(study.experiment_id, study.best_trial_id)
            if study.best_trial_id else None
        )
        best_metric = (
            best_trial.metrics.get(objective.metric)
            if best_trial is not None and objective is not None else None
        )
        self.tracker.update_hpo_experiment(
            study.experiment_id,
            extensions={"optimization": {
                "study": {
                    "study_id": study.study_id,
                    "experiment_id": study.experiment_id,
                    "status": study.status,
                    "strategy": study.strategy,
                    "candidate_strategy": study.candidate_strategy,
                    "best_trial_id": study.best_trial_id,
                    "best_metric": best_metric,
                    "objective": objective.to_dict() if objective else None,
                    "trial_ids": list(study.trial_ids),
                    "max_training_runs": study.max_training_runs,
                    "updated_at": study.updated_at,
                    "study_artifact": str(path),
                },
                "trial_summary": [
                    {
                        "trial_id": trial.trial_id,
                        "status": trial.status,
                        "phase": _trial_phase(trial),
                        "rung": trial.rung,
                        "parameters": trial.parameters,
                        "budget": trial.budget.to_dict(),
                        "metrics": trial.metrics,
                        "artifacts": trial.artifacts,
                        "stop_reason": trial.stop_reason,
                        "updated_at": trial.updated_at,
                    }
                    for trial in trials
                ],
            }},
            metrics={"best": {"trial_id": study.best_trial_id, objective.metric: best_metric} if objective and best_metric is not None else {}},
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



def _trial_phase(trial: Trial) -> str:
    if trial.status in {"failed", "stopped"}:
        return trial.status
    if trial.status in {"completed", "promoted"}:
        return "completed"
    if (trial.cost or {}).get("training") and not any(
        artifact.get("type") == "predictions" for artifact in trial.artifacts
    ):
        return "evaluation_pending"
    if trial.status == "running":
        return "training"
    return trial.status
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
        candidate_strategy=data.get("candidate_strategy"),
        reduction_factor=data.get("reduction_factor", 3),
        max_trials=data.get("max_trials"),
        initial_trial_count=data.get("initial_trial_count"),
        promotion_limits=data.get("promotion_limits") or [],
        max_training_runs=data.get("max_training_runs"),
        min_completed_per_rung=data.get("min_completed_per_rung", 1),
        constraints=data.get("constraints") or [],
        strategy_reviews=data.get("strategy_reviews") or [],
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
