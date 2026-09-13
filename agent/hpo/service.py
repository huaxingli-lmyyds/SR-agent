"""Persistent HPO study and trial lifecycle service."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from hashlib import sha256
from math import ceil, isfinite
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

from agent.core.metrics import is_finite_metric, require_finite_metric
from agent.utils import ExperimentTracker, get_experiment_artifact_dir  # noqa: F401

from .contracts import HPOStudy, Objective, SearchParameter, SearchSpace, StrategyProposal, Trial, TrialBudget
from .effects import summarize_decision_effect
from .feedback import HPOFeedbackAnalyzer
from .history import incompatibility_reason
from .policies import (
    EarlyStoppingPolicy,
    EvidenceGate,
    OptimizationPlanDecisionPolicy,
    StopDecision,
    StrategyDecisionPolicy,
)
from .strategies import (
    STRATEGIES,
    CandidateStrategy,
    SuccessiveHalvingStrategy,
)

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
CONTROLLER_MODES = {"auto", "fixed", "rule", "llm"}


class HPOService:
    def __init__(
        self,
        tracker: Optional[ExperimentTracker] = None,
        parameter_validator: Optional[Callable[[Dict[str, Any]], None]] = None,
        evidence_gate: Optional[EvidenceGate] = None,
    ) -> None:
        self.tracker = tracker or ExperimentTracker()
        self.parameter_validator = parameter_validator
        self.evidence_gate = evidence_gate or EvidenceGate()
        self.halving_strategy = SuccessiveHalvingStrategy()

    @staticmethod
    def register_strategy(strategy: CandidateStrategy) -> None:
        """Register a candidate generator without changing scheduler code."""
        STRATEGIES.register(strategy)

    @staticmethod
    def available_strategies() -> List[str]:
        return STRATEGIES.names()

    @staticmethod
    def available_samplers() -> List[str]:
        return [
            name for name in STRATEGIES.names()
            if name != "successive_halving"
        ]

    @staticmethod
    def available_pruners() -> List[str]:
        return ["none", "successive_halving"]

    @staticmethod
    def _strategy_components(
        strategy: str,
        sampler_strategy: Optional[str],
        pruner_strategy: Optional[str],
    ) -> tuple[str, str]:
        sampler = sampler_strategy or (
            "random_search" if strategy == "successive_halving" else strategy
        )
        pruner = pruner_strategy or (
            "successive_halving" if strategy == "successive_halving" else "none"
        )
        return sampler, pruner

    @classmethod
    def _active_pruner(cls, study: HPOStudy) -> str:
        return cls._strategy_components(
            study.strategy,
            study.sampler_strategy,
            study.pruner_strategy,
        )[1]

    def create_study(
        self,
        experiment_id: str,
        search_space: SearchSpace,
        objectives: List[Objective],
        budgets: List[TrialBudget],
        *,
        strategy: str = "successive_halving",
        sampler_strategy: Optional[str] = None,
        pruner_strategy: Optional[str] = None,
        reduction_factor: int = 3,
        random_seed: int = 0,
        max_trials: Optional[int] = None,
        initial_trial_count: Optional[int] = None,
        promotion_limits: Optional[List[int]] = None,
        max_training_runs: Optional[int] = None,
        min_completed_per_rung: int = 1,
        warm_start_trials: Optional[List[Dict[str, Any]]] = None,
        candidate_batch_size: Optional[int] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
        fallback_sampler_strategy: str = "random_search",
        hypotheses: Optional[List[Dict[str, Any]]] = None,
        candidate_proposals: Optional[List[Dict[str, Any]]] = None,
        proposal_id: Optional[str] = None,
        planning_decision_id: Optional[str] = None,
        controller_mode: str = "auto",
    ) -> HPOStudy:
        self._validate_record_id(experiment_id, "experiment_id")
        if pruner_strategy is None and strategy == "successive_halving" and len(budgets) < 2:
            pruner_strategy = "none"
        experiment = self.tracker.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"HPO experiment not found: {experiment_id}")
        if experiment.get("experiment_type") != "hpo":
            raise ValueError(f"experiment is not an HPO record: {experiment_id}")
        if controller_mode not in CONTROLLER_MODES:
            raise ValueError("controller_mode must be one of: fixed, rule, llm")
        recorded_task = experiment.get("task") or {}
        if objectives:
            recorded_metric = recorded_task.get("primary_metric")
            recorded_mode = recorded_task.get("metric_mode")
            if recorded_metric is not None and recorded_metric != objectives[0].metric:
                raise ValueError(
                    "Study objective conflicts with experiment primary_metric: "
                    f"{objectives[0].metric!r} != {recorded_metric!r}"
                )
            if recorded_mode is not None and recorded_mode != objectives[0].mode:
                raise ValueError(
                    "Study objective conflicts with experiment metric_mode: "
                    f"{objectives[0].mode!r} != {recorded_mode!r}"
                )
        self.validate_study_plan(
            search_space,
            objectives,
            budgets,
            strategy=strategy,
            sampler_strategy=sampler_strategy,
            pruner_strategy=pruner_strategy,
            reduction_factor=reduction_factor,
            max_trials=max_trials,
            initial_trial_count=initial_trial_count,
            promotion_limits=promotion_limits,
            max_training_runs=max_training_runs,
            min_completed_per_rung=min_completed_per_rung,
            candidate_batch_size=candidate_batch_size,
            sampler_config=sampler_config,
        )
        sampler_strategy, pruner_strategy = self._strategy_components(
            strategy,
            sampler_strategy,
            pruner_strategy,
        )
        if sampler_strategy == "agent_proposal" and not candidate_proposals:
            sampler_strategy = fallback_sampler_strategy
            sampler_config = {}
            if pruner_strategy == "none":
                strategy = sampler_strategy
            self.validate_study_plan(
                search_space,
                objectives,
                budgets,
                strategy=strategy,
                sampler_strategy=sampler_strategy,
                pruner_strategy=pruner_strategy,
                reduction_factor=reduction_factor,
                max_trials=max_trials,
                initial_trial_count=initial_trial_count,
                promotion_limits=promotion_limits,
                max_training_runs=max_training_runs,
                min_completed_per_rung=min_completed_per_rung,
                candidate_batch_size=candidate_batch_size,
                sampler_config=sampler_config,
            )
        normalized_sampler_config = self.validate_sampler_config(
            sampler_strategy, sampler_config
        )
        now = datetime.now().isoformat()
        study = HPOStudy(
            study_id=f"study_{uuid4().hex[:10]}",
            experiment_id=experiment_id,
            strategy=strategy,
            search_space=search_space,
            objectives=objectives,
            budgets=budgets,
            sampler_strategy=sampler_strategy,
            pruner_strategy=pruner_strategy,
            controller_mode=controller_mode,
            reduction_factor=reduction_factor,
            max_trials=max_trials,
            initial_trial_count=initial_trial_count,
            promotion_limits=list(promotion_limits or []),
            max_training_runs=max_training_runs or max_trials,
            min_completed_per_rung=min_completed_per_rung,
            candidate_batch_size=candidate_batch_size,
            sampler_config=normalized_sampler_config,
            search_phases=[{
                "phase_index": 0,
                "sampler": sampler_strategy,
                "sampler_config": normalized_sampler_config,
                "start_trial_index": 0,
                "end_trial_index": None,
                "trigger": "study_creation",
                "proposal_id": proposal_id,
                "decision_id": planning_decision_id,
                "reason_codes": [],
                "fallback_sampler": (
                    (
                        fallback_sampler_strategy
                        if fallback_sampler_strategy != "agent_proposal"
                        else "random_search"
                    )
                    if sampler_strategy == "agent_proposal" else None
                ),
                "fallback_sampler_config": {},
                "created_at": now,
            }],
            hypotheses=[],
            scheduler_state={
                "active_decision_id": planning_decision_id,
                "active_proposal_id": proposal_id,
            },
            warm_start_trials=list(warm_start_trials or []),
            history_context=self._build_history_context(experiment_id, objectives[0]),
            random_seed=random_seed,
            created_at=now,
            updated_at=now,
        )
        if hypotheses:
            hypothesis_review = self.register_hypotheses(
                study,
                hypotheses,
                proposal_id=proposal_id,
            )
            study.candidate_proposal_reviews.append({
                "trigger": "study_planning_hypotheses",
                "proposal_id": proposal_id,
                **hypothesis_review,
                "created_at": datetime.now().isoformat(),
            })
        if candidate_proposals:
            self.enqueue_candidate_proposals(
                study,
                candidate_proposals,
                proposal_id=proposal_id,
                trigger="study_planning",
                persist=False,
            )
        if sampler_strategy == "agent_proposal" and not study.pending_candidate_proposals:
            restored_sampler = (
                fallback_sampler_strategy
                if fallback_sampler_strategy != "agent_proposal" else "random_search"
            )
            STRATEGIES.get(restored_sampler).validate(search_space)
            study.sampler_strategy = restored_sampler
            study.candidate_strategy = restored_sampler
            if study.pruner_strategy == "none":
                study.strategy = restored_sampler
            study.sampler_config = self.validate_sampler_config(restored_sampler, {})
            study.search_phases[0]["sampler"] = restored_sampler
            study.search_phases[0]["sampler_config"] = {}
            study.candidate_proposal_reviews.append({
                "trigger": "study_planning_fallback",
                "proposal_id": proposal_id,
                "accepted": [],
                "rejected": [{
                    "reason": (
                        "agent_proposal produced no valid candidates; "
                        f"restored {restored_sampler}"
                    ),
                }],
                "created_at": datetime.now().isoformat(),
            })
        self._save_study(study)
        return study

    def validate_study_plan(
        self,
        search_space: SearchSpace,
        objectives: List[Objective],
        budgets: List[TrialBudget],
        *,
        strategy: str = "successive_halving",
        sampler_strategy: Optional[str] = None,
        pruner_strategy: Optional[str] = None,
        reduction_factor: int = 3,
        max_trials: Optional[int] = None,
        initial_trial_count: Optional[int] = None,
        promotion_limits: Optional[List[int]] = None,
        max_training_runs: Optional[int] = None,
        min_completed_per_rung: int = 1,
        candidate_batch_size: Optional[int] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Validate a proposed Study plan without creating records or artifacts."""
        if pruner_strategy is None and strategy == "successive_halving" and len(budgets) < 2:
            pruner_strategy = "none"
        if not search_space.parameters:
            raise ValueError("search space must contain at least one parameter")
        parameter_names = [parameter.name for parameter in search_space.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("search space parameter names must be unique")
        declared_parameters = set(parameter_names)
        for parameter in search_space.parameters:
            unknown_conditions = sorted(set(parameter.condition) - declared_parameters)
            if unknown_conditions:
                raise ValueError(
                    f"condition for {parameter.name} references unknown parameters: "
                    f"{', '.join(unknown_conditions)}"
                )
        for constraint in search_space.constraints:
            if not isinstance(constraint, dict):
                raise ValueError("search-space constraints must be objects")
            constrained_parameter = constraint.get("parameter")
            if constrained_parameter not in declared_parameters:
                raise ValueError(
                    f"constraint references unknown parameter: {constrained_parameter}"
                )
            operator = constraint.get("operator")
            if operator not in {"lte", "gte", "eq", "in"}:
                raise ValueError(f"unsupported constraint operator: {operator}")
            if operator == "in" and not isinstance(
                constraint.get("value"), (list, tuple, set)
            ):
                raise ValueError("constraint operator 'in' requires a collection value")
        if not objectives:
            raise ValueError("study must contain at least one objective")
        if len(objectives) != 1:
            raise ValueError(
                "this HPO workflow requires exactly one authoritative Study objective"
            )
        if not budgets:
            raise ValueError("study must contain at least one budget rung")
        sampler_strategy, pruner_strategy = self._strategy_components(
            strategy,
            sampler_strategy,
            pruner_strategy,
        )
        candidate_strategy = STRATEGIES.get(sampler_strategy)
        candidate_strategy.validate(search_space)
        self.validate_sampler_config(sampler_strategy, sampler_config)
        if pruner_strategy not in self.available_pruners():
            raise ValueError(f"unsupported HPO pruner: {pruner_strategy}")
        if pruner_strategy == "successive_halving" and len(budgets) < 2:
            raise ValueError("successive_halving pruner requires at least two budget rungs")
        if pruner_strategy == "none" and len(budgets) != 1:
            raise ValueError("pruner 'none' requires exactly one budget rung")
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
        if candidate_batch_size is not None and candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive")
        if any(limit < 0 for limit in (promotion_limits or [])):
            raise ValueError("promotion_limits must be non-negative")
        effective_run_limit = max_training_runs or max_trials
        if initial_trial_count is not None and effective_run_limit is not None:
            if initial_trial_count > effective_run_limit:
                raise ValueError("initial_trial_count cannot exceed max_training_runs")
        if len(promotion_limits or []) > max(len(budgets) - 1, 0):
            raise ValueError("promotion_limits cannot exceed the number of promotion rungs")
        if pruner_strategy == "none" and promotion_limits:
            raise ValueError("promotion_limits require a fidelity pruner")
        if (
            pruner_strategy == "successive_halving"
            and initial_trial_count is not None
            and effective_run_limit is not None
            and initial_trial_count + sum(promotion_limits or []) > effective_run_limit
        ):
            raise ValueError("planned initial and promotion runs exceed max_training_runs")
        for objective in objectives:
            if objective.mode not in {"min", "max"}:
                raise ValueError(f"unsupported objective mode: {objective.mode}")
        for budget in budgets:
            if budget.epochs is not None and budget.epochs <= 0:
                raise ValueError("budget epochs must be positive")
            if budget.data_fraction is not None and not 0 < budget.data_fraction <= 1:
                raise ValueError("budget data_fraction must be in (0, 1]")
            if budget.max_duration_seconds is not None and (
                isinstance(budget.max_duration_seconds, bool)
                or not isinstance(budget.max_duration_seconds, (int, float))
                or not isfinite(float(budget.max_duration_seconds))
                or float(budget.max_duration_seconds) <= 0
            ):
                raise ValueError(
                    "budget max_duration_seconds must be a finite positive number"
                )

    @staticmethod
    def validate_sampler_config(
        sampler_strategy: str,
        sampler_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        config = dict(sampler_config or {})
        if sampler_strategy != "tpe":
            if config:
                raise ValueError(f"sampler_config is not supported by {sampler_strategy}")
            return {}
        unknown = sorted(set(config) - {"n_startup_trials", "multivariate"})
        if unknown:
            raise ValueError(f"unknown TPE sampler_config fields: {', '.join(unknown)}")
        startup = config.get("n_startup_trials", 3)
        if isinstance(startup, bool) or not isinstance(startup, int) or startup <= 0:
            raise ValueError("TPE n_startup_trials must be a positive integer")
        multivariate = config.get("multivariate", False)
        if not isinstance(multivariate, bool):
            raise ValueError("TPE multivariate must be boolean")
        return {"n_startup_trials": startup, "multivariate": multivariate}

    def _build_history_context(self, experiment_id: str, objective: Objective) -> Dict[str, Any]:
        record = self.tracker.get_experiment(experiment_id) or {}
        task, model = record.get("task") or {}, record.get("model") or {}
        execution = record.get("execution") or {}
        return {
            "version": 1,
            "dataset": task.get("dataset"),
            "dataset_id": task.get("dataset_id"),
            "dataset_version": task.get("dataset_version"),
            "task_type": task.get("type"),
            "model_family": model.get("family"),
            "implementation": model.get("implementation"),
            "runner": execution.get("runner"),
            "objective": {"metric": objective.metric, "mode": objective.mode},
            "metric_protocol": task.get("metric_protocol"),
            "metric_units": task.get("metric_units"),
            "config_sha256": sha256(self.tracker.get_config_snapshot(experiment_id).read_bytes()).hexdigest(),
            "evaluation_config": execution.get("evaluation_config_path"),
            "verification_config_sha256": execution.get(
                "verification_config_sha256"
            ),
            "validation_pairs": execution.get("validation_pairs_path"),
            "validation_pairs_sha256": execution.get("validation_pairs_sha256"),
            "training_exclusion_pairs_sha256": execution.get(
                "training_exclusion_pairs_sha256"
            ),
        }

    def compatible_warm_start_trials(self, study: HPOStudy) -> List[Trial]:
        """Quarantine incompatible history from both sampling and deduplication."""
        if not study.history_context:
            study.history_context = self._build_history_context(study.experiment_id, study.objectives[0])
        accepted, rejected = [], []
        for item in study.warm_start_trials:
            reason = incompatibility_reason(
                item, study.budgets[0].to_dict(), study.history_context, study.objectives[0].metric,
            )
            if reason is None:
                try:
                    accepted.append(trial_from_dict(item))
                    continue
                except (KeyError, TypeError, ValueError):
                    reason = "invalid_history_record"
            rejected.append({
                "trial_id": item.get("trial_id") if isinstance(item, dict) else None,
                "reason": reason,
            })
        review = {"trigger": "warm_start_filter", "accepted_count": len(accepted), "rejected": rejected}
        prior = next((r for r in reversed(study.candidate_generation_reviews)
                      if r.get("trigger") == "warm_start_filter"), None)
        if study.warm_start_trials and prior != review:
            study.candidate_generation_reviews.append(review)
        return accepted

    def _agent_fallback(
        self, study: HPOStudy, preferred_sampler: Optional[str] = None,
        preferred_config: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Resolve a usable, non-agent fallback, including older saved phases."""
        choices = [(preferred_sampler, preferred_config or {})]
        choices.extend(
            (phase.get("fallback_sampler"), phase.get("fallback_sampler_config") or {})
            for phase in reversed(study.search_phases)
        )
        choices.extend([(study.sampler_strategy, {}), ("random_search", {})])
        for sampler, config in choices:
            if sampler not in self.available_samplers() or sampler == "agent_proposal":
                continue
            try:
                STRATEGIES.get(sampler).validate(study.search_space)
                return sampler, self.validate_sampler_config(sampler, config)
            except ValueError:
                continue
        raise ValueError("no valid fallback sampler for exhausted agent_proposal queue")

    def suggest_trials(self, study: HPOStudy, count: int) -> List[Trial]:
        existing_trials = self.list_trials(study.experiment_id)
        initial_trials = [trial for trial in existing_trials if trial.rung == 0]
        initial_limit = (
            study.initial_trial_count or study.max_trials
            if self._active_pruner(study) == "successive_halving"
            else study.max_training_runs or study.max_trials
        )
        if initial_limit is not None:
            count = min(count, max(initial_limit - len(initial_trials), 0))
        count = min(count, self.remaining_training_runs(study))
        if count <= 0:
            return []
        warm_start_trials = self.compatible_warm_start_trials(study)
        sampler_strategy = (
            study.candidate_strategy
            or study.sampler_strategy
            or self._strategy_components(study.strategy, None, None)[0]
        )
        all_history = [*warm_start_trials, *existing_trials]
        sampler_history = (
            [trial for trial in all_history if trial.rung == 0]
            if self._active_pruner(study) == "successive_halving"
            else all_history
        )
        strategy = STRATEGIES.get(sampler_strategy)
        suggest_kwargs = {
            "seed": study.random_seed + len(existing_trials),
            "existing": [trial.parameters for trial in all_history],
            "history": sampler_history,
            "objective": study.objectives[0],
            "config": study.sampler_config,
        }
        if sampler_strategy == "agent_proposal":
            suggest_kwargs["proposed_candidates"] = list(
                study.pending_candidate_proposals
            )
        raw_suggestions = strategy.suggest(
            study.search_space,
            # A stale invalid item must not hide valid candidates later in the
            # persisted queue. Apply the batch limit after validation instead.
            max(count, len(study.pending_candidate_proposals))
            if sampler_strategy == "agent_proposal" else count,
            **suggest_kwargs,
        )
        suggestions = []
        rejected = []
        proposal_by_signature = {
            _candidate_signature(item.get("parameters") or {}): item
            for item in study.pending_candidate_proposals
        }
        for index, raw in enumerate(raw_suggestions):
            try:
                parameters = self.validate_candidate_parameters(study, dict(raw))
                metadata = proposal_by_signature.get(_candidate_signature(parameters), {})
                suggestions.append((parameters, metadata))
            except Exception as exc:
                rejected.append({"index": index, "parameters": raw, "reason": str(exc)})
        suggestions = suggestions[:count]
        if rejected:
            study.candidate_generation_reviews.append({
                "sampler": sampler_strategy,
                "search_phase": len(study.search_phases) - 1,
                "rejected": rejected,
                "created_at": datetime.now().isoformat(),
            })
        if not suggestions and sampler_strategy == "agent_proposal":
            # Also covers resumed queues containing only consumed/invalid items.
            # This fallback is bounded: _agent_fallback never returns agent_proposal.
            fallback, config = self._agent_fallback(study)
            now = datetime.now().isoformat()
            study.candidate_generation_reviews.append({
                "sampler": sampler_strategy,
                "trigger": "agent_queue_exhausted",
                "reason": "no usable pending agent candidate",
                "discarded_candidates": list(study.pending_candidate_proposals),
                "fallback_sampler": fallback,
                "created_at": now,
            })
            study.pending_candidate_proposals = []
            study.candidate_strategy = fallback
            study.sampler_config = config
            # The prior decision no longer governs candidates once its finite
            # agent queue has been exhausted and the deterministic fallback is
            # activated.  Do not attribute fallback Trials to that LLM decision.
            study.scheduler_state["active_decision_id"] = None
            study.scheduler_state["active_proposal_id"] = None
            if study.search_phases:
                study.search_phases[-1]["end_trial_index"] = len(existing_trials) - 1
            study.search_phases.append({
                "phase_index": len(study.search_phases),
                "sampler": fallback,
                "sampler_config": dict(config),
                "start_trial_index": len(existing_trials),
                "end_trial_index": None,
                "trigger": "agent_queue_exhausted",
                "proposal_id": None,
                "reason_codes": ["no_usable_agent_candidate"],
                "fallback_sampler": None,
                "fallback_sampler_config": {},
                "created_at": now,
            })
            self._save_study(study)
            return self.suggest_trials(study, count)
        if (
            not suggestions
            and sampler_strategy != "agent_proposal"
            and self._active_pruner(study) == "successive_halving"
            and initial_trials
            and len(initial_trials) < int(study.initial_trial_count or len(initial_trials))
        ):
            # A finite/conditional search space may be exhausted before the
            # requested startup size. Promote the complete feasible cohort.
            study.initial_trial_count = len(initial_trials)
            self._save_study(study)
        consumed_signatures = {
            _candidate_signature(parameters) for parameters, _ in suggestions
        }
        if sampler_strategy == "agent_proposal" and consumed_signatures:
            study.pending_candidate_proposals = [
                item for item in study.pending_candidate_proposals
                if _candidate_signature(item.get("parameters") or {})
                not in consumed_signatures
            ]
        if not suggestions:
            study.scheduler_state["candidate_generation_exhausted"] = {
                "sampler": sampler_strategy,
                "search_phase": max(len(study.search_phases) - 1, 0),
                "recorded_at": datetime.now().isoformat(),
            }
        else:
            study.scheduler_state.pop("candidate_generation_exhausted", None)
        now = datetime.now().isoformat()
        budget = study.budgets[0]
        active_decision_id = study.scheduler_state.get("active_decision_id")
        active_proposal_id = study.scheduler_state.get("active_proposal_id")
        trials = [
            Trial(
                trial_id=f"trial_{uuid4().hex[:10]}",
                parameters=parameters,
                budget=budget,
                candidate_source=f"sampler:{sampler_strategy}",
                search_phase=max(len(study.search_phases) - 1, 0),
                proposal_id=metadata.get("proposal_id"),
                hypothesis_id=metadata.get("hypothesis_id"),
                provenance={
                    "sampler": sampler_strategy,
                    "sampler_config": dict(study.sampler_config),
                    **{
                        key: value for key, value in metadata.items()
                        if key not in {"parameters", "proposal_id", "hypothesis_id"}
                    },
                    "history_context": dict(study.history_context),
                    "decision_id": active_decision_id,
                    "phase_proposal_id": active_proposal_id,
                },
                created_at=now,
                updated_at=now,
            )
            for parameters, metadata in suggestions
        ]
        for trial in trials:
            self.save_trial(study.experiment_id, trial)
            study.trial_ids.append(trial.trial_id)
        if active_decision_id and trials:
            for review in reversed(study.strategy_reviews):
                if (review.get("decision") or {}).get("decision_id") != active_decision_id:
                    continue
                affected = list(review.get("affected_trial_ids") or [])
                affected.extend(trial.trial_id for trial in trials)
                review["affected_trial_ids"] = list(dict.fromkeys(affected))
                review["decision"]["affected_trial_ids"] = list(review["affected_trial_ids"])
                break
        study.status = "running"
        study.updated_at = now
        self._save_study(study)
        return trials

    def enqueue_candidate_proposals(
        self,
        study: HPOStudy,
        proposals: List[Dict[str, Any]],
        *,
        proposal_id: Optional[str],
        trigger: str,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Validate agent-proposal candidates and queue only safe, unique parameters."""
        active_sampler = (
            study.candidate_strategy or study.sampler_strategy or study.strategy
        )
        if proposals and active_sampler != "agent_proposal":
            review = {
                "trigger": trigger,
                "proposal_id": proposal_id,
                "accepted": [],
                "rejected": [
                    {
                        "index": index,
                        "candidate_id": item.get("candidate_id") if isinstance(item, dict) else None,
                        "reason": (
                            "candidate_proposals require requested_sampler='agent_proposal'; "
                            f"active sampler is {active_sampler}"
                        ),
                    }
                    for index, item in enumerate(proposals)
                ],
                "created_at": datetime.now().isoformat(),
            }
            study.candidate_proposal_reviews.append(review)
            if persist:
                self._save_study(study)
            return review
        if proposals and not self.has_future_candidate_capacity(study):
            review = {
                "trigger": trigger,
                "proposal_id": proposal_id,
                "accepted": [],
                "rejected": [
                    {
                        "index": index,
                        "candidate_id": item.get("candidate_id") if isinstance(item, dict) else None,
                        "reason": "Study has no future candidate-generation capacity",
                    }
                    for index, item in enumerate(proposals)
                ],
                "created_at": datetime.now().isoformat(),
            }
            study.candidate_proposal_reviews.append(review)
            if persist:
                self._save_study(study)
            return review
        existing = [
            trial.parameters for trial in self.list_trials(study.experiment_id)
        ]
        existing.extend(
            trial.parameters for trial in self.compatible_warm_start_trials(study)
        )
        existing.extend(
            dict(item.get("parameters") or {})
            for item in study.pending_candidate_proposals
            if isinstance(item, dict)
        )
        seen = {_candidate_signature(item) for item in existing}
        known_hypotheses = {
            str(item.get("id")) for item in study.hypotheses
            if isinstance(item, dict) and item.get("id") is not None
        }
        known_candidate_ids = {
            str(item.get("candidate_id"))
            for item in study.pending_candidate_proposals
            if isinstance(item, dict) and item.get("candidate_id") is not None
        }
        for prior_review in study.candidate_proposal_reviews:
            if not isinstance(prior_review, dict):
                continue
            for item in prior_review.get("accepted") or []:
                if isinstance(item, dict) and item.get("candidate_id") is not None:
                    known_candidate_ids.add(str(item["candidate_id"]))
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for index, raw in enumerate(proposals or []):
            if not isinstance(raw, dict):
                rejected.append({"index": index, "reason": "candidate proposal must be an object"})
                continue
            try:
                parameters = self.validate_candidate_parameters(
                    study,
                    dict(raw.get("parameters") or {}),
                )
                signature = _candidate_signature(parameters)
                if signature in seen:
                    raise ValueError("candidate duplicates an existing or pending parameter set")
                hypothesis_id = raw.get("hypothesis_id")
                if hypothesis_id is not None and str(hypothesis_id) not in known_hypotheses:
                    raise ValueError(f"unknown hypothesis_id: {hypothesis_id}")
                confidence = raw.get("confidence")
                if confidence is not None and (
                    isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not isfinite(float(confidence))
                    or not 0 <= float(confidence) <= 1
                ):
                    raise ValueError("candidate confidence must be in [0, 1]")
                if study.controller_mode == "llm" and (
                    confidence is None
                    or float(confidence) < self.evidence_gate.min_confidence
                ):
                    raise ValueError(
                        "LLM candidate confidence must be at least "
                        f"{self.evidence_gate.min_confidence}"
                    )
                candidate_id = str(
                    raw.get("candidate_id") or f"candidate_{uuid4().hex[:10]}"
                )
                if candidate_id in known_candidate_ids:
                    raise ValueError(f"duplicate candidate_id: {candidate_id}")
                normalized = {
                    "candidate_id": candidate_id,
                    "parameters": parameters,
                    "proposal_id": proposal_id,
                    "hypothesis_id": str(hypothesis_id) if hypothesis_id is not None else None,
                    "role": str(raw.get("role") or "hypothesis_test"),
                    "rationale": str(raw.get("rationale") or "")[:1000],
                    "expected_signal": dict(raw.get("expected_signal") or {}),
                    "confidence": float(confidence) if confidence is not None else None,
                }
                seen.add(signature)
                known_candidate_ids.add(candidate_id)
                accepted.append(normalized)
            except Exception as exc:
                rejected.append({
                    "index": index,
                    "candidate_id": raw.get("candidate_id"),
                    "parameters": raw.get("parameters"),
                    "reason": str(exc),
                })
        study.pending_candidate_proposals.extend(accepted)
        review = {
            "trigger": trigger,
            "proposal_id": proposal_id,
            "accepted": accepted,
            "rejected": rejected,
            "created_at": datetime.now().isoformat(),
        }
        study.candidate_proposal_reviews.append(review)
        if persist:
            self._save_study(study)
        return review

    def has_future_candidate_capacity(self, study: HPOStudy) -> bool:
        if self.remaining_training_runs(study) <= 0:
            return False
        active_sampler = (
            study.candidate_strategy or study.sampler_strategy or study.strategy
        )
        exhausted = study.scheduler_state.get("candidate_generation_exhausted") or {}
        if (
            exhausted.get("sampler") == active_sampler
            and exhausted.get("search_phase") == max(len(study.search_phases) - 1, 0)
        ):
            return False
        trials = self.list_trials(study.experiment_id)
        if self._active_pruner(study) != "none":
            initial_limit = study.initial_trial_count or study.max_trials or 1
            initial_trials = [trial for trial in trials if trial.rung == 0]
            if len(initial_trials) >= initial_limit:
                return False

        # Remaining run quota is not sufficient evidence of candidate capacity.
        # An exhausted deterministic grid cannot be expanded by a review of the
        # active Study; advice at that point must be deferred to the next Study.
        # agent_proposal remains eligible because review is how its queue is filled.
        sampler = active_sampler
        if sampler != "grid_search":
            return True
        history = [*self.compatible_warm_start_trials(study), *trials]
        return bool(STRATEGIES.get("grid_search").suggest(
            study.search_space,
            1,
            existing=[trial.parameters for trial in history],
        ))

    def register_hypotheses(
        self,
        study: HPOStudy,
        hypotheses: List[Dict[str, Any]],
        *,
        proposal_id: Optional[str],
    ) -> Dict[str, Any]:
        existing_ids = {
            str(item.get("id")) for item in study.hypotheses
            if isinstance(item, dict) and item.get("id") is not None
        }
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        known_trial_ids = {
            trial.trial_id for trial in self.list_trials(study.experiment_id)
        }
        known_trial_ids.update(
            str(item.get("trial_id")) for item in study.warm_start_trials
            if isinstance(item, dict) and item.get("trial_id") is not None
        )
        for index, raw in enumerate(hypotheses or []):
            if not isinstance(raw, dict):
                rejected.append({"index": index, "reason": "hypothesis must be an object"})
                continue
            hypothesis_id = str(raw.get("id") or "").strip()
            claim = str(raw.get("claim") or "").strip()
            if not hypothesis_id or not claim:
                rejected.append({"index": index, "reason": "hypothesis requires id and claim"})
                continue
            if hypothesis_id in existing_ids:
                rejected.append({"index": index, "id": hypothesis_id, "reason": "duplicate hypothesis id"})
                continue
            evidence_trial_ids = raw.get("evidence_trial_ids") or []
            expected_signal = raw.get("expected_signal") or {}
            if not isinstance(evidence_trial_ids, list):
                rejected.append({
                    "index": index,
                    "id": hypothesis_id,
                    "reason": "evidence_trial_ids must be a list",
                })
                continue
            if not isinstance(expected_signal, dict):
                rejected.append({
                    "index": index,
                    "id": hypothesis_id,
                    "reason": "expected_signal must be an object",
                })
                continue
            normalized_evidence = [str(item) for item in evidence_trial_ids][:20]
            normalized = {
                "id": hypothesis_id,
                "claim": claim[:2000],
                "evidence_trial_ids": normalized_evidence,
                "unverified_evidence_trial_ids": [
                    item for item in normalized_evidence if item not in known_trial_ids
                ],
                "expected_signal": dict(expected_signal),
                "proposal_id": proposal_id,
                "status": "proposed",
            }
            existing_ids.add(hypothesis_id)
            study.hypotheses.append(normalized)
            accepted.append(normalized)
        return {"accepted": accepted, "rejected": rejected}

    def validate_candidate_parameters(
        self,
        study: HPOStudy,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply one shared deterministic validator to every generated candidate."""
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("candidate parameters must be a non-empty object")
        declared_names = {parameter.name for parameter in study.search_space.parameters}
        unknown = sorted(set(parameters) - declared_names)
        if unknown:
            raise ValueError(f"candidate contains unknown parameters: {', '.join(unknown)}")
        normalized: Dict[str, Any] = {}
        for parameter in _ordered_search_parameters(study.search_space):
            active = _condition_matches(parameter.condition, normalized)
            if not active:
                if parameter.name in parameters:
                    raise ValueError(
                        f"inactive conditional parameter must be omitted: {parameter.name}"
                    )
                continue
            if parameter.name not in parameters:
                raise ValueError(f"candidate is missing active parameter: {parameter.name}")
            normalized[parameter.name] = _validate_candidate_value(
                parameter,
                parameters[parameter.name],
            )
        if not _constraints_match(normalized, study.search_space.constraints):
            raise ValueError("candidate violates search-space constraints")
        if self.parameter_validator is not None:
            self.parameter_validator(dict(normalized))
        return normalized

    def _controller_mode(
        self, study: HPOStudy, proposal: Optional[StrategyProposal]
    ) -> str:
        mode = study.controller_mode or "auto"
        if mode == "auto":
            mode = "llm" if proposal is not None else "rule"
            study.controller_mode = mode
        if mode not in CONTROLLER_MODES - {"auto"}:
            raise ValueError(f"unsupported controller_mode: {mode}")
        return mode

    def _compatible_history_count_for_space(
        self, study: HPOStudy, search_space: Dict[str, Any]
    ) -> int:
        shadow = replace(study, search_space=search_space_from_dict(search_space))
        history = [
            *self.compatible_warm_start_trials(study),
            *[
                trial for trial in self.list_trials(study.experiment_id)
                if trial.rung == 0 and trial.status in {"completed", "promoted"}
            ],
        ]
        retained = 0
        for trial in history:
            try:
                self.validate_candidate_parameters(shadow, trial.parameters)
                retained += 1
            except ValueError:
                continue
        return retained

    def defer_strategy_review(
        self,
        study: HPOStudy,
        proposal: Optional[StrategyProposal] = None,
        *,
        trigger: str,
    ) -> Dict[str, Any]:
        """Record advice for the next Study without mutating a completed search phase."""
        trials = self.list_trials(study.experiment_id)
        analyzer = HPOFeedbackAnalyzer()
        feedback = analyzer.analyze(study, trials)
        mode = self._controller_mode(study, proposal)
        submitted = proposal
        controller_rejections: List[Dict[str, Any]] = []
        gate_record: Dict[str, Any] = {}
        if mode == "rule":
            if submitted is not None:
                controller_rejections.append({
                    "field": "proposal",
                    "reason": "rule controller ignores external/LLM proposals",
                })
            proposal = analyzer.propose(study, feedback, self.available_samplers())
        elif mode == "fixed":
            if submitted is not None:
                controller_rejections.append({
                    "field": "proposal",
                    "reason": "fixed controller does not permit policy changes",
                })
            proposal = StrategyProposal(
                action="keep_strategy", reason_codes=["fixed_controller"]
            )
        else:
            gate = self.evidence_gate.review(
                study,
                feedback,
                proposal,
                compatible_history_count=lambda value: self._compatible_history_count_for_space(
                    study, value
                ),
            )
            proposal = gate.proposal
            controller_rejections.extend(gate.rejected_fields)
            gate_record = gate.evidence
        stored = (
            proposal.to_dict()
            if proposal is not None and proposal.action in StrategyDecisionPolicy.ACTIONS
            else None
        )
        if mode != "fixed" and stored is not None:
            study.next_study_proposal = stored
        candidate_review = None
        if submitted is not None and submitted.candidate_proposals:
            candidate_review = {
                "trigger": trigger,
                "proposal_id": submitted.proposal_id,
                "accepted": [],
                "rejected": [
                    {
                        "index": index,
                        "candidate_id": item.get("candidate_id"),
                        "reason": "Study has no future candidate-generation capacity",
                    }
                    for index, item in enumerate(submitted.candidate_proposals)
                ],
                "created_at": datetime.now().isoformat(),
            }
            study.candidate_proposal_reviews.append(candidate_review)
        review = {
            "trigger": trigger,
            "scope": "next_study",
            "controller_mode": mode,
            "feedback": feedback,
            "proposal": submitted.to_dict() if submitted else None,
            "effective_proposal": stored,
            "evidence_gate": gate_record,
            "applied": False,
            "candidate_generation_applied": False,
            "effective_from_trial_index": None,
            "affected_trial_ids": [],
            "realized_outcome": {},
            "effect_estimate": {},
            "decision": {
                "decision": "deferred" if stored and mode != "fixed" else "not_applied",
                "scope": "next_study",
                "applied": False,
                "effective_from_trial_index": None,
                "affected_trial_ids": [],
                "effect_estimate": {},
                "accepted_fields": [],
                "rejected_fields": [
                    {
                        "field": "current_study",
                        "reason": "Study has no future candidate-generation capacity",
                    },
                    *controller_rejections,
                ],
                "reason_codes": [
                    "next_study_proposal_deferred"
                    if stored and mode != "fixed" else "fixed_controller"
                ],
                "proposal_id": submitted.proposal_id if submitted else None,
            },
            "candidate_review": candidate_review,
            "search_phase": len(study.search_phases) - 1,
            "created_at": datetime.now().isoformat(),
        }
        study.strategy_reviews.append(review)
        study.scheduler_state.update({
            "reviewed_trial_ids": sorted(
                trial.trial_id for trial in trials
                if trial.status in TERMINAL_TRIAL_STATUSES
            ),
            "completed_since_review": 0,
        })
        study.updated_at = datetime.now().isoformat()
        self._save_study(study)
        return review

    def review_strategy(
        self,
        study: HPOStudy,
        proposal: Optional[StrategyProposal] = None,
        *,
        trigger: str = "periodic",
    ) -> Dict[str, Any]:
        """Analyze feedback and safely update only subsequent candidate generation."""
        if not self.has_future_candidate_capacity(study):
            return self.defer_strategy_review(study, proposal, trigger=trigger)
        trials = self.list_trials(study.experiment_id)
        analyzer = HPOFeedbackAnalyzer()
        feedback = analyzer.analyze(study, trials)
        mode = self._controller_mode(study, proposal)
        submitted_proposal = proposal
        controller_rejections: List[Dict[str, Any]] = []
        gate_record: Dict[str, Any] = {}
        if mode == "fixed":
            if submitted_proposal is not None:
                controller_rejections.append({
                    "field": "proposal",
                    "reason": "fixed controller does not permit runtime policy changes",
                })
            proposal = StrategyProposal(
                action="keep_strategy", reason_codes=["fixed_controller"]
            )
        elif mode == "rule":
            if submitted_proposal is not None:
                controller_rejections.append({
                    "field": "proposal",
                    "reason": "rule controller ignores external/LLM proposals",
                })
            proposal = analyzer.propose(study, feedback, self.available_samplers())
        else:
            gate = self.evidence_gate.review(
                study,
                feedback,
                proposal,
                compatible_history_count=lambda value: self._compatible_history_count_for_space(
                    study, value
                ),
            )
            proposal = gate.proposal
            controller_rejections.extend(gate.rejected_fields)
            gate_record = gate.evidence

        gate_approved_proposal = proposal
        blocked_fields: List[Dict[str, Any]] = list(controller_rejections)
        legacy_pruner_request = (
            proposal.requested_strategy
            if proposal is not None and proposal.requested_strategy == "successive_halving"
            else None
        )
        runtime_blocked = {
            "max_training_runs": proposal.max_training_runs if proposal else None,
            "budgets": proposal.budgets if proposal else None,
            "requested_strategy": legacy_pruner_request,
            "requested_pruner": proposal.requested_pruner if proposal else None,
            "initial_trial_count": proposal.initial_trial_count if proposal else None,
            "promotion_limits": proposal.promotion_limits if proposal else None,
            "reduction_factor": proposal.reduction_factor if proposal else None,
        }
        if proposal is not None and any(value is not None for value in runtime_blocked.values()):
            for field_name, value in runtime_blocked.items():
                if value is not None:
                    blocked_fields.append({
                        "field": field_name,
                        "reason": f"runtime reviews cannot change active Study {field_name}",
                    })
            value = proposal.to_dict()
            for field_name, field_value in runtime_blocked.items():
                if field_value is not None:
                    value[field_name] = None
            proposal = StrategyProposal.from_dict(
                value, preserve_audit_fields=True
            )
        effective_proposal = proposal

        sampler_strategy, pruner_strategy = self._strategy_components(
            study.strategy, study.sampler_strategy, study.pruner_strategy
        )
        previous_sampler = study.candidate_strategy or sampler_strategy
        previous_sampler_config = dict(study.sampler_config)
        previous_search_space = study.search_space.to_dict()
        decision = OptimizationPlanDecisionPolicy().review(
            proposal,
            base_sampler=previous_sampler,
            base_pruner=pruner_strategy,
            base_search_space=study.search_space,
            base_budgets=study.budgets,
            hard_max_training_runs=int(study.max_training_runs or study.max_trials or 1),
            objectives=study.objectives,
            available_strategies=self.available_samplers(),
            validate_plan=self.validate_study_plan,
            base_initial_trial_count=study.initial_trial_count,
            base_promotion_limits=study.promotion_limits,
            base_reduction_factor=study.reduction_factor,
            confirmation_budget=study.budgets[-1],
        )
        decision.scope = "current_study"
        decision.effective_from_trial_index = len(trials)
        if blocked_fields:
            decision.proposal_id = submitted_proposal.proposal_id if submitted_proposal else None
            decision.proposal = submitted_proposal.to_dict() if submitted_proposal else None
            decision.rejected_fields.extend(blocked_fields)

        study.candidate_strategy = decision.adopted_sampler
        study.search_space = search_space_from_dict(decision.adopted_search_space)
        proposed_config = (
            proposal.sampler_config
            if proposal is not None and proposal.action in StrategyDecisionPolicy.ACTIONS
            else {}
        )
        try:
            if proposed_config or decision.adopted_sampler != previous_sampler:
                study.sampler_config = self.validate_sampler_config(
                    str(decision.adopted_sampler), proposed_config
                )
                if proposed_config:
                    decision.accepted_fields.append("sampler_config")
            else:
                study.sampler_config = previous_sampler_config
        except Exception as exc:
            study.sampler_config = (
                previous_sampler_config
                if decision.adopted_sampler == previous_sampler
                else self.validate_sampler_config(str(decision.adopted_sampler), {})
            )
            decision.rejected_fields.append({"field": "sampler_config", "reason": str(exc)})
        decision.adopted_sampler_config = dict(study.sampler_config)

        candidate_review = None
        hypothesis_review = None
        if proposal is not None and proposal.action in StrategyDecisionPolicy.ACTIONS:
            hypothesis_review = self.register_hypotheses(
                study, proposal.hypotheses, proposal_id=proposal.proposal_id
            )
            if previous_search_space != study.search_space.to_dict() and study.pending_candidate_proposals:
                pending = list(study.pending_candidate_proposals)
                study.pending_candidate_proposals = []
                self.enqueue_candidate_proposals(
                    study, pending, proposal_id=None,
                    trigger=f"{trigger}_pending_revalidation", persist=False,
                )
            candidate_review = self.enqueue_candidate_proposals(
                study, proposal.candidate_proposals,
                proposal_id=proposal.proposal_id, trigger=trigger, persist=False,
            )
            if hypothesis_review["accepted"]:
                decision.accepted_fields.append("hypotheses")
            if hypothesis_review["rejected"]:
                decision.rejected_fields.append({
                    "field": "hypotheses", "reason": hypothesis_review["rejected"]
                })
            if candidate_review["accepted"]:
                decision.accepted_fields.append("candidate_proposals")
            if candidate_review["rejected"]:
                decision.rejected_fields.append({
                    "field": "candidate_proposals", "reason": candidate_review["rejected"]
                })

        if study.candidate_strategy == "agent_proposal" and not study.pending_candidate_proposals:
            restored_sampler, restored_config = self._agent_fallback(
                study, previous_sampler, previous_sampler_config
            )
            study.candidate_strategy = restored_sampler
            study.sampler_config = restored_config
            decision.adopted_sampler = restored_sampler
            decision.adopted_strategy = restored_sampler
            decision.adopted_sampler_config = dict(restored_config)
            decision.accepted_fields = [
                field_name for field_name in decision.accepted_fields
                if field_name not in {"requested_sampler", "sampler_config"}
            ]
            decision.rejected_fields.append({
                "field": "requested_sampler",
                "reason": "agent_proposal requires at least one valid pending candidate; "
                f"restored {restored_sampler}",
            })
            decision.reason_codes.append("agent_queue_exhausted_fallback")

        decision.accepted_fields = list(dict.fromkeys(decision.accepted_fields))
        decision.applied = bool(decision.accepted_fields)
        if decision.rejected_fields:
            decision.decision = "approved_with_changes" if decision.applied else "rejected"
            decision.reason_codes = list(dict.fromkeys([
                *decision.reason_codes,
                "proposal_partially_approved" if decision.applied else "proposal_rejected",
            ]))
        elif decision.applied:
            decision.decision = "approved"
            decision.reason_codes = ["proposal_approved"]

        phase_changed = (
            study.candidate_strategy != previous_sampler
            or study.sampler_config != previous_sampler_config
            or study.search_space.to_dict() != previous_search_space
        )
        if phase_changed:
            study.scheduler_state.pop("candidate_generation_exhausted", None)
            if study.search_phases:
                study.search_phases[-1]["end_trial_index"] = len(trials) - 1
            study.search_phases.append({
                "phase_index": len(study.search_phases),
                "sampler": study.candidate_strategy,
                "sampler_config": dict(study.sampler_config),
                "start_trial_index": len(trials),
                "end_trial_index": None,
                "trigger": trigger,
                "proposal_id": proposal.proposal_id if proposal else None,
                "decision_id": decision.decision_id,
                "reason_codes": list(proposal.reason_codes) if proposal else [],
                "fallback_sampler": previous_sampler if study.candidate_strategy == "agent_proposal" else None,
                "fallback_sampler_config": previous_sampler_config if study.candidate_strategy == "agent_proposal" else {},
                "created_at": datetime.now().isoformat(),
            })
        generation_applied = bool(
            set(decision.accepted_fields)
            & {"requested_sampler", "requested_strategy", "sampler_config", "search_space", "candidate_proposals"}
        )
        if generation_applied:
            study.scheduler_state["active_decision_id"] = decision.decision_id
            study.scheduler_state["active_proposal_id"] = proposal.proposal_id if proposal else None
        review = {
            "trigger": trigger,
            "scope": "current_study",
            "controller_mode": mode,
            "feedback": feedback,
            "proposal": submitted_proposal.to_dict() if submitted_proposal else None,
            "gate_approved_proposal": (
                gate_approved_proposal.to_dict() if gate_approved_proposal else None
            ),
            "effective_proposal": effective_proposal.to_dict() if effective_proposal else None,
            "evidence_gate": gate_record,
            "decision": decision.to_dict(),
            "applied": decision.applied,
            "candidate_generation_applied": generation_applied,
            "effective_from_trial_index": decision.effective_from_trial_index,
            "affected_trial_ids": [],
            "realized_outcome": {},
            "effect_estimate": {},
            "applied_candidate_strategy": study.candidate_strategy,
            "applied_sampler_config": dict(study.sampler_config),
            "active_pruner_strategy": pruner_strategy,
            "applied_search_space": study.search_space.to_dict(),
            "hypothesis_review": hypothesis_review,
            "candidate_review": candidate_review,
            "pending_agent_candidate_count": len(study.pending_candidate_proposals),
            "search_phase": len(study.search_phases) - 1,
            "created_at": datetime.now().isoformat(),
        }
        study.strategy_reviews.append(review)
        study.scheduler_state.update({
            "reviewed_trial_ids": sorted(
                trial.trial_id for trial in trials if trial.status in TERMINAL_TRIAL_STATUSES
            ),
            "completed_since_review": 0,
        })
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
        merged_metrics = {**trial.metrics, **(metrics or {})}
        primary = study.objectives[0].metric
        if status in {"completed", "promoted"}:
            # Validate before mutating either the Trial or the Study's best value.
            require_finite_metric(merged_metrics, primary)
        elif primary in merged_metrics and not is_finite_metric(merged_metrics[primary]):
            trial.provenance["invalid_primary_metric"] = {
                "metric": primary, "value": repr(merged_metrics.pop(primary)),
            }
        trial.status = status
        trial.metrics = merged_metrics
        trial.intermediate_metrics = intermediate_metrics or trial.intermediate_metrics
        trial.cost.update(cost or {})
        trial.artifacts = _merge_trial_artifacts(trial.artifacts, artifacts or [])
        trial.stop_reason = stop_reason
        trial.updated_at = datetime.now().isoformat()
        self.save_trial(study.experiment_id, trial)
        self._refresh_study(study)
        return trial

    def retry_trial(self, study: HPOStudy, trial_id: str, reason: str) -> Trial:
        """Explicitly reopen a failed trial for a bounded scheduler retry."""
        if self.remaining_training_runs(study) <= 0:
            raise ValueError("max_training_runs exhausted")
        if not self.retry_budget_available(study):
            raise ValueError("retry would consume reserved initial/promotion training budget")
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

    def prepare_resume(self, study: HPOStudy) -> Dict[str, Any]:
        """Make a persisted Study safe to continue after its owning process stopped."""
        now = datetime.now().isoformat()
        previous_status = study.status
        promotion_repairs = self._reconcile_promotions(study)
        persisted_trials = self.list_trials(study.experiment_id)
        persisted_ids = {trial.trial_id for trial in persisted_trials}
        known_ids = [trial_id for trial_id in study.trial_ids if trial_id in persisted_ids]
        known_id_set = set(known_ids)
        orphan_trials = sorted(
            (
                trial for trial in persisted_trials
                if trial.trial_id not in known_id_set
            ),
            key=lambda trial: (trial.created_at, trial.trial_id),
        )
        reconciled_trial_ids = [trial.trial_id for trial in orphan_trials]
        study.trial_ids = known_ids + reconciled_trial_ids
        existing_signatures = {
            _candidate_signature(trial.parameters) for trial in persisted_trials
        }
        pending_before = len(study.pending_candidate_proposals)
        study.pending_candidate_proposals = [
            proposal for proposal in study.pending_candidate_proposals
            if isinstance(proposal, dict)
            and _candidate_signature(proposal.get("parameters") or {})
            not in existing_signatures
        ]
        removed_pending_candidate_count = (
            pending_before - len(study.pending_candidate_proposals)
        )
        self._refresh_study(study)
        if (
            previous_status == "completed"
            and not self.completion_errors(study)
            and not promotion_repairs
            and not any(t.status in {"running", "suggested"} for t in persisted_trials)
            and not (
                study.pending_candidate_proposals
                and self.has_future_candidate_capacity(study)
            )
        ):
            return {
                "resumed": False,
                "reason": "study_already_completed",
                "recovered_trial_ids": [],
                "created_at": now,
            }
        recovered_trial_ids: List[str] = []
        for trial in persisted_trials:
            if trial.status != "running":
                continue
            trial.status = "suggested"
            trial.stop_reason = None
            trial.cost["resume_count"] = int(trial.cost.get("resume_count", 0)) + 1
            events = list(trial.provenance.get("resume_events") or [])
            events.append({
                "reason": "owning_process_interrupted",
                "recovered_at": now,
            })
            trial.provenance["resume_events"] = events[-20:]
            trial.updated_at = now
            self.save_trial(study.experiment_id, trial)
            recovered_trial_ids.append(trial.trial_id)

        resume_count = int(study.scheduler_state.get("resume_count", 0)) + 1
        resume_event = {
            "resume_count": resume_count,
            "previous_status": previous_status,
            "recovered_trial_ids": recovered_trial_ids,
            "reconciled_trial_ids": reconciled_trial_ids,
            "promotion_repairs": promotion_repairs,
            "removed_pending_candidate_count": removed_pending_candidate_count,
            "created_at": now,
        }
        events = list(study.scheduler_state.get("resume_events") or [])
        events.append(resume_event)
        study.scheduler_state.update({
            "resume_count": resume_count,
            "resume_events": events[-20:],
            "last_resume_at": now,
            "current_trial_id": None,
            "terminal": False,
            "terminal_status": None,
            "completed_since_review": self.unreviewed_trial_count(study),
        })
        study.status = "running"
        study.stop_reason = None
        study.updated_at = now
        self._refresh_study(study)
        return {"resumed": True, **resume_event}

    def unreviewed_trial_count(self, study: HPOStudy) -> int:
        terminal_ids = {
            trial.trial_id for trial in self.list_trials(study.experiment_id)
            if trial.status in TERMINAL_TRIAL_STATUSES
        }
        reviewed_ids = study.scheduler_state.get("reviewed_trial_ids")
        if reviewed_ids is not None:
            return len(terminal_ids - set(reviewed_ids))
        # Legacy reviews contain counts but no identity watermark.
        previous_count = (
            int((study.strategy_reviews[-1].get("feedback") or {}).get("terminal_trials", 0))
            if study.strategy_reviews else 0
        )
        return max(len(terminal_ids) - previous_count, 0)

    def _reconcile_promotions(self, study: HPOStudy) -> List[Dict[str, Any]]:
        """Use durable child records as promotion commits; repair legacy gaps."""
        trials = self.list_trials(study.experiment_id)
        by_id = {trial.trial_id: trial for trial in trials}
        children = {}
        for child in trials:
            if child.parent_trial_id is None:
                continue
            parent = by_id.get(child.parent_trial_id)
            if parent is None or child.rung != parent.rung + 1:
                raise ValueError(f"invalid promotion parent for {child.trial_id}")
            key = (parent.trial_id, child.rung)
            if key in children:
                raise ValueError(f"duplicate promotion children for {parent.trial_id}")
            children[key] = child
        repairs = []
        for parent in trials:
            child = children.get((parent.trial_id, parent.rung + 1))
            next_status = parent.status
            if child is not None and parent.status == "completed":
                next_status = "promoted"
            elif child is None and parent.status == "promoted":
                # Old versions wrote the parent before creating the child.
                next_status = "completed"
            if next_status != parent.status:
                if study.status == "completed":
                    # Reopen before repairing Trial files. Even a second crash
                    # during repair must not leave a falsely completed Study.
                    study.status = "running"
                    study.stop_reason = None
                    study.scheduler_state.update({"terminal": False, "terminal_status": None})
                    self._save_study(study)
                repairs.append({
                    "trial_id": parent.trial_id,
                    "previous_status": parent.status,
                    "status": next_status,
                    "child_trial_id": child.trial_id if child else None,
                })
                parent.status = next_status
                parent.updated_at = datetime.now().isoformat()
                self.save_trial(study.experiment_id, parent)
        return repairs

    def update_scheduler_state(self, study: HPOStudy, **updates: Any) -> None:
        """Persist minimal graph progress needed to reconstruct a scheduler run."""
        study.scheduler_state.update(updates)
        study.scheduler_state["updated_at"] = datetime.now().isoformat()
        study.updated_at = study.scheduler_state["updated_at"]
        self._save_study(study)

    def finish_study(self, study: HPOStudy, status: str, stop_reason: Optional[str] = None) -> HPOStudy:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"unsupported study terminal status: {status}")
        study.status = status
        study.stop_reason = stop_reason
        study.scheduler_state.update({
            "current_trial_id": None,
            "terminal": True,
            "terminal_status": status,
        })
        if study.search_phases:
            study.search_phases[-1]["end_trial_index"] = (
                len(self.list_trials(study.experiment_id)) - 1
            )
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
            and is_finite_metric(trial.metrics.get(objective.metric))
        ]
        errors: List[str] = []
        if not completed:
            errors.append("no completed trial with a valid primary metric")
        active = [trial.trial_id for trial in trials if trial.status in {"suggested", "running"}]
        if active:
            errors.append(f"trials without terminal status: {', '.join(active)}")
        if study.best_trial_id is None:
            errors.append("best_trial_id is missing")
        elif study.best_trial_id not in {trial.trial_id for trial in completed}:
            errors.append("best_trial_id has no valid primary metric")
        required_rung = len(self.planned_rung_counts(study)) - 1
        if required_rung > 0 and not any(trial.rung >= required_rung for trial in completed):
            errors.append(
                f"required confirmation rung {required_rung} has no valid completed trial; "
                "optimization is incomplete (budget exhausted or no eligible promotion)"
            )
        return errors

    def complete_study(self, study: HPOStudy, stop_reason: Optional[str] = None) -> HPOStudy:
        errors = self.completion_errors(study)
        if errors:
            raise ValueError("; ".join(errors))
        return self.finish_study(study, "completed", stop_reason)

    def promote_trials(self, study: HPOStudy) -> List[Trial]:
        # Also reconcile when a caller retries promotion without restarting the
        # full scheduler. Never create another child for a committed parent.
        if self._reconcile_promotions(study):
            self._refresh_study(study)
        objective = study.objectives[0]
        trials = self.list_trials(study.experiment_id)
        completed_by_rung: Dict[int, List[Trial]] = {}
        active_by_rung: Dict[int, List[Trial]] = {}
        for trial in trials:
            if trial.status == "completed":
                completed_by_rung.setdefault(trial.rung, []).append(trial)
            elif trial.status in {"suggested", "running"}:
                active_by_rung.setdefault(trial.rung, []).append(trial)
        eligible_rungs = []
        for rung in completed_by_rung:
            cohort_size = sum(
                trial.rung == rung and trial.status in {"completed", "promoted"}
                and is_finite_metric(trial.metrics.get(objective.metric))
                for trial in trials
            )
            if rung + 1 >= len(study.budgets) or cohort_size < study.min_completed_per_rung:
                continue
            if active_by_rung.get(rung):
                continue
            limit = study.promotion_limits[rung] if rung < len(study.promotion_limits) else None
            destination_count = len([trial for trial in trials if trial.rung == rung + 1])
            remaining_limit = max(limit - destination_count, 0) if limit is not None else None
            if self.halving_strategy.promote(
                trials, objective, study.reduction_factor, rung=rung, limit=remaining_limit,
            ):
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
                trial_id=f"trial_{uuid5(NAMESPACE_URL, f'{study.study_id}:{candidate.trial_id}:{next_rung}').hex}",
                parameters=dict(candidate.parameters),
                budget=study.budgets[next_rung],
                parent_trial_id=candidate.trial_id,
                rung=next_rung,
                candidate_source="promotion",
                search_phase=candidate.search_phase,
                proposal_id=candidate.proposal_id,
                hypothesis_id=candidate.hypothesis_id,
                provenance={
                    **dict(candidate.provenance),
                    "promoted_from": candidate.trial_id,
                    "original_candidate_source": candidate.candidate_source,
                },
                created_at=now,
                updated_at=now,
            )
            # The child is the durable promotion commit. If the next write is
            # interrupted, resume infers the parent's state from this child.
            self.save_trial(study.experiment_id, trial)
            candidate.status = "promoted"
            candidate.updated_at = now
            self.save_trial(study.experiment_id, candidate)
            study.trial_ids.append(trial.trial_id)
            promoted.append(trial)
        self._save_study(study)
        return promoted

    def planned_rung_counts(self, study: HPOStudy) -> List[int]:
        """Expected reachable allocations; explicit zero promotion ends the plan."""
        initial = int(study.initial_trial_count or study.max_trials or 1)
        counts = [initial]
        if self._active_pruner(study) == "none":
            return counts
        for rung in range(len(study.budgets) - 1):
            count = max(1, ceil(counts[-1] / study.reduction_factor))
            if rung < len(study.promotion_limits):
                count = min(count, study.promotion_limits[rung])
            if count <= 0:
                break
            counts.append(count)
        return counts

    def reserved_training_runs(self, study: HPOStudy) -> int:
        """Protect uncreated initial candidates and promotions from retries.

        Already-created Trials have consumed a slot in training_runs_used even
        if still suggested/running; do not reserve those slots a second time.
        """
        if self._active_pruner(study) == "none":
            return 0
        trials = self.list_trials(study.experiment_id)
        return sum(
            max(count - sum(trial.rung == rung for trial in trials), 0)
            for rung, count in enumerate(self.planned_rung_counts(study))
        )

    def retry_budget_available(self, study: HPOStudy) -> bool:
        return self.remaining_training_runs(study) > self.reserved_training_runs(study)

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
        self._validate_record_id(experiment_id, "experiment_id")
        data = json.loads(self._study_path(experiment_id).read_text(encoding="utf-8"))
        return study_from_dict(data)

    def list_trials(self, experiment_id: str) -> List[Trial]:
        self._validate_record_id(experiment_id, "experiment_id")
        trial_dir = self._trial_dir(experiment_id)
        if not trial_dir.exists():
            return []
        return [
            trial_from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(trial_dir.glob("trial_*.json"))
        ]

    def load_trial(self, experiment_id: str, trial_id: str) -> Trial:
        self._validate_record_id(experiment_id, "experiment_id")
        self._validate_record_id(trial_id, "trial_id")
        path = self._trial_dir(experiment_id) / f"{trial_id}.json"
        return trial_from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_trial(self, experiment_id: str, trial: Trial) -> None:
        self._validate_record_id(experiment_id, "experiment_id")
        self._validate_record_id(trial.trial_id, "trial_id")
        path = self._trial_dir(experiment_id, create=True) / f"{trial.trial_id}.json"
        _write_json_atomic(path, trial.to_dict())

    def best_metric_value(self, study: HPOStudy, exclude_trial_id: Optional[str] = None) -> Optional[float]:
        objective = study.objectives[0]
        values = [
            trial.metrics[objective.metric]
            for trial in self.list_trials(study.experiment_id)
            if trial.trial_id != exclude_trial_id
            and trial.status in {"completed", "promoted"}
            and is_finite_metric(trial.metrics.get(objective.metric))
        ]
        if not values:
            return None
        return min(values) if objective.mode == "min" else max(values)

    def _refresh_study(self, study: HPOStudy) -> None:
        objective = study.objectives[0]
        trials = self.list_trials(study.experiment_id)
        persisted_ids = {trial.trial_id for trial in trials}
        study.trial_ids = list(dict.fromkeys(
            [trial_id for trial_id in study.trial_ids if trial_id in persisted_ids]
            + [trial.trial_id for trial in sorted(trials, key=lambda t: (t.created_at, t.trial_id))]
        ))
        for hypothesis in study.hypotheses:
            hypothesis_id = str(hypothesis.get("id") or "")
            related = [
                trial for trial in trials
                if str(trial.hypothesis_id or "") == hypothesis_id
            ]
            completed = [
                trial for trial in related
                if trial.status in {"completed", "promoted"}
                and is_finite_metric(trial.metrics.get(objective.metric))
            ]
            if completed:
                values = [float(trial.metrics[objective.metric]) for trial in completed]
                hypothesis["status"] = "evaluated"
                hypothesis["result"] = {
                    "trial_ids": [trial.trial_id for trial in completed],
                    "best_metric": min(values) if objective.mode == "min" else max(values),
                    "metric": objective.metric,
                }
            elif any(trial.status in {"suggested", "running"} for trial in related):
                hypothesis["status"] = "testing"
                hypothesis.pop("result", None)
            elif related:
                hypothesis["status"] = "inconclusive"
                hypothesis.pop("result", None)
            else:
                hypothesis["status"] = "proposed"
                hypothesis.pop("result", None)
        candidates = [
            trial for trial in trials
            if trial.status in {"completed", "promoted"}
            and is_finite_metric(trial.metrics.get(objective.metric))
        ]
        study.best_trial_id = None
        if candidates:
            if self._active_pruner(study) == "successive_halving":
                # Metrics from different fidelity levels are not directly
                # comparable. Prefer the highest rung that has a valid result.
                highest_rung = max(trial.rung for trial in candidates)
                candidates = [trial for trial in candidates if trial.rung == highest_rung]
            reverse = objective.mode == "max"
            candidates.sort(key=lambda trial: trial.metrics[objective.metric], reverse=reverse)
            study.best_trial_id = candidates[0].trial_id
        self._refresh_strategy_review_outcomes(study, trials, objective)
        required_rung = len(self.planned_rung_counts(study)) - 1
        highest_rung = max((trial.rung for trial in candidates), default=None)
        study.scheduler_state["completion"] = {
            "required_rung": required_rung,
            "highest_completed_rung": highest_rung,
            "confirmation_satisfied": highest_rung is not None and highest_rung >= required_rung,
        }
        study.updated_at = datetime.now().isoformat()
        self._save_study(study)

    @staticmethod
    def _refresh_strategy_review_outcomes(
        study: HPOStudy, trials: List[Trial], objective: Objective
    ) -> None:
        for review in study.strategy_reviews:
            decision = review.get("decision") or {}
            decision_id = decision.get("decision_id")
            if review.get("scope") != "current_study" or not decision_id:
                continue
            summary = summarize_decision_effect(
                trials,
                str(decision_id),
                objective,
                decision_created_at=review.get("created_at"),
            )
            review.update(summary)
            decision["affected_trial_ids"] = summary["affected_trial_ids"]
            decision["realized_outcome"] = summary["realized_outcome"]
            decision["effect_estimate"] = summary["effect_estimate"]

    def _save_study(self, study: HPOStudy) -> None:
        path = self._study_path(study.experiment_id, create=True)
        _write_json_atomic(path, study.to_dict())
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
        if not is_finite_metric(best_metric):
            best_metric = None
        self.tracker.update_hpo_experiment(
            study.experiment_id,
            extensions={"optimization": {
                "study": {
                    "study_id": study.study_id,
                    "experiment_id": study.experiment_id,
                    "status": study.status,
                    "strategy": study.strategy,
                    "sampler_strategy": study.sampler_strategy,
                    "candidate_strategy": study.candidate_strategy,
                    "controller_mode": study.controller_mode,
                    "sampler_config": dict(study.sampler_config),
                    "search_phases": list(study.search_phases),
                    "best_trial_id": study.best_trial_id,
                    "best_metric": best_metric,
                    "objective": objective.to_dict() if objective else None,
                    "metric_protocol": study.history_context.get(
                        "metric_protocol"
                    ),
                    "metric_units": study.history_context.get("metric_units"),
                    "validation_pairs_sha256": study.history_context.get(
                        "validation_pairs_sha256"
                    ),
                    "trial_ids": list(study.trial_ids),
                    "max_training_runs": study.max_training_runs,
                    "completion": dict(study.scheduler_state.get("completion") or {}),
                    "next_study_proposal": study.next_study_proposal,
                    "updated_at": study.updated_at,
                    "study_artifact": str(path),
                },
                "trial_summary": [
                    {
                        "trial_id": trial.trial_id,
                        "status": trial.status,
                        "phase": _trial_phase(trial),
                        "rung": trial.rung,
                        "candidate_source": trial.candidate_source,
                        "search_phase": trial.search_phase,
                        "proposal_id": trial.proposal_id,
                        "hypothesis_id": trial.hypothesis_id,
                        "provenance": trial.provenance,
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
            metrics={
                "best": {
                    "trial_id": study.best_trial_id,
                    "primary_metric": objective.metric,
                    "primary_value": best_metric,
                    "primary_mode": objective.mode,
                    objective.metric: best_metric,
                }
                if objective else {}
            },
            artifacts=[{
                "type": "hpo_study",
                "name": study.study_id,
                "path": str(path),
            }],
        )

    @staticmethod
    def _validate_record_id(value: str, field: str) -> None:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError(f"invalid {field}")

    def _study_artifact_dir(self, experiment_id: str, *, create: bool = False) -> Path:
        self._validate_record_id(experiment_id, "experiment_id")
        root = self.tracker.experiments_dir.resolve()
        path = (root / experiment_id / "hpo_study").resolve()
        if root != path and root not in path.parents:
            raise ValueError("HPO artifact path escapes the experiment store")
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _study_path(self, experiment_id: str, *, create: bool = False) -> Path:
        return self._study_artifact_dir(experiment_id, create=create) / "study.json"

    def _trial_dir(self, experiment_id: str, *, create: bool = False) -> Path:
        path = self._study_artifact_dir(experiment_id, create=create) / "trials"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path



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


def _merge_trial_artifacts(
    existing: List[Dict[str, Any]],
    updates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Append Trial artifacts while preserving first-seen order and uniqueness."""
    merged: List[Dict[str, Any]] = []
    seen = set()
    for artifact in [*existing, *updates]:
        key = (
            artifact.get("type"),
            artifact.get("name"),
            artifact.get("path"),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(artifact)
    return merged


def _write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _candidate_signature(parameters: Dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, ensure_ascii=False, default=str)


def _condition_matches(condition: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    return not condition or all(candidate.get(key) == value for key, value in condition.items())


def _ordered_search_parameters(search_space: SearchSpace) -> List[SearchParameter]:
    """Resolve conditional parameters without relying on declaration order."""
    remaining = list(search_space.parameters)
    resolved = set()
    ordered: List[SearchParameter] = []
    while remaining:
        ready = [
            parameter for parameter in remaining
            if set(parameter.condition).issubset(resolved)
        ]
        if not ready:
            cycle = ", ".join(parameter.name for parameter in remaining)
            raise ValueError(f"cyclic or unknown search-space conditions: {cycle}")
        for parameter in ready:
            remaining.remove(parameter)
            resolved.add(parameter.name)
            ordered.append(parameter)
    return ordered


def _constraints_match(candidate: Dict[str, Any], constraints: List[Dict[str, Any]]) -> bool:
    for constraint in constraints:
        parameter = constraint.get("parameter")
        operator = constraint.get("operator")
        expected = constraint.get("value")
        current = candidate.get(parameter)
        if current is None:
            continue
        try:
            if operator == "lte":
                matches = current <= expected
            elif operator == "gte":
                matches = current >= expected
            elif operator == "eq":
                matches = current == expected
            elif operator == "in":
                matches = current in expected
            else:
                return False
            if not matches:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _validate_candidate_value(parameter: SearchParameter, value: Any) -> Any:
    if parameter.parameter_type == "categorical":
        if value not in parameter.choices:
            raise ValueError(
                f"candidate value for {parameter.name} is not one of the allowed choices"
            )
        return value
    if parameter.parameter_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"candidate value for {parameter.name} must be an integer")
        normalized: Any = int(value)
    elif parameter.parameter_type == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"candidate value for {parameter.name} must be numeric")
        normalized = float(value)
    else:
        raise ValueError(f"unsupported parameter type: {parameter.parameter_type}")
    if parameter.low is None or parameter.high is None:
        raise ValueError(f"low/high are required for {parameter.name}")
    if normalized < parameter.low or normalized > parameter.high:
        raise ValueError(
            f"candidate value for {parameter.name} is outside [{parameter.low}, {parameter.high}]"
        )
    return normalized


def search_space_from_dict(data: Dict[str, Any]) -> SearchSpace:
    return SearchSpace(
        parameters=[SearchParameter(**item) for item in data.get("parameters") or []],
        constraints=data.get("constraints") or [],
    )


def study_from_dict(data: Dict[str, Any]) -> HPOStudy:
    study = HPOStudy(
        study_id=data["study_id"],
        experiment_id=data["experiment_id"],
        strategy=data["strategy"],
        search_space=search_space_from_dict(data["search_space"]),
        objectives=[Objective(**item) for item in data.get("objectives") or []],
        budgets=[TrialBudget(**item) for item in data.get("budgets") or []],
        sampler_strategy=data.get("sampler_strategy"),
        pruner_strategy=data.get("pruner_strategy"),
        candidate_strategy=data.get("candidate_strategy"),
        controller_mode=data.get("controller_mode", "auto"),
        reduction_factor=data.get("reduction_factor", 3),
        max_trials=data.get("max_trials"),
        initial_trial_count=data.get("initial_trial_count"),
        promotion_limits=data.get("promotion_limits") or [],
        max_training_runs=data.get("max_training_runs"),
        min_completed_per_rung=data.get("min_completed_per_rung", 1),
        candidate_batch_size=data.get("candidate_batch_size"),
        sampler_config=data.get("sampler_config") or {},
        search_phases=data.get("search_phases") or [],
        hypotheses=data.get("hypotheses") or [],
        pending_candidate_proposals=data.get("pending_candidate_proposals") or [],
        candidate_proposal_reviews=data.get("candidate_proposal_reviews") or [],
        candidate_generation_reviews=data.get("candidate_generation_reviews") or [],
        scheduler_state=data.get("scheduler_state") or {},
        constraints=data.get("constraints") or [],
        strategy_reviews=data.get("strategy_reviews") or [],
        next_study_proposal=data.get("next_study_proposal"),
        warm_start_trials=data.get("warm_start_trials") or [],
        history_context=data.get("history_context") or {},
        trial_ids=data.get("trial_ids") or [],
        best_trial_id=data.get("best_trial_id"),
        status=data.get("status", "created"),
        stop_reason=data.get("stop_reason"),
        random_seed=data.get("random_seed", 0),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )
    if not study.search_phases:
        sampler = (
            study.candidate_strategy
            or study.sampler_strategy
            or ("random_search" if study.strategy == "successive_halving" else study.strategy)
        )
        study.search_phases = [{
            "phase_index": 0,
            "sampler": sampler,
            "sampler_config": dict(study.sampler_config),
            "start_trial_index": 0,
            "end_trial_index": None,
            "trigger": "legacy_study_migration",
            "proposal_id": None,
            "decision_id": None,
            "reason_codes": [],
            "fallback_sampler": None,
            "fallback_sampler_config": {},
            "created_at": study.created_at,
        }]
    return study


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
        candidate_source=str(data.get("candidate_source") or "optimizer"),
        search_phase=int(data.get("search_phase") or 0),
        proposal_id=(
            str(data["proposal_id"]) if data.get("proposal_id") is not None else None
        ),
        hypothesis_id=(
            str(data["hypothesis_id"]) if data.get("hypothesis_id") is not None else None
        ),
        provenance=data.get("provenance") or {},
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )
