"""LangGraph HPO agent with a single structured task interface."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.agents.base_agent import LangGraphAgent
from agent.agents.communication import AgentTaskRequest, AgentTaskResult
from agent.hpo import (
    HPOFeedbackAnalyzer,
    HPOPlanningPolicy,
    HPOScheduler,
    HPOService,
    Objective,
    OptimizationCampaign,
    OptimizationPlanDecisionPolicy,
    CampaignPolicy,
    RetryPolicy,
    SearchParameter,
    SearchSpace,
    StrategyProposal,
    TrialBudget,
)
from agent.data_processing.handoff import resolve_data_handoff
from agent.memory import EpisodeMemory, MemoryQuery, MemoryScope, MemoryService
from agent.models import get_model_adapter
from agent.prompt import render_prompt
from agent.utils import ConfigParser, ExperimentTracker
from agent.utils.path_tool import (
    get_config_file,
    get_hpo_experiments_dir,
    resolve_config_path,
    resolve_config_value_path,
)


class HPOAgent(LangGraphAgent):
    """Execute validated HPO workflows; the LLM may only submit structured proposals."""

    action = "optimize_hyperparameters"

    def __init__(
        self,
        model_name: str = "GLM-4.7",
        temperature: float = 0.2,
        max_iterations: int = 10,
        verbose: bool = True,
        config_path: str = str(get_config_file("train_ecapa_tdnn.yaml")),
        experiments_dir: Optional[str] = None,
        task_type: str = "speaker_verification",
        model_family: str = "ecapa_tdnn",
        implementation: str = "speechbrain",
        runner: str = "speechbrain",
        enable_llm_advisor: bool = False,
        planning_policy: Optional[HPOPlanningPolicy] = None,
        decision_policy: Optional[OptimizationPlanDecisionPolicy] = None,
    ) -> None:
        super().__init__(model_name, temperature, max_iterations, verbose)
        self.config_path = str(resolve_config_path(config_path))
        self.experiments_dir = Path(experiments_dir).resolve() if experiments_dir else get_hpo_experiments_dir()
        self.task_type = task_type
        self.model_family = model_family
        self.implementation = implementation
        self.runner = runner
        self.enable_llm_advisor = enable_llm_advisor
        self.planning_policy = planning_policy or HPOPlanningPolicy()
        self.decision_policy = decision_policy or OptimizationPlanDecisionPolicy(self.planning_policy)
        self.memory_service = MemoryService()
        self.memory_scope = MemoryScope(
            agent_type="hpo_agent",
            task_type=task_type,
            model_family=model_family,
            tags=["optimization", "langgraph"],
        )

    def run_workflow(self, request: AgentTaskRequest) -> AgentTaskResult:
        started_at = datetime.now()
        tracker = ExperimentTracker(self.experiments_dir)
        config = ConfigParser(self.config_path).load_config(resolve_references=True)
        data_handoff = resolve_data_handoff(request.context, config.get("data_folder"))
        data_folder = data_handoff["consumer_uri"]
        output = resolve_config_value_path(config.get("output_folder"))
        self.memory_scope.dataset_key = data_folder

        search_space = self._resolve_search_space(request.context.get("search_space"))
        per_study_limit = int(request.budget.get("max_training_runs") or self.max_iterations)
        requested_strategy = str(request.context.get("strategy") or "auto")
        requested_sampler = request.context.get("sampler")
        requested_pruner = request.context.get("pruner")
        budgets = self._build_budgets(
            request.context.get("budgets"),
            requested_pruner or requested_strategy,
        )
        self._validate_search_budget_compatibility(search_space, budgets)
        service = HPOService(tracker)
        base_sampler, base_pruner = self.planning_policy.select_components(
            requested_strategy,
            str(requested_sampler) if requested_sampler is not None else None,
            str(requested_pruner) if requested_pruner is not None else None,
            search_space,
            budgets,
            per_study_limit,
            service.available_samplers(),
        )
        if base_pruner == "none" and len(budgets) > 1:
            budgets = [budgets[-1]]
        objectives = [Objective(
            str(request.context.get("primary_metric", "eer")),
            str(request.context.get("metric_mode", "min")),
        )]
        max_studies = max(int(request.budget.get("max_studies", 1)), 1)
        target_value = request.context.get("target_value", request.budget.get("target_value"))
        campaign = OptimizationCampaign(
            objective=objectives[0],
            target_value=float(target_value) if target_value is not None else None,
            max_studies=max_studies,
            patience=max(int(request.budget.get("campaign_patience", max_studies)), 1),
            min_improvement=float(request.budget.get("campaign_min_improvement", 0.0)),
            max_total_training_runs=int(request.budget.get("max_total_training_runs") or per_study_limit * max_studies),
        )
        campaign_policy = CampaignPolicy()
        study_results: List[Dict[str, Any]] = []
        best_trial = None
        best_experiment_id = None
        sampler = base_sampler
        pruner = base_pruner
        reduction_factor = int(request.budget.get("reduction_factor", 3))
        campaign_history: List[Dict[str, Any]] = []
        for study_index in range(max_studies):
            remaining = campaign_policy.remaining_runs(campaign)
            run_limit = min(per_study_limit, remaining) if remaining is not None else per_study_limit
            if run_limit <= 0:
                campaign.status, campaign.stop_reason = "completed", "max_total_training_runs_reached"
                break
            base_initial_count = min(
                int(
                    request.budget.get("initial_trial_count")
                    or self._default_initial_trial_count(
                        "successive_halving" if pruner == "successive_halving" else sampler,
                        budgets,
                        run_limit,
                    )
                ),
                run_limit,
            )
            base_promotions = (
                list(request.budget.get("promotion_limits") or self._default_promotion_limits(
                    base_initial_count,
                    budgets,
                    reduction_factor,
                ))
                if pruner == "successive_halving"
                else []
            )
            proposal = (
                self._planning_proposal(
                    request,
                    search_space,
                    budgets,
                    sampler,
                    pruner,
                    run_limit,
                    base_initial_count,
                    base_promotions,
                    reduction_factor,
                    campaign.to_dict(),
                )
                if self.enable_llm_advisor else None
            )
            decision = self.decision_policy.review(
                proposal,
                base_sampler=sampler,
                base_pruner=pruner,
                base_search_space=search_space,
                base_budgets=budgets,
                hard_max_training_runs=run_limit,
                objectives=objectives,
                available_strategies=service.available_samplers(),
                validate_plan=service.validate_study_plan,
                base_initial_trial_count=base_initial_count,
                base_promotion_limits=base_promotions,
                base_reduction_factor=reduction_factor,
            )
            sampler = str(decision.adopted_sampler or sampler)
            pruner = str(decision.adopted_pruner or pruner)
            search_space = self._resolve_search_space(decision.adopted_search_space)
            budgets = self._build_budgets(decision.adopted_budgets, pruner)
            if pruner == "none" and len(budgets) > 1:
                budgets = [budgets[-1]]
            self._validate_search_budget_compatibility(search_space, budgets)
            run_limit = decision.adopted_max_training_runs
            reduction_factor = decision.adopted_reduction_factor
            initial_count = min(
                int(decision.adopted_initial_trial_count or run_limit),
                run_limit,
            )
            promotion_limits = (
                list(decision.adopted_promotion_limits)
                if pruner == "successive_halving"
                else []
            )
            if initial_count + sum(promotion_limits) > run_limit:
                initial_count = self._default_initial_trial_count(
                    "successive_halving" if pruner == "successive_halving" else sampler,
                    budgets,
                    run_limit,
                )
                promotion_limits = (
                    self._default_promotion_limits(initial_count, budgets, reduction_factor)
                    if pruner == "successive_halving"
                    else []
                )
            decision.adopted_initial_trial_count = initial_count
            decision.adopted_promotion_limits = promotion_limits
            legacy_strategy = (
                "successive_halving"
                if pruner == "successive_halving"
                else sampler
            )
            resource_profile = self._resource_snapshot(
                request.context.get("runtime_options"),
                request.context.get("resource_profile"),
            )
            search_budget_analysis = self._search_budget_analysis(
                search_space,
                budgets,
                run_limit,
                initial_count,
                promotion_limits,
                reduction_factor,
                campaign.to_dict(),
            )
            adopted_plan = {
                "strategy": legacy_strategy,
                "sampler": sampler,
                "pruner": pruner,
                "search_space": search_space.to_dict(),
                "budgets": [budget.to_dict() for budget in budgets],
                "max_training_runs": run_limit,
                "initial_trial_count": initial_count,
                "promotion_limits": promotion_limits,
                "reduction_factor": reduction_factor,
                "warm_start_trial_count": len(campaign_history),
                "resource_profile": resource_profile,
                "search_budget_analysis": search_budget_analysis,
            }
            experiment_id = tracker.create_hpo_experiment(
                config_path=self.config_path,
                data_folder=data_folder,
                output_folder=str(output) if output else None,
                description=f"Optimization campaign {campaign.campaign_id} study {study_index + 1}",
                task={
                    "type": self.task_type,
                    "dataset": data_folder,
                    "dataset_id": data_handoff.get("dataset_id"),
                    "dataset_version": data_handoff.get("dataset_version"),
                    "data_processing_experiment_id": data_handoff.get("data_processing_experiment_id"),
                    "primary_metric": objectives[0].metric,
                    "metric_mode": objectives[0].mode,
                },
                model={"family": self.model_family, "implementation": self.implementation, "config_path": self.config_path},
                execution={"runner": self.runner, "output_folder": str(output) if output else None},
                extra_fields={"version": {
                    "campaign_id": campaign.campaign_id,
                    "study_index": study_index + 1,
                    "model_key": f"{self.task_type}/{self.model_family}/{self.implementation}",
                    "dataset_id": data_handoff.get("dataset_id"),
                    "dataset_version": data_handoff.get("dataset_version"),
                    "data_processing_experiment_id": data_handoff.get("data_processing_experiment_id"),
                }},
            )
            study = service.create_study(
                experiment_id,
                search_space,
                objectives,
                budgets,
                strategy=legacy_strategy,
                sampler_strategy=sampler,
                pruner_strategy=pruner,
                reduction_factor=reduction_factor,
                max_trials=run_limit,
                initial_trial_count=initial_count,
                promotion_limits=promotion_limits,
                max_training_runs=run_limit,
                min_completed_per_rung=int(request.budget.get("min_completed_per_rung", 1)),
                warm_start_trials=campaign_history,
                random_seed=int(config.get("seed", 0) or 0) + study_index,
            )
            tracker.update_hpo_experiment(
                experiment_id,
                extensions={"optimization": {"study_id": study.study_id}},
            )
            tracker.update_hpo_experiment(experiment_id, status="running", extensions={"optimization": {
                "objective": request.objective, "workflow": "langgraph", "campaign": campaign.to_dict(),
                "strategy_proposal": proposal.to_dict() if proposal else None,
                "strategy_decision": decision.to_dict(), "adopted_plan": adopted_plan, "data_handoff": data_handoff,
            }})
            scheduled = HPOScheduler(
                service, self._trial_executor(experiment_id, data_folder, request.context.get("runtime_options")),
                retry_policy=RetryPolicy(int(request.budget.get("max_retries", 1))),
                strategy_reviewer=self._runtime_strategy_reviewer(request, campaign),
                review_interval_trials=int(request.budget.get("strategy_review_interval_trials", 3)),
            ).run(study)
            if scheduled.study.status != "completed":
                error = "; ".join(scheduled.errors) or "HPO workflow failed"
                tracker.update_hpo_experiment(experiment_id, status="failed", error=error)
                raise RuntimeError(error)
            current_best = service.load_trial(experiment_id, str(scheduled.study.best_trial_id))
            current_value = float(current_best.metrics[objectives[0].metric])
            campaign_policy.record_study(
                campaign, experiment_id=experiment_id, study_id=scheduled.study.study_id,
                best_value=current_value, training_runs=service.training_runs_used(scheduled.study),
            )
            campaign.study_summaries[-1]["best_parameters"] = current_best.parameters
            campaign.study_summaries[-1]["best_metrics"] = self._best_metric_record(current_best, objectives[0])
            campaign.study_summaries[-1]["strategy_reviews"] = scheduled.study.strategy_reviews
            campaign.study_summaries[-1]["sampler"] = scheduled.study.candidate_strategy or scheduled.study.sampler_strategy
            campaign.study_summaries[-1]["pruner"] = scheduled.study.pruner_strategy
            campaign.study_summaries[-1]["warm_start_trial_count"] = len(scheduled.study.warm_start_trials)
            campaign.study_summaries[-1]["learning_summary"] = self._study_learning_summary(scheduled.study, current_best, objectives[0])
            study_feedback = HPOFeedbackAnalyzer().analyze(
                scheduled.study,
                scheduled.trials,
            )
            campaign.study_summaries[-1]["resource_summary"] = {
                "failure_rate": study_feedback.get("failure_rate"),
                "failure_clusters": study_feedback.get("failure_clusters") or {},
                "cost_summary": study_feedback.get("cost_summary") or {},
                "rung_summaries": study_feedback.get("rung_summaries") or [],
            }
            campaign_history = self._merge_campaign_history(
                campaign_history,
                scheduled.trials,
                objectives[0],
                scheduled.study.pruner_strategy or "none",
            )
            campaign.study_summaries[-1]["campaign_history_count"] = len(campaign_history)
            study_results.append({"experiment_id": experiment_id, "study_id": scheduled.study.study_id, "status": scheduled.study.status, "trial_count": len(scheduled.trials), "best_trial_id": scheduled.study.best_trial_id})
            if best_trial is None or (
                current_value < float(best_trial.metrics[objectives[0].metric])
                if objectives[0].mode == "min" else current_value > float(best_trial.metrics[objectives[0].metric])
            ):
                best_trial, best_experiment_id = current_best, experiment_id
            tracker.update_hpo_experiment(
                experiment_id,
                status="success",
                parameters=current_best.parameters,
                metrics={"best": self._best_metric_record(current_best, objectives[0])},
                extensions={"optimization": {
                    "campaign": campaign.to_dict(),
                    "latest_trial": {
                        "trial_id": current_best.trial_id,
                        "phase": "completed",
                        "status": current_best.status,
                        "updated_at": current_best.updated_at,
                    },
                }},
            )
            if not campaign_policy.should_continue(campaign):
                break
            sampler = (
                scheduled.study.candidate_strategy
                or scheduled.study.sampler_strategy
                or sampler
            )
            pruner = scheduled.study.pruner_strategy or pruner
            search_space = scheduled.study.search_space

        if best_trial is None or best_experiment_id is None:
            raise RuntimeError("optimization campaign produced no valid completed Study")
        duration = (datetime.now() - started_at).total_seconds()
        for item in study_results:
            tracker.update_hpo_experiment(
                item["experiment_id"],
                duration=duration,
                extensions={"optimization": {"campaign": campaign.to_dict()}},
            )
        self.memory_service.remember_episode(EpisodeMemory(
            agent_type="hpo_agent",
            objective=request.objective,
            action={
                "strategy": "successive_halving" if pruner == "successive_halving" else sampler,
                "sampler": sampler,
                "pruner": pruner,
                "best_config": best_trial.parameters,
            },
            outcome={"best_metrics": best_trial.metrics, "campaign": campaign.to_dict(), "duration": duration},
            summary=f"campaign completed {len(campaign.study_summaries)} studies: {campaign.stop_reason}",
            experiment_ids=[item["experiment_id"] for item in study_results],
            scope=self.memory_scope,
            importance=0.9,
        ))
        return AgentTaskResult(
            status="success",
            summary={
                "strategy": "successive_halving" if pruner == "successive_halving" else sampler,
                "sampler": sampler,
                "pruner": pruner,
                "best_trial_id": best_trial.trial_id,
                "best_parameters": best_trial.parameters,
                "campaign": campaign.to_dict(),
                "studies": study_results,
                "data_handoff": data_handoff,
            },
            metrics=self._best_metric_record(best_trial, objectives[0]),
            recommendations=[],
            artifacts=best_trial.artifacts,
            experiment_ids={"hpo": best_experiment_id, "campaign": [item["experiment_id"] for item in study_results]},
            request_id=request.request_id,
        )

    @staticmethod
    def _merge_campaign_history(
        existing: List[Dict[str, Any]],
        trials: List[Any],
        objective: Objective,
        pruner: str,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Keep fidelity-compatible observations for subsequent Study samplers."""
        merged = {
            json.dumps(item.get("parameters") or {}, sort_keys=True, default=str): dict(item)
            for item in existing
            if isinstance(item, dict)
        }
        for trial in trials:
            if pruner == "successive_halving" and int(getattr(trial, "rung", 0)) != 0:
                continue
            value = (getattr(trial, "metrics", {}) or {}).get(objective.metric)
            if not isinstance(value, (int, float)):
                continue
            if getattr(trial, "status", None) not in {"completed", "promoted"}:
                continue
            parameters = dict(getattr(trial, "parameters", {}) or {})
            signature = json.dumps(parameters, sort_keys=True, default=str)
            merged[signature] = {
                "trial_id": f"warm_{getattr(trial, 'trial_id', signature)}",
                "parameters": parameters,
                "budget": trial.budget.to_dict(),
                "status": "completed",
                "parent_trial_id": getattr(trial, "parent_trial_id", None),
                "rung": int(getattr(trial, "rung", 0)),
                "metrics": dict(getattr(trial, "metrics", {}) or {}),
                "intermediate_metrics": [],
                "cost": {"source": "prior_study"},
                "artifacts": [],
                "stop_reason": None,
                "created_at": getattr(trial, "created_at", ""),
                "updated_at": getattr(trial, "updated_at", ""),
            }
        return list(merged.values())[-max(int(limit), 1):]

    @staticmethod
    def _study_learning_summary(study: Any, best_trial: Any, objective: Objective) -> Dict[str, Any]:
        reviews = list(getattr(study, "strategy_reviews", []) or [])
        last_review = reviews[-1] if reviews else {}
        decision = last_review.get("decision") or {}
        proposal = last_review.get("proposal") or {}
        final_sampler = (
            getattr(study, "candidate_strategy", None)
            or getattr(study, "sampler_strategy", None)
            or (
                "random_search"
                if getattr(study, "strategy", None) == "successive_halving"
                else getattr(study, "strategy", None)
            )
        )
        final_pruner = (
            getattr(study, "pruner_strategy", None)
            or (
                "successive_halving"
                if getattr(study, "strategy", None) == "successive_halving"
                else "none"
            )
        )
        final_strategy = (
            "successive_halving"
            if final_pruner == "successive_halving"
            else final_sampler
        )
        final_search_space = study.search_space.to_dict()
        return {
            "local_search_anchor": {
                "trial_id": getattr(best_trial, "trial_id", None),
                "parameters": dict(getattr(best_trial, "parameters", {}) or {}),
                "metric": objective.metric,
                "mode": objective.mode,
                "value": (getattr(best_trial, "metrics", {}) or {}).get(objective.metric),
            },
            "final_strategy": final_strategy,
            "final_candidate_strategy": final_sampler,
            "final_sampler": final_sampler,
            "final_pruner": final_pruner,
            "warm_start_trial_count": len(getattr(study, "warm_start_trials", []) or []),
            "final_search_space": final_search_space,
            "last_review": {
                "trigger": last_review.get("trigger"),
                "proposal_action": proposal.get("action"),
                "requested_strategy": proposal.get("requested_strategy"),
                "reason_codes": proposal.get("reason_codes") or [],
                "accepted_fields": decision.get("accepted_fields") or [],
                "rejected_fields": decision.get("rejected_fields") or [],
                "applied_candidate_strategy": last_review.get("applied_candidate_strategy"),
            },
            "next_study_recommendation": {
                "strategy": final_strategy,
                "sampler": final_sampler,
                "pruner": final_pruner,
                "search_space": final_search_space,
                "anchor_parameters": dict(getattr(best_trial, "parameters", {}) or {}),
                "reason_codes": proposal.get("reason_codes") or [],
            },
        }

    @staticmethod
    def _cross_study_memory(campaign: Dict[str, Any]) -> Dict[str, Any]:
        summaries = list(campaign.get("study_summaries") or [])
        recent = summaries[-3:]
        best_summary = None
        best_value = campaign.get("best_value")
        best_experiment_id = campaign.get("best_experiment_id")
        if best_experiment_id:
            best_summary = next(
                (item for item in summaries if item.get("experiment_id") == best_experiment_id),
                None,
            )
        if best_summary is None and summaries:
            best_summary = summaries[-1]
        latest_learning = (recent[-1].get("learning_summary") if recent else None) or {}
        return {
            "prior_study_count": len(summaries),
            "best_value": best_value,
            "best_experiment_id": best_experiment_id,
            "best_parameters": (best_summary or {}).get("best_parameters") or {},
            "local_search_anchor": latest_learning.get("local_search_anchor"),
            "next_study_recommendation": latest_learning.get("next_study_recommendation"),
            "recent_learnings": [
                {
                    "experiment_id": item.get("experiment_id"),
                    "study_id": item.get("study_id"),
                    "best_value": item.get("best_value"),
                    "improvement": item.get("improvement"),
                    "improved": item.get("improved"),
                    "best_parameters": item.get("best_parameters") or {},
                    "learning_summary": item.get("learning_summary") or {},
                }
                for item in recent
            ],
        }

    @staticmethod
    def _best_metric_record(trial: Any, objective: Objective) -> Dict[str, Any]:
        metrics = dict(trial.metrics or {})
        training, evaluation = HPOAgent._split_trial_metrics(metrics, objective.metric)
        primary_value = metrics.get(objective.metric)
        return {
            "trial_id": trial.trial_id,
            "primary_metric": objective.metric,
            "primary_mode": objective.mode,
            "primary_value": primary_value,
            **metrics,
            "training": training,
            "evaluation": evaluation,
        }

    @staticmethod
    def _split_trial_metrics(
        metrics: Dict[str, Any],
        primary_metric: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        training: Dict[str, Any] = {}
        evaluation: Dict[str, Any] = {}
        training_names = {
            "valid_error_rate",
            "final_epoch",
            "final_lr",
            "final_train_loss",
            "final_valid_loss",
            "final_valid_error_rate",
            "total_epochs",
            "best_epoch",
            "best_valid_loss",
            "best_error_rate",
        }
        evaluation_names = {
            primary_metric,
            "eer",
            "min_dcf",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
        }
        for key, value in metrics.items():
            normalized = key.lower()
            if key in training_names or normalized.startswith(("train_", "valid_", "final_", "best_")):
                training[key] = value
            elif key in evaluation_names:
                evaluation[key] = value
        return training, evaluation

    def _trial_executor(self, experiment_id: str, data_folder: str, runtime_options: Any = None):
        runtime_options = dict(runtime_options or {}) if isinstance(runtime_options, dict) else {}

        def execute(trial: Any, attempt: int) -> Dict[str, Any]:
            from agent.tools.evaluation_tools import RunEvaluation
            from agent.tools.training_tools import TrainModel

            parameters = dict(trial.parameters)
            model_family = str(parameters.pop("model_family", self.model_family))
            implementation = str(parameters.pop("implementation", self.implementation))
            runner = str(parameters.pop("runner", self.runner))
            train_result = json.loads(TrainModel.invoke({
                "experiment_id": experiment_id,
                "trial_id": trial.trial_id,
                "data_folder": data_folder,
                "parameters_json": json.dumps(parameters, ensure_ascii=False),
                "budget_json": json.dumps(trial.budget.to_dict(), ensure_ascii=False),
                "task_type": self.task_type,
                "model_family": model_family,
                "implementation": implementation,
                "runner": runner,
                "experiments_dir": str(self.experiments_dir),
                "device": runtime_options.get("device"),
                "precision": runtime_options.get("precision"),
                "eval_precision": runtime_options.get("eval_precision"),
            }))
            if train_result.get("status") != "success":
                return train_result
            evaluation = json.loads(RunEvaluation.invoke({
                "experiment_id": experiment_id,
                "trial_id": trial.trial_id,
                "data_folder": data_folder,
                "experiments_dir": str(self.experiments_dir),
                "runner": runner,
                "task_type": self.task_type,
                "model_family": model_family,
                "implementation": implementation,
                "device": runtime_options.get("device"),
                "precision": runtime_options.get("precision"),
                "eval_precision": runtime_options.get("eval_precision"),
            }))
            metrics: Dict[str, Any] = {}
            for values in (evaluation.get("metrics") or {}).values():
                metrics.update(values or {})
            return {
                "status": evaluation.get("status", "failed"),
                "error": evaluation.get("error"),
                "metrics": metrics,
                "artifacts": evaluation.get("artifacts") or [],
                "cost": {"attempt": attempt},
            }
        return execute

    def _planning_proposal(
        self,
        request: AgentTaskRequest,
        search_space: SearchSpace,
        budgets: List[TrialBudget],
        selected_sampler: str,
        selected_pruner: str,
        hard_max_training_runs: int,
        initial_trial_count: int,
        promotion_limits: List[int],
        reduction_factor: int,
        campaign: Optional[Dict[str, Any]] = None,
    ) -> StrategyProposal:
        memory_context = self.memory_service.format_context(MemoryQuery(
            agent_type="hpo_agent",
            task_type=self.task_type,
            model_family=self.model_family,
            dataset_key=self.memory_scope.dataset_key,
            limit=5,
        ))
        prompt = self._strategy_proposal_prompt({
            "phase": "study_planning",
            "objective": request.objective,
            "task_type": self.task_type,
            "model_family": self.model_family,
            "selected_strategy": (
                "successive_halving"
                if selected_pruner == "successive_halving"
                else selected_sampler
            ),
            "selected_sampler": selected_sampler,
            "selected_pruner": selected_pruner,
            "available_samplers": HPOService.available_samplers(),
            "available_pruners": HPOService.available_pruners(),
            "requested_strategy": request.context.get("strategy", "auto"),
            "requested_sampler": request.context.get("sampler"),
            "requested_pruner": request.context.get("pruner"),
            "primary_metric": request.context.get("primary_metric", "eer"),
            "metric_mode": request.context.get("metric_mode", "min"),
            "hard_max_training_runs": hard_max_training_runs,
            "allocation": {
                "initial_trial_count": initial_trial_count,
                "promotion_limits": promotion_limits,
                "reduction_factor": reduction_factor,
            },
            "resource_profile": self._resource_snapshot(
                request.context.get("runtime_options"),
                request.context.get("resource_profile"),
            ),
            "search_budget_analysis": self._search_budget_analysis(
                search_space,
                budgets,
                hard_max_training_runs,
                initial_trial_count,
                promotion_limits,
                reduction_factor,
                campaign,
            ),
            "search_space": search_space.to_dict(),
            "budgets": [item.to_dict() for item in budgets],
            "campaign": self._compact_campaign(campaign or {}),
            "cross_study_memory": self._cross_study_memory(campaign or {}),
            "reference_profile": self._reference_search_profile(self.model_family),
            "historical_memory": memory_context,
        })
        try:
            value = json.loads(self._extract_message_content(self.llm.invoke(prompt)))
            return StrategyProposal.from_dict(value)
        except Exception as exc:
            return StrategyProposal(
                action="invalid_proposal",
                reason_codes=["proposal_parse_error"],
                evidence={"error": f"{type(exc).__name__}: {exc}"},
            )

    def _runtime_strategy_reviewer(self, request: AgentTaskRequest, campaign: OptimizationCampaign):
        if not self.enable_llm_advisor:
            return None

        def review(study: Any, feedback: Dict[str, Any]) -> StrategyProposal:
            memory_context = self.memory_service.format_context(MemoryQuery(
                agent_type="hpo_agent",
                task_type=self.task_type,
                model_family=self.model_family,
                dataset_key=self.memory_scope.dataset_key,
                limit=5,
            ))
            prompt = self._strategy_proposal_prompt({
                "phase": "runtime_review",
                "objective": request.objective,
                "task_type": self.task_type,
                "model_family": self.model_family,
                "study": self._compact_study(study),
                "available_samplers": HPOService.available_samplers(),
                "available_pruners": HPOService.available_pruners(),
                "runtime_mutable_fields": ["requested_sampler", "search_space"],
                "feedback": feedback,
                "resource_profile": self._resource_snapshot(
                    request.context.get("runtime_options"),
                    request.context.get("resource_profile"),
                ),
                "search_budget_analysis": self._search_budget_analysis(
                    study.search_space,
                    list(study.budgets),
                    int(study.max_training_runs or 1),
                    int(study.initial_trial_count or 1),
                    list(study.promotion_limits or []),
                    int(study.reduction_factor or 3),
                    campaign.to_dict(),
                ),
                "campaign": self._compact_campaign(campaign.to_dict()),
                "cross_study_memory": self._cross_study_memory(campaign.to_dict()),
                "reference_profile": self._reference_search_profile(self.model_family),
                "historical_memory": memory_context,
            })
            try:
                return StrategyProposal.from_dict(json.loads(self._extract_message_content(self.llm.invoke(prompt))))
            except Exception as exc:
                return StrategyProposal(
                    action="invalid_proposal",
                    reason_codes=["runtime_proposal_parse_error"],
                    evidence={"error": f"{type(exc).__name__}: {exc}"},
                )
        return review

    @staticmethod
    def _default_initial_trial_count(strategy: str, budgets: List[TrialBudget], run_limit: int) -> int:
        if strategy != "successive_halving" or len(budgets) <= 1:
            return min(3, max(int(run_limit), 1))
        reduction_factor = 3
        # Start from one complete 3x halving bracket: three rungs -> 9 + 3 + 1.
        baseline = reduction_factor ** max(len(budgets) - 1, 0)
        target = min(max(int(run_limit), 1), baseline)
        for count in range(target, 0, -1):
            planned_runs = count + sum(
                HPOAgent._default_promotion_limits(
                    count,
                    budgets,
                    reduction_factor,
                )
            )
            if planned_runs <= run_limit:
                return count
        return 1

    @staticmethod
    def _default_promotion_limits(
        initial_count: int,
        budgets: List[TrialBudget],
        reduction_factor: int = 3,
    ) -> List[int]:
        limits: List[int] = []
        current = max(int(initial_count), 0)
        for _ in range(max(len(budgets) - 1, 0)):
            current = max(1, (current + reduction_factor - 1) // reduction_factor)
            limits.append(current)
        return limits

    @staticmethod
    def _resource_snapshot(
        runtime_options: Any = None,
        declared_profile: Any = None,
    ) -> Dict[str, Any]:
        """Collect bounded planning evidence without making resources mandatory."""
        runtime = dict(runtime_options or {}) if isinstance(runtime_options, dict) else {}
        snapshot: Dict[str, Any] = {
            "requested_runtime": {
                key: runtime.get(key)
                for key in ("device", "precision", "eval_precision")
                if runtime.get(key) is not None
            },
            "cpu_count": os.cpu_count(),
        }
        if isinstance(declared_profile, dict) and declared_profile:
            snapshot["declared_limits"] = dict(declared_profile)
        try:
            import psutil

            memory = psutil.virtual_memory()
            snapshot["system_memory_gb"] = {
                "total": round(float(memory.total) / (1024 ** 3), 2),
                "available": round(float(memory.available) / (1024 ** 3), 2),
            }
        except Exception:
            snapshot["system_memory_gb"] = None
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            snapshot["cuda"] = {
                "available": cuda_available,
                "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
                "devices": [],
            }
            if cuda_available:
                for index in range(int(torch.cuda.device_count())):
                    properties = torch.cuda.get_device_properties(index)
                    device = {
                        "index": index,
                        "name": str(properties.name),
                        "total_memory_gb": round(
                            float(properties.total_memory) / (1024 ** 3),
                            2,
                        ),
                    }
                    try:
                        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                        device["free_memory_gb"] = round(
                            float(free_bytes) / (1024 ** 3),
                            2,
                        )
                        device["observed_total_memory_gb"] = round(
                            float(total_bytes) / (1024 ** 3),
                            2,
                        )
                    except Exception:
                        pass
                    snapshot["cuda"]["devices"].append(device)
        except Exception as exc:
            snapshot["cuda"] = {
                "available": None,
                "probe_error": type(exc).__name__,
            }
        return snapshot

    @staticmethod
    def _search_budget_analysis(
        search_space: SearchSpace,
        budgets: List[TrialBudget],
        hard_max_training_runs: int,
        initial_trial_count: int,
        promotion_limits: List[int],
        reduction_factor: int,
        campaign: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        counts = [int(initial_trial_count), *[int(item) for item in promotion_limits]]
        counts = counts[:len(budgets)]
        counts.extend([0] * max(len(budgets) - len(counts), 0))
        rung_plan = []
        estimated_work_units = 0.0
        work_is_complete = True
        for budget, count in zip(budgets, counts):
            unit_work = None
            if budget.epochs is not None and budget.data_fraction is not None:
                unit_work = float(budget.epochs) * float(budget.data_fraction)
                estimated_work_units += count * unit_work
            else:
                work_is_complete = False
            rung_plan.append({
                "stage": budget.stage,
                "planned_runs": count,
                "epochs": budget.epochs,
                "data_fraction": budget.data_fraction,
                "max_duration_seconds": budget.max_duration_seconds,
                "relative_work_per_run": unit_work,
            })
        campaign = campaign or {}
        summaries = list(campaign.get("study_summaries") or [])
        used_runs = sum(int(item.get("training_runs") or 0) for item in summaries)
        total_limit = campaign.get("max_total_training_runs")
        remaining_runs = (
            max(int(total_limit) - used_runs, 0)
            if total_limit is not None else None
        )
        resource_sensitive = {
            "batch_size",
            "sentence_len",
            "sample_rate",
            "embedding_dim",
            "channels",
        }
        return {
            "search_dimension_count": len(search_space.parameters),
            "search_parameter_names": [
                parameter.name for parameter in search_space.parameters
            ],
            "resource_sensitive_parameters": [
                parameter.name
                for parameter in search_space.parameters
                if parameter.name in resource_sensitive
            ],
            "finite_grid_cardinality": HPOPlanningPolicy.grid_cardinality(
                search_space
            ),
            "hard_max_training_runs": int(hard_max_training_runs),
            "planned_training_runs": sum(counts),
            "unused_training_run_capacity": max(
                int(hard_max_training_runs) - sum(counts),
                0,
            ),
            "reduction_factor": int(reduction_factor),
            "rung_plan": rung_plan,
            "estimated_relative_work_units": (
                round(estimated_work_units, 4) if work_is_complete else None
            ),
            "campaign_used_training_runs": used_runs,
            "campaign_remaining_training_runs": remaining_runs,
            "recent_resource_summaries": [
                item.get("resource_summary")
                for item in summaries[-3:]
                if item.get("resource_summary")
            ],
        }
    @staticmethod
    def _strategy_proposal_prompt(context: Dict[str, Any]) -> str:
        return render_prompt("hpo_strategy_proposal", context=context)

    @staticmethod
    def _compact_campaign(campaign: Dict[str, Any]) -> Dict[str, Any]:
        objective = campaign.get("objective") or {}
        summaries = list(campaign.get("study_summaries") or [])
        return {
            "campaign_id": campaign.get("campaign_id"),
            "status": campaign.get("status"),
            "stop_reason": campaign.get("stop_reason"),
            "objective": {
                "metric": objective.get("metric"),
                "mode": objective.get("mode"),
            },
            "target_value": campaign.get("target_value"),
            "max_studies": campaign.get("max_studies"),
            "max_total_training_runs": campaign.get("max_total_training_runs"),
            "best_value": campaign.get("best_value"),
            "study_count": len(summaries),
            "recent_studies": summaries[-3:],
        }

    @staticmethod
    def _compact_study(study: Any) -> Dict[str, Any]:
        return {
            "study_id": getattr(study, "study_id", None),
            "experiment_id": getattr(study, "experiment_id", None),
            "status": getattr(study, "status", None),
            "strategy": getattr(study, "strategy", None),
            "sampler_strategy": getattr(study, "sampler_strategy", None),
            "pruner_strategy": getattr(study, "pruner_strategy", None),
            "candidate_strategy": getattr(study, "candidate_strategy", None),
            "best_trial_id": getattr(study, "best_trial_id", None),
            "trial_count": len(getattr(study, "trial_ids", []) or []),
            "max_training_runs": getattr(study, "max_training_runs", None),
            "initial_trial_count": getattr(study, "initial_trial_count", None),
            "promotion_limits": list(getattr(study, "promotion_limits", []) or []),
            "reduction_factor": getattr(study, "reduction_factor", None),
            "warm_start_trial_count": len(getattr(study, "warm_start_trials", []) or []),
            "search_space": study.search_space.to_dict(),
            "budgets": [item.to_dict() for item in getattr(study, "budgets", [])],
            "recent_reviews": list(getattr(study, "strategy_reviews", []) or [])[-3:],
        }

    @staticmethod
    def _reference_search_profile(model_family: str) -> Dict[str, Any]:
        profiles = {
            "ecapa_tdnn": {
                "baseline_parameters": {
                    "lr": 0.001,
                    "batch_size": 32,
                    "margin": 0.2,
                    "weight_decay": 2e-6,
                },
                "stable_search_space": {
                    "parameters": [
                        {"name": "lr", "parameter_type": "float", "low": 3e-4, "high": 3e-3, "scale": "log"},
                        {"name": "batch_size", "parameter_type": "categorical", "choices": [16, 24, 32]},
                        {"name": "margin", "parameter_type": "float", "low": 0.15, "high": 0.3},
                        {"name": "weight_decay", "parameter_type": "float", "low": 5e-7, "high": 2e-5, "scale": "log"},
                    ],
                    "constraints": [],
                },
                "local_adjustment_policy": {
                    "max_changed_parameters_per_review": 2,
                    "lr_boundary_factor": 2.0,
                    "weight_decay_boundary_factor": 3.0,
                    "margin_step": 0.05,
                    "resource_first_on_oom": ["batch_size"],
                    "preferred_strategy_progression": ["successive_halving", "adaptive_search", "tpe"],
                },
                "rationale": "SpeechBrain ECAPA recipe is stable near lr=0.001, margin=0.2, weight_decay=2e-6; HPO should make local evidence-backed moves before broad exploration.",
            },
            "resnet": {
                "baseline_parameters": {
                    "lr": 0.001,
                    "batch_size": 32,
                    "margin": 0.2,
                    "weight_decay": 2e-6,
                },
                "stable_search_space": {
                    "parameters": [
                        {"name": "lr", "parameter_type": "float", "low": 3e-4, "high": 3e-3, "scale": "log"},
                        {"name": "batch_size", "parameter_type": "categorical", "choices": [16, 24, 32]},
                        {"name": "sentence_len", "parameter_type": "categorical", "choices": [2.0, 3.0, 4.0]},
                        {"name": "margin", "parameter_type": "float", "low": 0.15, "high": 0.3},
                        {"name": "weight_decay", "parameter_type": "float", "low": 5e-7, "high": 2e-5, "scale": "log"},
                    ],
                    "constraints": [],
                },
                "local_adjustment_policy": {
                    "max_changed_parameters_per_review": 2,
                    "lr_boundary_factor": 2.0,
                    "weight_decay_boundary_factor": 3.0,
                    "margin_step": 0.05,
                    "resource_first_on_oom": ["batch_size", "sentence_len"],
                    "preferred_strategy_progression": ["successive_halving", "adaptive_search", "tpe"],
                },
                "rationale": "ResNet speaker-recognition tuning should stay near the stable SpeechBrain recipe before widening multiple coupled parameters.",
            },
        }
        return profiles.get(model_family, {
            "baseline_parameters": {},
            "stable_search_space": None,
            "local_adjustment_policy": {
                "max_changed_parameters_per_review": 2,
                "preferred_strategy_progression": ["successive_halving", "adaptive_search", "tpe"],
            },
        })


    def _resolve_search_space(self, value: Optional[Dict[str, Any]]) -> SearchSpace:
        return self._build_search_space(value or self._default_model_search_space())

    def _default_model_search_space(self) -> Dict[str, Any]:
        adapter = get_model_adapter(self.model_family)
        factory = getattr(adapter, "default_search_space", None)
        if not callable(factory):
            raise ValueError(
                f"model adapter '{self.model_family}' does not declare a default search space; "
                "provide context.search_space explicitly or implement default_search_space()"
            )
        value = factory()
        if not isinstance(value, dict) or not value.get("parameters"):
            raise ValueError(f"model adapter '{self.model_family}' returned an empty default search space")
        return value

    @staticmethod
    def _build_search_space(
        value: Optional[Dict[str, Any]],
        default: Optional[Dict[str, Any]] = None,
    ) -> SearchSpace:
        selected = value or default
        if selected:
            return SearchSpace(
                [SearchParameter(**item) for item in selected.get("parameters") or []],
                selected.get("constraints") or [],
            )
        raise ValueError("search space must contain at least one parameter")

    @staticmethod
    def _validate_search_budget_compatibility(
        search_space: SearchSpace,
        budgets: List[TrialBudget],
    ) -> None:
        """Reject parameters whose values would be silently replaced by a budget."""
        parameter_names = {parameter.name for parameter in search_space.parameters}
        if "number_of_epochs" in parameter_names and any(
            budget.epochs is not None for budget in budgets
        ):
            raise ValueError(
                "search parameter 'number_of_epochs' conflicts with budget.epochs; "
                "remove it from the search space or set every budget epochs value to null"
            )

    @staticmethod
    def _build_budgets(
        value: Optional[List[Dict[str, Any]]],
        requested_strategy: str = "auto",
    ) -> List[TrialBudget]:
        if value:
            return [TrialBudget(**item) for item in value]
        if requested_strategy != "successive_halving":
            return [TrialBudget("full", epochs=20, data_fraction=1.0)]
        return [
            TrialBudget("screening", epochs=3, data_fraction=0.25),
            TrialBudget("promotion", epochs=8, data_fraction=0.5),
            TrialBudget("confirmation", epochs=20, data_fraction=1.0),
        ]

    def get_execution_details(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "workflow": "langgraph",
            "available_samplers": HPOService.available_samplers(),
            "available_pruners": HPOService.available_pruners(),
            "llm_role": "structured_strategy_proposal",
            "decision_authority": "HPOService + OptimizationPlanDecisionPolicy",
            "feedback_loop": "trial/rung reviews + historical memory + optimization campaign",
            "enable_llm_advisor": self.enable_llm_advisor,
        }


def create_hpo_agent(**kwargs: Any) -> HPOAgent:
    return HPOAgent(**kwargs)


__all__ = ["HPOAgent", "create_hpo_agent"]
