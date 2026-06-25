"""LangGraph HPO agent with a single structured task interface."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.agents.base_agent import LangGraphAgent
from agent.agents.communication import AgentTaskRequest, AgentTaskResult
from agent.hpo import (
    HPOPlanningPolicy,
    HPOScheduler,
    HPOService,
    Objective,
    OptimizationCampaign,
    CampaignPolicy,
    RetryPolicy,
    SearchParameter,
    SearchSpace,
    StrategyDecisionPolicy,
    StrategyProposal,
    TrialBudget,
)
from agent.data_processing.handoff import resolve_data_handoff
from agent.memory import EpisodeMemory, MemoryQuery, MemoryScope, MemoryService
from agent.models import get_model_adapter
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
        decision_policy: Optional[StrategyDecisionPolicy] = None,
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
        self.decision_policy = decision_policy or StrategyDecisionPolicy(self.planning_policy)
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
        budgets = self._build_budgets(request.context.get("budgets"), requested_strategy)
        self._validate_search_budget_compatibility(search_space, budgets)
        service = HPOService(tracker)
        base_strategy = self.planning_policy.select_strategy(
            requested_strategy,
            search_space,
            budgets,
            per_study_limit,
            service.available_strategies(),
        )
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
        strategy = base_strategy
        for study_index in range(max_studies):
            remaining = campaign_policy.remaining_runs(campaign)
            run_limit = min(per_study_limit, remaining) if remaining is not None else per_study_limit
            if run_limit <= 0:
                campaign.status, campaign.stop_reason = "completed", "max_total_training_runs_reached"
                break
            proposal = (
                self._planning_proposal(request, search_space, budgets, strategy, run_limit, campaign.to_dict())
                if self.enable_llm_advisor else None
            )
            decision = self.decision_policy.review(
                proposal,
                base_strategy=strategy,
                base_search_space=search_space,
                base_budgets=budgets,
                hard_max_training_runs=run_limit,
                objectives=objectives,
                available_strategies=service.available_strategies(),
                validate_plan=service.validate_study_plan,
            )
            strategy = decision.adopted_strategy
            search_space = self._resolve_search_space(decision.adopted_search_space)
            budgets = self._build_budgets(decision.adopted_budgets, strategy)
            self._validate_search_budget_compatibility(search_space, budgets)
            run_limit = decision.adopted_max_training_runs
            adopted_plan = {
                "strategy": strategy,
                "search_space": search_space.to_dict(),
                "budgets": [budget.to_dict() for budget in budgets],
                "max_training_runs": run_limit,
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
            initial_count = min(int(request.budget.get("initial_trial_count") or min(3, run_limit)), run_limit)
            default_promotions = [max(1, initial_count // 3), 1][: max(len(budgets) - 1, 0)]
            promotion_limits = [] if strategy != "successive_halving" else request.budget.get("promotion_limits", default_promotions)
            study = service.create_study(
                experiment_id, search_space, objectives, budgets, strategy=strategy,
                max_trials=run_limit, initial_trial_count=initial_count, promotion_limits=promotion_limits,
                max_training_runs=run_limit,
                min_completed_per_rung=int(request.budget.get("min_completed_per_rung", 1)),
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
                service, self._trial_executor(experiment_id, data_folder),
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
            campaign.study_summaries[-1]["strategy_reviews"] = scheduled.study.strategy_reviews
            study_results.append({"experiment_id": experiment_id, "study": scheduled.study.to_dict(), "trials": [item.to_dict() for item in scheduled.trials]})
            if best_trial is None or (
                current_value < float(best_trial.metrics[objectives[0].metric])
                if objectives[0].mode == "min" else current_value > float(best_trial.metrics[objectives[0].metric])
            ):
                best_trial, best_experiment_id = current_best, experiment_id
            tracker.update_hpo_experiment(
                experiment_id, status="success", parameters=current_best.parameters, metrics={"best": current_best.metrics},
                extensions={"optimization": {"campaign": campaign.to_dict(), "study": scheduled.study.to_dict(), "trials": [item.to_dict() for item in scheduled.trials]}},
            )
            if not campaign_policy.should_continue(campaign):
                break
            strategy = scheduled.study.candidate_strategy or scheduled.study.strategy
            search_space = scheduled.study.search_space
            if strategy != "successive_halving" and len(budgets) > 1:
                budgets = [budgets[-1]]

        if best_trial is None or best_experiment_id is None:
            raise RuntimeError("optimization campaign produced no valid completed Study")
        duration = (datetime.now() - started_at).total_seconds()
        for item in study_results:
            tracker.update_hpo_experiment(
                item["experiment_id"],
                extensions={"optimization": {"campaign": campaign.to_dict()}},
            )
        self.memory_service.remember_episode(EpisodeMemory(
            agent_type="hpo_agent",
            objective=request.objective,
            action={"strategy": strategy, "best_config": best_trial.parameters},
            outcome={"best_metrics": best_trial.metrics, "campaign": campaign.to_dict(), "duration": duration},
            summary=f"campaign completed {len(campaign.study_summaries)} studies: {campaign.stop_reason}",
            experiment_ids=[item["experiment_id"] for item in study_results],
            scope=self.memory_scope,
            importance=0.9,
        ))
        return AgentTaskResult(
            status="success",
            summary={
                "strategy": strategy,
                "best_trial_id": best_trial.trial_id,
                "best_parameters": best_trial.parameters,
                "campaign": campaign.to_dict(),
                "studies": study_results,
                "data_handoff": data_handoff,
            },
            metrics=best_trial.metrics,
            recommendations=[],
            artifacts=best_trial.artifacts,
            experiment_ids={"hpo": best_experiment_id, "campaign": [item["experiment_id"] for item in study_results]},
            request_id=request.request_id,
        )

    def _trial_executor(self, experiment_id: str, data_folder: str):
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
        selected_strategy: str,
        hard_max_training_runs: int,
        campaign: Optional[Dict[str, Any]] = None,
    ) -> StrategyProposal:
        memory_context = self.memory_service.format_context(MemoryQuery(
            agent_type="hpo_agent",
            task_type=self.task_type,
            model_family=self.model_family,
            dataset_key=self.memory_scope.dataset_key,
            limit=5,
        ))
        prompt = (
            "Return one JSON object matching this StrategyProposal schema exactly: "
            '{"action":"keep_strategy|refine_search_space|expand_search_space|switch_strategy|adjust_budget",'
            '"requested_strategy":"auto|random_search|grid_search|adaptive_search|tpe|successive_halving|null",'
            '"search_space":{"parameters":[],"constraints":[]} or null,'
            '"budgets":[{"stage":"name","epochs":1,"data_fraction":1.0,"max_duration_seconds":null}] or null,'
            '"max_training_runs":integer or null,"reason_codes":[],"evidence":{},'
            '"expected_effect":{},"confidence":0.0}. '
            "Submit a proposal only; do not execute tools. The service rejects invalid fields and "
            "max_training_runs cannot exceed the hard limit. "
            f"Selected strategy={selected_strategy}; request={json.dumps(request.to_dict(), ensure_ascii=False)}; "
            f"search_space={json.dumps(search_space.to_dict(), ensure_ascii=False)}; "
            f"budgets={json.dumps([item.to_dict() for item in budgets], ensure_ascii=False)}; "
            f"hard_max_training_runs={hard_max_training_runs}; "
            f"campaign={json.dumps(campaign or {}, ensure_ascii=False)}; "
            f"historical_memory={memory_context}"
        )
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
            prompt = (
                "Return one valid StrategyProposal JSON object for subsequent candidate generation only. "
                "Use completed metrics, failure clusters, boundary hits, prior reviews, campaign progress, "
                "and historical memory. Do not request tools or mutate Trial state. "
                f"request={json.dumps(request.to_dict(), ensure_ascii=False)}; "
                f"study={json.dumps(study.to_dict(), ensure_ascii=False)}; "
                f"feedback={json.dumps(feedback, ensure_ascii=False)}; "
                f"campaign={json.dumps(campaign.to_dict(), ensure_ascii=False)}; "
                f"historical_memory={memory_context}"
            )
            try:
                return StrategyProposal.from_dict(json.loads(self._extract_message_content(self.llm.invoke(prompt))))
            except Exception as exc:
                return StrategyProposal(
                    action="invalid_proposal",
                    reason_codes=["runtime_proposal_parse_error"],
                    evidence={"error": f"{type(exc).__name__}: {exc}"},
                )
        return review

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
            "available_strategies": HPOService.available_strategies(),
            "llm_role": "structured_strategy_proposal",
            "decision_authority": "HPOService + StrategyDecisionPolicy",
            "feedback_loop": "trial/rung reviews + historical memory + optimization campaign",
            "enable_llm_advisor": self.enable_llm_advisor,
        }


def create_hpo_agent(**kwargs: Any) -> HPOAgent:
    return HPOAgent(**kwargs)


__all__ = ["HPOAgent", "create_hpo_agent"]
