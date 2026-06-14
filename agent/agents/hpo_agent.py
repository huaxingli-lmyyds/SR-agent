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
    RetryPolicy,
    SearchParameter,
    SearchSpace,
    TrialBudget,
)
from agent.memory import EpisodeMemory, MemoryScope, MemoryService
from agent.utils import ConfigParser, ExperimentTracker
from agent.utils.path_tool import (
    get_config_file,
    get_hpo_experiments_dir,
    resolve_config_path,
    resolve_config_value_path,
    resolve_data_path,
)


class HPOAgent(LangGraphAgent):
    """Execute validated HPO workflows; the LLM only returns recommendations."""

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
        data_folder = str(resolve_data_path(config.get("data_folder")))
        output = resolve_config_value_path(config.get("output_folder"))
        self.memory_scope.dataset_key = data_folder

        search_space = self._build_search_space(request.context.get("search_space"))
        run_limit = int(request.budget.get("max_training_runs") or self.max_iterations)
        requested_strategy = str(request.context.get("strategy") or "auto")
        budgets = self._build_budgets(request.context.get("budgets"), requested_strategy)
        service = HPOService(tracker)
        strategy = self.planning_policy.select_strategy(
            requested_strategy,
            search_space,
            budgets,
            run_limit,
            service.available_strategies(),
        )
        recommendations = self._planning_advice(request, search_space, strategy) if self.enable_llm_advisor else {}
        experiment_id = tracker.create_hpo_experiment(
            config_path=self.config_path,
            data_folder=data_folder,
            output_folder=str(output) if output else None,
            description="LangGraph HPO run",
            task={
                "type": self.task_type,
                "dataset": data_folder,
                "primary_metric": request.context.get("primary_metric", "eer"),
                "metric_mode": request.context.get("metric_mode", "min"),
            },
            model={"family": self.model_family, "implementation": self.implementation, "config_path": self.config_path},
            execution={"runner": self.runner, "output_folder": str(output) if output else None},
        )
        initial_count = int(request.budget.get("initial_trial_count") or min(3, run_limit))
        default_promotions = [max(1, initial_count // 3), 1][: max(len(budgets) - 1, 0)]
        requested_promotions = request.budget.get("promotion_limits")
        promotion_limits = (
            []
            if strategy != "successive_halving"
            else default_promotions if requested_promotions is None else requested_promotions
        )
        study = service.create_study(
            experiment_id,
            search_space,
            [Objective(
                str(request.context.get("primary_metric", "eer")),
                str(request.context.get("metric_mode", "min")),
            )],
            budgets,
            strategy=strategy,
            max_trials=run_limit,
            initial_trial_count=initial_count,
            promotion_limits=promotion_limits,
            max_training_runs=run_limit,
            min_completed_per_rung=int(request.budget.get("min_completed_per_rung", 1)),
            random_seed=int(config.get("seed", 0) or 0),
        )
        tracker.update_hpo_experiment(
            experiment_id,
            status="running",
            extensions={"optimization": {
                "objective": request.objective,
                "workflow": "langgraph",
                "strategy": strategy,
                "recommendations": recommendations,
            }},
        )
        scheduled = HPOScheduler(
            service,
            self._trial_executor(experiment_id),
            retry_policy=RetryPolicy(int(request.budget.get("max_retries", 1))),
        ).run(study)
        if scheduled.study.status != "completed":
            error = "; ".join(scheduled.errors) or "HPO workflow failed"
            tracker.update_hpo_experiment(experiment_id, status="failed", error=error)
            raise RuntimeError(error)

        best_trial = service.load_trial(experiment_id, str(scheduled.study.best_trial_id))
        duration = (datetime.now() - started_at).total_seconds()
        tracker.update_hpo_experiment(
            experiment_id,
            status="success",
            duration=duration,
            parameters=best_trial.parameters,
            metrics={"best": best_trial.metrics},
            extensions={"optimization": {
                "workflow": "langgraph",
                "strategy": strategy,
                "recommendations": recommendations,
                "study": scheduled.study.to_dict(),
                "trials": [trial.to_dict() for trial in scheduled.trials],
            }},
        )
        self.memory_service.remember_episode(EpisodeMemory(
            agent_type="hpo_agent",
            objective=request.objective,
            action={"strategy": strategy, "best_config": best_trial.parameters},
            outcome={"best_metrics": best_trial.metrics, "recommendations": recommendations},
            summary=f"{strategy} completed {len(scheduled.trials)} trials",
            experiment_ids=[experiment_id],
            scope=self.memory_scope,
            importance=0.9,
        ))
        return AgentTaskResult(
            status="success",
            summary={
                "strategy": strategy,
                "best_trial_id": best_trial.trial_id,
                "best_parameters": best_trial.parameters,
                "study": scheduled.study.to_dict(),
                "trials": [trial.to_dict() for trial in scheduled.trials],
            },
            metrics=best_trial.metrics,
            recommendations=[recommendations] if recommendations else [],
            artifacts=best_trial.artifacts,
            experiment_ids={"hpo": experiment_id},
            request_id=request.request_id,
        )

    def _trial_executor(self, experiment_id: str):
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
                "experiments_dir": str(self.experiments_dir),
                "runner": runner,
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

    def _planning_advice(
        self,
        request: AgentTaskRequest,
        search_space: SearchSpace,
        selected_strategy: str,
    ) -> Dict[str, Any]:
        prompt = (
            "Return JSON recommendations only. Suggest search-space refinements, model candidates, "
            "parameter adjustments, and diagnostics for a future HPO run. Do not execute tools or "
            "change the selected strategy. "
            f"Selected strategy={selected_strategy}; request={json.dumps(request.to_dict(), ensure_ascii=False)}; "
            f"search_space={json.dumps(search_space.to_dict(), ensure_ascii=False)}"
        )
        try:
            value = json.loads(self._extract_message_content(self.llm.invoke(prompt)))
            return value if isinstance(value, dict) else {"advice": value}
        except Exception as exc:
            return {"advice_error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _build_search_space(value: Optional[Dict[str, Any]]) -> SearchSpace:
        if value:
            return SearchSpace(
                [SearchParameter(**item) for item in value.get("parameters") or []],
                value.get("constraints") or [],
            )
        return SearchSpace([
            SearchParameter("lr", "float", low=1e-5, high=1e-2, scale="log"),
            SearchParameter("batch_size", "categorical", choices=[16, 32, 64]),
            SearchParameter("number_of_epochs", "int", low=5, high=30),
        ])

    @staticmethod
    def _build_budgets(
        value: Optional[List[Dict[str, Any]]],
        requested_strategy: str = "auto",
    ) -> List[TrialBudget]:
        if value:
            return [TrialBudget(**item) for item in value]
        if requested_strategy in {"random_search", "grid_search", "adaptive_search"}:
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
            "llm_role": "planning_recommendations",
            "enable_llm_advisor": self.enable_llm_advisor,
        }


def create_hpo_agent(**kwargs: Any) -> HPOAgent:
    return HPOAgent(**kwargs)


__all__ = ["HPOAgent", "create_hpo_agent"]
