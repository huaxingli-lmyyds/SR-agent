"""LangGraph coordinator for registered specialized agents."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.agents.base_agent import AdvisoryAgentBase
from agent.agents.communication import AgentTaskRequest, MessageService
from agent.agents.coordination import (
    AgentRegistration,
    AgentRegistry,
    CompletionDecision,
    CompletionPolicy,
    TaskDispatcher,
)
from agent.agents.orchestration_workflow import OrchestrationWorkflow
from agent.tasks import get_task_adapter
from agent.memory import EpisodeMemory, MemoryScope, MemoryService
from agent.utils import ConfigParser, ExperimentTracker
from agent.utils.path_tool import (
    get_config_file,
    get_data_processing_experiments_dir,
    get_hpo_experiments_dir,
    get_manage_experiments_dir,
    resolve_config_path,
    resolve_config_value_path,
    resolve_data_path,
)


@dataclass
class OrchestrationResult:
    experiment_id: str
    status: str
    rounds: int
    completion: Dict[str, Any]
    task_results: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CoordinatorAgent(AdvisoryAgentBase):
    """Coordinate every registered agent through a dynamic LangGraph workflow."""

    def __init__(
        self,
        model_name: str = "GLM-4.7",
        temperature: float = 0.2,
        max_iterations: int = 10,
        data_iterations: int = 6,
        verbose: bool = True,
        config_path: str = str(get_config_file("train_ecapa_tdnn.yaml")),
        task_type: str = "speaker_verification",
        model_family: str = "ecapa_tdnn",
        implementation: str = "speechbrain",
        runner: str = "speechbrain",
        registry: Optional[AgentRegistry] = None,
        completion_policy: Optional[CompletionPolicy] = None,
        enable_llm_advisor: bool = False,
    ) -> None:
        super().__init__(model_name, temperature, max_iterations, verbose)
        self.config_path = str(resolve_config_path(config_path))
        self.data_iterations = data_iterations
        self.task_type = task_type
        self.model_family = model_family
        self.implementation = implementation
        self.runner = runner
        self.enable_llm_advisor = enable_llm_advisor
        self.manage_tracker = ExperimentTracker(get_manage_experiments_dir())
        self.memory_service = MemoryService()
        self.memory_scope = MemoryScope(
            agent_type="coordinator",
            task_type=task_type,
            model_family="multi_agent_optimization",
            tags=["orchestration", "langgraph"],
        )
        self.registry = registry or self._build_default_registry()
        self.completion_policy = completion_policy or CompletionPolicy(
            required_agent_types={item["agent_type"] for item in self.registry.describe()}
        )
        self._manage_experiment_id: Optional[str] = None
        self._message_service: Optional[MessageService] = None
        self._linked_experiments: Dict[str, List[str]] = {}
        self._run_started_at: Optional[datetime] = None

    def _build_default_registry(self) -> AgentRegistry:
        from agent.agents.data_processing_agent import create_data_processing_agent
        from agent.agents.hpo_agent import create_hpo_agent

        registry = AgentRegistry()
        registry.register(AgentRegistration(
            "data_processing_agent",
            ("optimize_data_processing",),
            lambda: create_data_processing_agent(
                model_name=self.model_name,
                temperature=self.temperature,
                max_iterations=self.data_iterations,
                verbose=self.verbose,
                config_path=self.config_path,
                experiments_dir=get_data_processing_experiments_dir(),
                task_type=self.task_type,
                model_family=self.model_family,
                implementation=self.implementation,
                runner=self.runner,
                enable_llm_advisor=self.enable_llm_advisor,
            ),
            "Inspect, validate, and publish datasets.",
        ))
        registry.register(AgentRegistration(
            "hpo_agent",
            ("optimize_hyperparameters",),
            lambda: create_hpo_agent(
                model_name=self.model_name,
                temperature=self.temperature,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
                config_path=self.config_path,
                experiments_dir=get_hpo_experiments_dir(),
                task_type=self.task_type,
                model_family=self.model_family,
                implementation=self.implementation,
                runner=self.runner,
                enable_llm_advisor=self.enable_llm_advisor,
            ),
            "Run a model-agnostic HPO study.",
        ))
        return registry

    def register_agent(self, registration: AgentRegistration, *, required: bool = True) -> None:
        """Public extension point for adding specialized agents before run()."""
        self.registry.register(registration)
        if required:
            self.completion_policy.required_agent_types.add(registration.agent_type)

    def run(
        self,
        objective: str = "Improve the primary model metric",
        context: Optional[Dict[str, Any]] = None,
        budget: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        self._run_started_at = datetime.now()
        config = ConfigParser(self.config_path).load_config(resolve_references=True)
        run_context = context or {}
        requested_dataset = run_context.get("dataset_uri") or run_context.get("data_folder")
        data_folder = str(resolve_data_path(requested_dataset or config.get("data_folder")))
        output = resolve_config_value_path(config.get("output_folder"))
        task_adapter = get_task_adapter(self.task_type)
        self._manage_experiment_id = self.manage_tracker.create_orchestration_experiment(
            config_path=self.config_path,
            data_folder=data_folder,
            output_folder=str(output) if output else None,
            description="LangGraph orchestration run",
            task={
                "type": self.task_type,
                "dataset": data_folder,
                "primary_metric": task_adapter.primary_metric,
                "metric_mode": task_adapter.metric_mode,
            },
            model={"family": "multi_agent_optimization", "implementation": "langgraph", "config_path": self.config_path},
            execution={"runner": "langgraph", "output_folder": str(output) if output else None},
        )
        self._linked_experiments = {"manage": [self._manage_experiment_id]}
        self._message_service = MessageService(self._manage_experiment_id)
        dispatcher = TaskDispatcher(self.registry, self._message_service)
        workflow_context = {
            **run_context,
            "target_goal": run_context.get("target_goal") or "validate and prepare the dataset",
            "primary_metric": task_adapter.primary_metric,
            "metric_mode": task_adapter.metric_mode,
            "config_path": self.config_path,
        }
        workflow_budget = {
            "max_runs": self.data_iterations,
            "max_training_runs": self.max_iterations,
            **(budget or {}),
        }

        def request_factory(current_context: Dict[str, Any], current_budget: Dict[str, Any]) -> AgentTaskRequest:
            experiment_ids = self._latest_experiment_ids()
            for result in (current_context.get("previous_results") or {}).values():
                for category, value in (result.get("experiment_ids") or {}).items():
                    experiment_ids[category] = value
            return AgentTaskRequest(
                action="",
                objective=objective,
                context=current_context,
                budget=current_budget,
                experiment_ids=experiment_ids,
            )

        workflow = OrchestrationWorkflow(
            self.registry,
            dispatcher,
            self.completion_policy,
            request_factory,
            advisor=self._coordination_advisor if self.enable_llm_advisor else None,
        )
        self.manage_tracker.update_orchestration_experiment(
            self._manage_experiment_id,
            status="running",
            extensions={"orchestration": {"workflow": "langgraph", "registered_agents": self.registry.describe()}},
        )
        state = workflow.run(workflow_context, workflow_budget)
        records = state.get("records") or []
        for record in records:
            self._merge_experiment_ids(record.result.get("experiment_ids") or {})
        completion = CompletionDecision(**state["completion"])
        status = "success" if completion.complete else "failed"
        duration = (datetime.now() - self._run_started_at).total_seconds()
        compact_records = [self._compact_task_record(record) for record in records]
        compact_messages = [
            self._compact_message(message.to_dict())
            for message in self._message_service.history()
        ]
        orchestration_summary = self._orchestration_summary(compact_records)
        self.manage_tracker.update_orchestration_experiment(
            self._manage_experiment_id,
            status=status,
            duration=duration,
            linked_experiments=self._linked_experiments,
            agent_messages=compact_messages,
            metrics=orchestration_summary.get("metrics", {}),
            extensions={"orchestration": {
                "workflow": "langgraph",
                "advice": state.get("advice") or {},
                "task_results": compact_records,
                "completion": completion.to_dict(),
                "linked_experiment_summary": orchestration_summary,
            }},
        )
        self.memory_service.remember_episode(EpisodeMemory(
            agent_type="coordinator",
            objective=objective,
            action={"task_results": compact_records},
            outcome={
                "completion": completion.to_dict(),
                "linked_experiments": self._linked_experiments,
                "linked_experiment_summary": orchestration_summary,
            },
            summary="LangGraph orchestration completed",
            experiment_ids=[self._manage_experiment_id],
            scope=self.memory_scope,
            status=status,
            importance=0.9,
        ))
        return OrchestrationResult(
            experiment_id=self._manage_experiment_id,
            status=status,
            rounds=len(records),
            completion=completion.to_dict(),
            task_results=compact_records,
        )

    def _coordination_advisor(self, agents: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        compact_context = self._compact_request_context(context)
        prompt = json.dumps({
            "schema": {
                "diagnostics": [],
                "risks": [],
                "notes": [],
            },
            "rules": [
                "Return raw JSON only.",
                "Do not use markdown, code fences, comments, or explanatory text.",
                "Do not choose agent order, assign tasks, or execute tools.",
                "Only provide diagnostics about the current multi-agent run.",
            ],
            "agents": agents,
            "context": compact_context,
        }, ensure_ascii=False, separators=(",", ":"))
        response = self.llm.invoke(prompt)
        try:
            value = json.loads(self._extract_message_content(response))
            return value if isinstance(value, dict) else {}
        except Exception as exc:
            return {"advice_error": f"{type(exc).__name__}: {exc}"}

    def _latest_experiment_ids(self) -> Dict[str, Any]:
        return {key: values[-1] if values else None for key, values in self._linked_experiments.items()}

    def _merge_experiment_ids(self, values: Dict[str, Any]) -> None:
        for category, raw_ids in values.items():
            ids = raw_ids if isinstance(raw_ids, list) else [raw_ids]
            target = self._linked_experiments.setdefault(category, [])
            for experiment_id in ids:
                if experiment_id and str(experiment_id) not in target:
                    target.append(str(experiment_id))

    def _compact_task_record(self, record: Any) -> Dict[str, Any]:
        result = record.result or {}
        request = record.request or {}
        return {
            "agent_type": record.agent_type,
            "action": record.action,
            "status": result.get("status", "unknown"),
            "request_id": result.get("request_id") or request.get("request_id"),
            "started_at": record.started_at,
            "completed_at": record.completed_at,
            "experiment_ids": result.get("experiment_ids") or {},
            "error": result.get("error"),
            "summary": self._compact_agent_summary(record.agent_type, result.get("summary") or {}),
            "metrics": result.get("metrics") or {},
            "artifact_count": len(result.get("artifacts") or []),
            "recommendation_count": len(result.get("recommendations") or []),
        }

    def _compact_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        compact = dict(message)
        payload = dict(compact.get("payload") or {})
        message_type = compact.get("message_type")
        if message_type == "task.request":
            compact["payload"] = {
                "action": payload.get("action"),
                "objective": payload.get("objective"),
                "budget": payload.get("budget") or {},
                "experiment_ids": payload.get("experiment_ids") or {},
                "context": self._compact_request_context(payload.get("context") or {}),
                "request_id": payload.get("request_id"),
            }
        elif message_type in {"task.result", "error"}:
            compact["payload"] = self._compact_result_payload(payload)
        else:
            compact["payload"] = payload
        return compact

    def _compact_result_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": payload.get("status"),
            "summary": self._compact_agent_summary(
                str(payload.get("agent_type") or ""),
                payload.get("summary") or {},
            ),
            "metrics": payload.get("metrics") or {},
            "experiment_ids": payload.get("experiment_ids") or {},
            "artifact_count": len(payload.get("artifacts") or []),
            "recommendation_count": len(payload.get("recommendations") or []),
            "error": payload.get("error"),
            "request_id": payload.get("request_id"),
        }

    @staticmethod
    def _compact_request_context(context: Dict[str, Any]) -> Dict[str, Any]:
        allowed = (
            "dataset_uri",
            "data_folder",
            "target_goal",
            "primary_metric",
            "metric_mode",
            "strategy",
            "runtime_options",
            "data_handoff",
        )
        compact = {key: context[key] for key in allowed if key in context}
        if "previous_results" in context:
            compact["previous_result_agents"] = sorted((context.get("previous_results") or {}).keys())
        if "search_space" in context:
            search_space = context.get("search_space") or {}
            compact["search_space"] = {
                "parameter_count": len(search_space.get("parameters") or []),
                "constraint_count": len(search_space.get("constraints") or []),
            }
        if "budgets" in context:
            compact["budget_count"] = len(context.get("budgets") or [])
        return compact

    @staticmethod
    def _compact_agent_summary(agent_type: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        if not summary:
            return {}
        if agent_type == "hpo_agent" or "best_trial_id" in summary:
            campaign = summary.get("campaign") or {}
            return {
                "strategy": summary.get("strategy"),
                "best_trial_id": summary.get("best_trial_id"),
                "best_parameters": summary.get("best_parameters") or {},
                "campaign": {
                    "campaign_id": campaign.get("campaign_id"),
                    "status": campaign.get("status"),
                    "stop_reason": campaign.get("stop_reason"),
                    "study_count": len(campaign.get("study_summaries") or []),
                    "best_experiment_id": campaign.get("best_experiment_id"),
                    "best_value": campaign.get("best_value"),
                },
                "studies": summary.get("studies") or [],
                "data_handoff": summary.get("data_handoff") or {},
            }
        if agent_type == "data_processing_agent" or "dataset_version" in summary:
            version = summary.get("dataset_version") or {}
            return {
                "dataset_version": {
                    "dataset_id": version.get("dataset_id"),
                    "version": version.get("version"),
                    "source_uri": version.get("source_uri"),
                    "quality_metrics": version.get("quality_metrics") or {},
                },
                "data_handoff": summary.get("data_handoff") or {},
            }
        return {
            key: summary[key]
            for key in ("status", "message", "experiment_id")
            if key in summary
        }

    @staticmethod
    def _orchestration_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary_metrics: Dict[str, Any] = {
            "agent_count": len(records),
            "failed_agent_count": sum(1 for item in records if item.get("status") == "failed"),
        }
        metrics: Dict[str, Any] = {"summary": summary_metrics}
        hpo_record = next((item for item in records if item.get("agent_type") == "hpo_agent"), None)
        if hpo_record:
            hpo_metrics = hpo_record.get("metrics") or {}
            summary = hpo_record.get("summary") or {}
            summary_metrics["hpo_status"] = hpo_record.get("status")
            best_metrics = dict(hpo_metrics)
            if summary.get("best_trial_id"):
                best_metrics["trial_id"] = summary["best_trial_id"]
            if best_metrics:
                metrics["best"] = best_metrics
        return {
            "metrics": metrics,
            "agents": [
                {
                    "agent_type": item.get("agent_type"),
                    "status": item.get("status"),
                    "experiment_ids": item.get("experiment_ids") or {},
                }
                for item in records
            ],
        }

    def get_execution_details(self) -> Dict[str, Any]:
        return {
            "workflow": "langgraph",
            "llm_role": "coordination_advisor",
            "registered_agents": self.registry.describe(),
        }


__all__ = ["CoordinatorAgent", "OrchestrationResult"]
