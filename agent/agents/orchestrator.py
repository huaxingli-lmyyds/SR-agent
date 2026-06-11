"""Registry-driven coordinator for model optimization agents."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from agent.agents.base_agent import BaseLangChainAgent
from agent.agents.communication import AgentTaskRequest, MessageService
from agent.agents.coordination import (
    AgentRegistration,
    AgentRegistry,
    CompletionDecision,
    CompletionPolicy,
    TaskDispatcher,
    TaskExecutionRecord,
)
from agent.core.adapters import get_task_adapter
from agent.memory import EpisodeMemory, MemoryQuery, MemoryScope, MemoryService
from agent.tools.experiment_history_tools import (
    CompareOrchestrationExperiments,
    GetOrchestrationExperimentResults,
    ListOrchestrationExperiments,
)
from agent.utils import ConfigParser, ExperimentTracker
from agent.utils.logger import AgentLogger
from agent.utils.path_tool import (
    get_config_file,
    get_data_processing_experiments_dir,
    get_hpo_experiments_dir,
    get_logs_dir,
    get_manage_experiments_dir,
    resolve_config_path,
    resolve_config_value_path,
    resolve_data_path,
)


@dataclass
class OrchestrationResult:
    """Serializable top-level result independent of specialized agent classes."""

    experiment_id: str
    status: str
    rounds: int
    completion: Dict[str, Any]
    task_results: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CoordinatorAgent(BaseLangChainAgent):
    """Coordinate registered agents through one structured dispatch interface."""

    def __init__(
        self,
        model_name: str = "GLM-4.7",
        temperature: float = 0.2,
        max_iterations: int = 10,
        data_iterations: int = 6,
        max_rounds: int = 3,
        verbose: bool = True,
        config_path: str = str(get_config_file("train_ecapa_tdnn.yaml")),
        task_type: str = "speaker_verification",
        model_family: str = "ecapa_tdnn",
        implementation: str = "speechbrain",
        runner: str = "speechbrain",
        registry: Optional[AgentRegistry] = None,
        completion_policy: Optional[CompletionPolicy] = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        self.config_path = str(resolve_config_path(config_path))
        self.data_iterations = data_iterations
        self.max_rounds = max_rounds
        self.task_type = task_type
        self.model_family = model_family
        self.implementation = implementation
        self.runner = runner
        self.manage_tracker = ExperimentTracker(get_manage_experiments_dir())
        self.agent_logger = AgentLogger(get_logs_dir() / "agent_manage.log")
        self.memory_service = MemoryService()
        self.memory_scope = MemoryScope(
            agent_type="coordinator",
            task_type=self.task_type,
            model_family="multi_agent_optimization",
            tags=["orchestration", "workflow", "optimization"],
        )
        self.registry = registry or self._build_default_registry()
        self.completion_policy = completion_policy or CompletionPolicy(
            required_agent_types={
                item["agent_type"] for item in self.registry.describe()
            }
        )

        self._manage_experiment_id: Optional[str] = None
        self._message_service: Optional[MessageService] = None
        self._dispatcher: Optional[TaskDispatcher] = None
        self._target_eer = 0.02
        self._custom_objective: Optional[str] = None
        self._task_records: List[TaskExecutionRecord] = []
        self._latest_results: Dict[str, Dict[str, Any]] = {}
        self._linked_experiments: Dict[str, List[str]] = {}
        self._run_started_at: Optional[datetime] = None

        self.tools = self._load_tools()
        self.system_prompt = self._create_system_prompt()
        self.agent = self._build_agent(
            tools=self.tools,
            system_prompt=self.system_prompt,
            middleware=self.agent_logger.build_middleware(),
        )

    def _build_default_registry(self) -> AgentRegistry:
        """Register built-in agents; callers may inject a different registry."""
        from agent.agents.data_processing_agent import create_data_processing_agent
        from agent.agents.hpo_agent import create_hpo_agent

        registry = AgentRegistry()
        registry.register(AgentRegistration(
            agent_type="data_processing_agent",
            actions=("optimize_data_processing",),
            description="Inspect, prepare, validate, and publish datasets.",
            factory=lambda: create_data_processing_agent(
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
            ),
        ))
        registry.register(AgentRegistration(
            agent_type="hpo_agent",
            actions=("optimize_hyperparameters",),
            description="Plan and execute model-agnostic hyperparameter studies.",
            factory=lambda: create_hpo_agent(
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
            ),
        ))
        return registry

    def _reset_runtime_state(self) -> None:
        self._manage_experiment_id = None
        self._message_service = None
        self._dispatcher = None
        self._task_records = []
        self._latest_results = {}
        self._linked_experiments = {}
        self._run_started_at = None

    def _ensure_context(self) -> None:
        if self._manage_experiment_id is not None:
            return
        config = ConfigParser(self.config_path).load_config(resolve_references=True)
        data_folder = str(resolve_data_path(config.get("data_folder")))
        output = resolve_config_value_path(config.get("output_folder"))
        self.memory_scope.dataset_key = data_folder
        task_adapter = get_task_adapter(self.task_type)
        self._manage_experiment_id = self.manage_tracker.create_orchestration_experiment(
            config_path=self.config_path,
            data_folder=data_folder,
            output_folder=str(output) if output else None,
            description="registry-driven orchestrated run",
            task={
                "type": self.memory_scope.task_type,
                "dataset": data_folder,
                "primary_metric": task_adapter.primary_metric,
                "metric_mode": task_adapter.metric_mode,
            },
            model={
                "family": "multi_agent_optimization",
                "implementation": "coordinator",
                "config_path": self.config_path,
            },
            execution={"runner": "coordinator", "output_folder": str(output) if output else None},
            extra_fields={
                "extensions": {
                    "orchestration": {
                        "target_eer": self._target_eer,
                        "custom_objective": self._custom_objective,
                        "registered_agents": self.registry.describe(),
                    }
                }
            },
        )
        self._linked_experiments = {"manage": [self._manage_experiment_id]}
        self._message_service = MessageService(session_id=self._manage_experiment_id)
        self._dispatcher = TaskDispatcher(self.registry, self._message_service)
        self._sync_manage_record(status="running", last_action="initialized")

    @staticmethod
    def _parse_json(value: Optional[str], field_name: str) -> Dict[str, Any]:
        if not value:
            return {}
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError(f"{field_name} must contain a JSON object")
        return parsed

    def _request_experiment_ids(self) -> Dict[str, Any]:
        return {
            key: values[-1] if values else None
            for key, values in self._linked_experiments.items()
        }

    def _merge_experiment_ids(self, experiment_ids: Dict[str, Any]) -> None:
        for category, raw_ids in experiment_ids.items():
            values = raw_ids if isinstance(raw_ids, list) else [raw_ids]
            target = self._linked_experiments.setdefault(category, [])
            for experiment_id in values:
                if experiment_id and experiment_id not in target:
                    target.append(str(experiment_id))

    def _dispatch(
        self,
        agent_type: str,
        action: str,
        objective: str,
        context: Dict[str, Any],
        budget: Dict[str, Any],
    ) -> TaskExecutionRecord:
        self._ensure_context()
        if sum(record.agent_type == agent_type for record in self._task_records) >= self.max_rounds:
            raise RuntimeError(f"max_rounds reached for agent '{agent_type}'")
        assert self._dispatcher is not None
        request = AgentTaskRequest(
            action=action,
            objective=objective,
            context={
                **context,
                "config_path": self.config_path,
                "previous_results": self._latest_results,
            },
            budget=budget,
            experiment_ids=self._request_experiment_ids(),
        )
        record = self._dispatcher.dispatch(agent_type, request)
        self._task_records.append(record)
        self._latest_results[agent_type] = record.result
        self._merge_experiment_ids(record.result.get("experiment_ids") or {})
        self._sync_manage_record(
            status="running",
            last_action=f"{agent_type}:{action}:{record.status}",
        )
        return record

    def _create_system_prompt(self) -> str:
        return (
            "You are the coordinator of an extensible model optimization system. "
            "Use DispatchAgentTask for all specialized work. Select agents and actions "
            "from ListRegisteredAgents, pass prior results through structured context, "
            "and do not invent execution results. A run is only successful when its "
            "completion policy is satisfied."
        )

    def _load_tools(self):
        @tool
        def ListRegisteredAgents() -> str:
            """List available agent types, actions, and descriptions."""
            return json.dumps(self.registry.describe(), ensure_ascii=False)

        @tool
        def DispatchAgentTask(
            agent_type: str,
            action: str,
            objective: str,
            context_json: Optional[str] = None,
            budget_json: Optional[str] = None,
        ) -> str:
            """Dispatch one structured task to any registered specialized agent."""
            try:
                record = self._dispatch(
                    agent_type=agent_type,
                    action=action,
                    objective=objective,
                    context=self._parse_json(context_json, "context_json"),
                    budget=self._parse_json(budget_json, "budget_json"),
                )
                return json.dumps(record.to_dict(), ensure_ascii=False, default=str)
            except Exception as exc:
                return json.dumps(
                    {"status": "failed", "agent_type": agent_type, "action": action, "error": str(exc)},
                    ensure_ascii=False,
                )

        @tool
        def CheckCompletion() -> str:
            """Evaluate the configured completion policy against dispatched tasks."""
            return json.dumps(
                self.completion_policy.evaluate(self._task_records).to_dict(),
                ensure_ascii=False,
            )

        return [
            ListRegisteredAgents,
            DispatchAgentTask,
            CheckCompletion,
            CompareOrchestrationExperiments,
            GetOrchestrationExperimentResults,
            ListOrchestrationExperiments,
        ]

    def _completion_decision(self) -> CompletionDecision:
        return self.completion_policy.evaluate(self._task_records)

    def _sync_manage_record(
        self,
        status: Optional[str] = None,
        last_action: Optional[str] = None,
        final_answer: Optional[str] = None,
        completion: Optional[CompletionDecision] = None,
    ) -> None:
        if self._manage_experiment_id is None:
            return
        task_results = [record.to_dict() for record in self._task_records]
        messages = (
            [message.to_dict() for message in self._message_service.history()]
            if self._message_service else []
        )
        duration = None
        if status in {"success", "failed", "cancelled"} and self._run_started_at:
            duration = (datetime.now() - self._run_started_at).total_seconds()
        state = {
            "last_action": last_action,
            "target_eer": self._target_eer,
            "custom_objective": self._custom_objective,
            "rounds": len(task_results),
            "latest_results": self._latest_results,
            "task_results": task_results,
            "completion": completion.to_dict() if completion else None,
            "final_answer": final_answer,
        }
        self.manage_tracker.update_orchestration_experiment(
            experiment_id=self._manage_experiment_id,
            status=status,
            duration=duration,
            linked_experiments=self._linked_experiments,
            agent_messages=messages,
            extensions={"orchestration": state},
        )
        self.memory_service.update_working_state(
            self._manage_experiment_id,
            {
                "status": status or "running",
                "current_stage": last_action,
                "linked_experiments": self._linked_experiments,
                "task_results": task_results,
                "completion": completion.to_dict() if completion else None,
            },
        )

    def _persist_episode(self, status: str, summary: str) -> None:
        if not self._manage_experiment_id:
            return
        try:
            self.memory_service.remember_episode(EpisodeMemory(
                agent_type="coordinator",
                objective=self._custom_objective or f"target_eer={self._target_eer}",
                action={"task_results": [record.to_dict() for record in self._task_records]},
                outcome={"linked_experiments": self._linked_experiments},
                summary=summary,
                experiment_ids=[self._manage_experiment_id],
                scope=self.memory_scope,
                status=status,
                importance=0.9,
            ))
        except Exception as exc:
            self.agent_logger.append(f"memory_update_failed error={exc}")

    def _memory_context(self) -> str:
        return self.memory_service.format_context(
            MemoryQuery(task_type=self.memory_scope.task_type, visibility="shared", limit=5),
            max_chars=1200,
        )

    def run(
        self,
        target_eer: float = 0.02,
        custom_objective: Optional[str] = None,
    ) -> OrchestrationResult:
        self._reset_runtime_state()
        self._run_started_at = datetime.now()
        self._target_eer = target_eer
        self._custom_objective = custom_objective
        self._ensure_context()

        objective = (
            "Coordinate the registered agents to improve the model. "
            f"Target EER: {target_eer}. Maximum rounds per agent: {self.max_rounds}. "
            f"Additional objective: {custom_objective or 'none'}. "
            "List registered agents, dispatch the required work, then check completion.\n"
            f"{self._memory_context()}"
        )
        self.agent_logger.append(f"objective={objective}")
        try:
            response = self._invoke(objective)
            messages = response.get("messages", [])
            final_answer = self._extract_message_content(messages[-1]) if messages else ""
            completion = self._completion_decision()
            tracker_status = "success" if completion.complete else "failed"
            self._sync_manage_record(
                status=tracker_status,
                last_action="complete" if completion.complete else "completion_rejected",
                final_answer=final_answer,
                completion=completion,
            )
            self._persist_episode(tracker_status, final_answer or "; ".join(completion.reasons))
            return OrchestrationResult(
                experiment_id=self._manage_experiment_id or "",
                status=completion.status,
                rounds=len(self._task_records),
                completion=completion.to_dict(),
                task_results=[record.to_dict() for record in self._task_records],
            )
        except Exception as exc:
            completion = CompletionDecision(False, "failed", [str(exc)])
            self._sync_manage_record(
                status="failed",
                last_action="error",
                final_answer=str(exc),
                completion=completion,
            )
            self._persist_episode("failed", str(exc))
            raise


class OrchestratedPipeline(CoordinatorAgent):
    """Backward-compatible alias."""


ManagerAgent = CoordinatorAgent


__all__ = ["CoordinatorAgent", "ManagerAgent", "OrchestratedPipeline", "OrchestrationResult"]
