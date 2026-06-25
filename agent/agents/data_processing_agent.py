"""LangGraph data-processing agent with a single structured task interface."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from agent.agents.base_agent import LangGraphAgent
from agent.agents.communication import AgentTaskRequest, AgentTaskResult
from agent.data_processing.workflow import DataProcessingWorkflow
from agent.data_processing.handoff import build_data_handoff
from agent.memory import EpisodeMemory, MemoryScope, MemoryService
from agent.utils import ConfigParser, ExperimentTracker
from agent.utils.path_tool import (
    get_config_file,
    get_data_processing_experiments_dir,
    resolve_config_path,
    resolve_config_value_path,
    resolve_data_path,
)


class DataProcessingAgent(LangGraphAgent):
    """Execute the data lifecycle; the LLM only returns recommendations."""

    action = "optimize_data_processing"

    def __init__(
        self,
        model_name: str = "GLM-4.7",
        temperature: float = 0.2,
        max_iterations: int = 6,
        verbose: bool = True,
        config_path: str = str(get_config_file("train_ecapa_tdnn.yaml")),
        experiments_dir: Optional[Union[str, Path]] = None,
        task_type: str = "speaker_verification",
        model_family: str = "ecapa_tdnn",
        implementation: str = "speechbrain",
        runner: str = "speechbrain",
        enable_llm_advisor: bool = False,
    ) -> None:
        super().__init__(model_name, temperature, max_iterations, verbose)
        self.config_path = str(resolve_config_path(config_path))
        self.experiments_dir = Path(experiments_dir).resolve() if experiments_dir else get_data_processing_experiments_dir()
        self.task_type = task_type
        self.model_family = model_family
        self.implementation = implementation
        self.runner = runner
        self.enable_llm_advisor = enable_llm_advisor
        self.tracker = ExperimentTracker(self.experiments_dir)
        self.memory_service = MemoryService()
        self.memory_scope = MemoryScope(
            agent_type="data_processing",
            task_type=task_type,
            model_family=model_family,
            tags=["data_processing", "langgraph"],
        )

    def run_workflow(self, request: AgentTaskRequest) -> AgentTaskResult:
        started_at = datetime.now()
        config = ConfigParser(self.config_path).load_config(resolve_references=True)
        requested_dataset = request.context.get("dataset_uri") or request.context.get("data_folder")
        data_folder = str(resolve_data_path(requested_dataset or config.get("data_folder")))
        self.memory_scope.dataset_key = data_folder
        output = resolve_config_value_path(config.get("save_folder") or config.get("output_folder"))
        experiment_id = request.experiment_ids.get("data_processing") or self.tracker.create_data_processing_experiment(
            config_path=self.config_path,
            data_folder=data_folder,
            output_folder=str(output) if output else None,
            description="LangGraph data processing run",
            task={"type": self.task_type, "dataset": data_folder},
            model={"family": self.model_family, "implementation": self.implementation, "config_path": self.config_path},
            execution={"runner": self.runner, "output_folder": str(output) if output else None},
        )
        version_path = self.experiments_dir / str(experiment_id) / "dataset_versions" / "dataset_version.json"
        workflow = DataProcessingWorkflow(
            strategy_advisor=self._planning_advice if self.enable_llm_advisor else None,
        )
        processing_config = config.get("data_processing") if isinstance(config.get("data_processing"), dict) else {}
        context_processing = (
            request.context.get("data_processing")
            if isinstance(request.context.get("data_processing"), dict)
            else {}
        )
        requested_operations = (
            request.context.get("data_operations")
            or context_processing.get("operations")
            or processing_config.get("operations")
            or []
        )
        self.tracker.update_data_processing_experiment(
            str(experiment_id),
            status="running",
            extensions={"data_processing": {
                "objective": request.objective,
                "workflow": "langgraph",
                "hpo_feedback": request.context.get("hpo_feedback"),
                "requested_operations": requested_operations,
                "input_dataset_uri": data_folder,
            }},
        )
        state = workflow.run({
            "dataset_uri": data_folder,
            "dataset_type": request.context.get("dataset_type", "auto"),
            "task_type": self.task_type,
            "target_goal": request.context.get("target_goal") or request.objective,
            "output_path": str(version_path),
            "requested_operations": requested_operations,
        })
        if state.get("status") != "success":
            raise RuntimeError(state.get("error") or "data processing workflow failed")
        duration = (datetime.now() - started_at).total_seconds()
        published = state.get("published_version") or {}
        advice = state.get("advice") or {}
        handoff = build_data_handoff(published, experiment_id)
        operation_artifacts = [
            artifact
            for result in state.get("results") or []
            for artifact in result.get("artifacts") or []
        ]
        artifacts = [
            {"type": "dataset_version", "name": "published_dataset", "path": str(version_path)},
            *operation_artifacts,
        ]
        self.tracker.update_data_processing_experiment(
            str(experiment_id),
            status="success",
            duration=duration,
            metrics={"summary": published.get("quality_metrics") or {}},
            artifacts=artifacts,
            extensions={"data_lifecycle": {
                "profile_before": state.get("profile"),
                "plan": state.get("plan"),
                "operation_results": state.get("results"),
                "published_version": published,
                "recommendations": advice,
            }},
        )
        self.memory_service.remember_episode(EpisodeMemory(
            agent_type="data_processing",
            objective=request.objective,
            action={"plan": state.get("plan")},
            outcome={"published_version": published, "recommendations": advice},
            summary=f"Published dataset version {published.get('version')}",
            experiment_ids=[str(experiment_id)],
            scope=self.memory_scope,
            importance=0.8,
        ))
        return AgentTaskResult(
            status="success",
            summary={
                "dataset_profile": state.get("profile"),
                "processing_plan": state.get("plan"),
                "dataset_version": published,
                "data_handoff": handoff,
            },
            metrics={"data_quality": published.get("quality_metrics") or {}},
            artifacts=artifacts,
            recommendations=[advice] if advice else [],
            experiment_ids={"data_processing": str(experiment_id)},
            request_id=request.request_id,
        )

    def _planning_advice(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response = self.llm.invoke(
            "Return JSON recommendations only for future data-processing changes and diagnostics. "
            f"Do not execute operations or change the workflow. Context={json.dumps(state, ensure_ascii=False)}"
        )
        try:
            value = json.loads(self._extract_message_content(response))
            return value if isinstance(value, dict) else {"advice": value}
        except Exception as exc:
            return {"advice_error": f"{type(exc).__name__}: {exc}"}

    def get_execution_details(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "workflow": "langgraph",
            "llm_role": "planning_recommendations",
            "enable_llm_advisor": self.enable_llm_advisor,
        }


def create_data_processing_agent(**kwargs: Any) -> DataProcessingAgent:
    return DataProcessingAgent(**kwargs)


__all__ = ["DataProcessingAgent", "create_data_processing_agent"]
