"""
Tool-based orchestrator for speaker recognition optimization.

The coordinator is a real LangChain tool-using agent. It can launch the
specialized data-processing and HPO sub-agents, inspect orchestration history,
and keep the top-level experiment record in the manage directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

from langchain_core.tools import tool

from agent.agents.communication import AgentTaskRequest, MessageService
from agent.agents.base_agent import BaseLangChainAgent
from agent.agents.data_processing_agent import create_data_processing_agent, DataProcessingResult
from agent.agents.hpo_agent import create_hpo_agent, OptimizationResult
from agent.memory import EpisodeMemory, MemoryQuery, MemoryScope, MemoryService
from agent.tools.experiment_history_tools import (
    CompareOrchestrationExperiments,
    GetOrchestrationExperimentResults,
    ListOrchestrationExperiments,
)
from agent.utils import ExperimentTracker, ConfigParser
from agent.utils.logger import AgentLogger
from agent.utils.path_tool import (
    get_data_processing_experiments_dir,
    get_hpo_experiments_dir,
    get_manage_experiments_dir,
    get_agent_dir,
    get_config_file,
    get_logs_dir,
    resolve_config_path,
    resolve_config_value_path,
    resolve_data_path,
)


@dataclass
class OrchestrationResult:
    experiment_id: str
    rounds: int
    data_processing: List[DataProcessingResult]
    hpo: List[OptimizationResult]


class CoordinatorAgent(BaseLangChainAgent):
    """Tool-based top-level coordinator for the speaker recognition system."""

    def __init__(
        self,
        model_name: str = "GLM-4.7",
        temperature: float = 0.2,
        max_iterations: int = 10,
        data_iterations: int = 6,
        max_rounds: int = 3,
        verbose: bool = True,
        config_path: str = str(get_config_file("train_ecapa_tdnn.yaml")),
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

        self.manage_tracker = ExperimentTracker(get_manage_experiments_dir())
        self.data_tracker = ExperimentTracker(get_data_processing_experiments_dir())
        self.agent_logger = AgentLogger(get_logs_dir() / "agent_manage.log")
        self.memory_service = MemoryService()
        self.memory_scope = MemoryScope(
            agent_type="coordinator",
            task_type="speaker_verification",
            model_family="multi_agent_optimization",
            tags=["orchestration", "workflow", "optimization"],
        )

        self._manage_experiment_id: Optional[str] = None
        self._data_experiment_id: Optional[str] = None
        self._message_service: Optional[MessageService] = None
        self._target_eer: float = 0.02
        self._custom_objective: Optional[str] = None
        self._latest_data_summary: Optional[Dict[str, Any]] = None
        self._latest_hpo_result: Optional[Dict[str, Any]] = None
        self._decision_history: List[str] = []
        self._data_results: List[DataProcessingResult] = []
        self._hpo_results: List[OptimizationResult] = []
        self._linked_experiments: Dict[str, Any] = {}
        self._run_started_at: Optional[datetime] = None

        self.tools = self._load_tools()
        self.system_prompt = self._create_system_prompt()
        self.agent = self._build_agent(
            tools=self.tools,
            system_prompt=self.system_prompt,
            middleware=self.agent_logger.build_middleware(),
        )

    def _load_config(self) -> Dict[str, Any]:
        parser = ConfigParser(self.config_path)
        return parser.load_config(resolve_references=True)

    def _get_memory_context(self) -> str:
        return self.memory_service.format_context(
            MemoryQuery(
                task_type=self.memory_scope.task_type,
                dataset_key=self.memory_scope.dataset_key,
                visibility="shared",
                limit=5,
            ),
            max_chars=1200,
        )

    def _persist_episode(self, status: str, summary: str) -> None:
        if self._manage_experiment_id is None:
            return
        try:
            self.memory_service.remember_episode(EpisodeMemory(
                agent_type="coordinator",
                objective=self._custom_objective or f"target_eer={self._target_eer}",
                action={
                    "decision_history": self._decision_history,
                    "rounds": max(len(self._data_results), len(self._hpo_results)),
                },
                outcome={
                    "latest_data_summary": self._latest_data_summary,
                    "latest_hpo_result": self._latest_hpo_result,
                    "linked_experiments": self._linked_experiments,
                },
                summary=summary,
                experiment_ids=[self._manage_experiment_id],
                scope=self.memory_scope,
                status=status,
                importance=0.9,
            ))
        except Exception as exc:
            self.agent_logger.append(f"memory_update_failed error={exc}")

    def _reset_runtime_state(self) -> None:
        self._manage_experiment_id = None
        self._data_experiment_id = None
        self._message_service = None
        self._latest_data_summary = None
        self._latest_hpo_result = None
        self._decision_history = []
        self._data_results = []
        self._hpo_results = []
        self._linked_experiments = {}
        self._run_started_at = None

    def _ensure_orchestration_context(self) -> None:
        if self._manage_experiment_id is not None and self._data_experiment_id is not None:
            return

        config_data = self._load_config()
        data_folder = str(resolve_data_path(config_data.get("data_folder")))
        self.memory_scope.dataset_key = data_folder
        output_folder = config_data.get("output_folder")
        resolved_output = resolve_config_value_path(output_folder)

        if self._manage_experiment_id is None:
            self._manage_experiment_id = self.manage_tracker.create_orchestration_experiment(
                config_path=self.config_path,
                data_folder=data_folder,
                output_folder=str(resolved_output) if resolved_output else None,
                description="orchestrated run",
                extra_fields={
                    "extensions": {
                        "orchestration": {
                            "target_eer": self._target_eer,
                            "custom_objective": self._custom_objective,
                        }
                    }
                },
            )

        if self._data_experiment_id is None:
            self._data_experiment_id = self.data_tracker.create_data_processing_experiment(
                config_path=self.config_path,
                data_folder=data_folder,
                output_folder=str(resolved_output) if resolved_output else None,
                description="data processing run",
            )

        if self._message_service is None and self._manage_experiment_id is not None:
            self._message_service = MessageService(session_id=self._manage_experiment_id)

        self._linked_experiments = {
            "manage": self._manage_experiment_id,
            "data_processing": self._data_experiment_id,
            "hpo": [],
        }
        self.manage_tracker.update_orchestration_experiment(
            experiment_id=self._manage_experiment_id,
            status="running",
            linked_experiments=self._linked_experiments,
        )
        self.memory_service.update_working_state(
            self._manage_experiment_id,
            {
                "status": "running",
                "current_stage": "initialized",
                "target_eer": self._target_eer,
                "custom_objective": self._custom_objective,
                "linked_experiments": self._linked_experiments,
                "rounds": 0,
            },
        )

    def _create_system_prompt(self) -> str:
        return f"""你是声纹识别系统的统筹智能体。

            你的任务是协调两个专门子智能体：
            - 数据处理智能体：负责数据准备、划分和质量优化
            - 超参数智能体：负责训练、评估和超参数优化

            你必须通过工具完成工作，不要直接编造结果。

            可用工具策略：
            1. 先查看统筹记录，了解当前状态
            2. 根据需要调用数据处理工具，获得最新数据摘要
            3. 再调用 HPO 工具，结合数据摘要进行训练/评估优化
            4. 你可以在两者之间反复切换，但不要超过 {self.max_rounds} 轮核心迭代
            5. 完成后给出最终总结，必须包含 manage / dp / hpo 的实验 ID

            当前目标：将声纹识别系统优化到目标 EER 以下，并保持数据处理与训练链路可追踪。
            请只通过工具推进，不要跳过实验记录。"""

    def _load_tools(self):
        @tool
        def LaunchDataProcessingRound(
            target_goal: Optional[str] = None,
            custom_objective: Optional[str] = None,
            hpo_feedback_json: Optional[str] = None,
        ) -> str:
            """启动一轮数据处理智能体，并同步写入统筹记录。"""
            self._ensure_orchestration_context()

            feedback: Optional[Dict[str, Any]] = None
            if hpo_feedback_json:
                try:
                    feedback = json.loads(hpo_feedback_json)
                except json.JSONDecodeError:
                    feedback = {"raw": hpo_feedback_json}

            data_agent = create_data_processing_agent(
                model_name=self.model_name,
                temperature=self.temperature,
                max_iterations=self.data_iterations,
                verbose=self.verbose,
                config_path=self.config_path,
                experiments_dir=get_data_processing_experiments_dir(),
            )
            request = AgentTaskRequest(
                action="optimize_data_processing",
                objective=custom_objective or target_goal or "提升数据质量并保持训练/验证分布稳定",
                context={
                    "target_goal": target_goal,
                    "hpo_feedback": feedback,
                    "config_path": self.config_path,
                },
                budget={"max_runs": 1, "max_iterations": self.data_iterations},
                experiment_ids={
                    "manage": self._manage_experiment_id,
                    "data_processing": self._data_experiment_id,
                },
            )
            request_message = self._message_service.send_task(
                "coordinator", "data_processing_agent", request
            )
            task_result = data_agent.execute_task(request)
            self._message_service.send_result(
                "data_processing_agent", "coordinator", request_message, task_result
            )
            if task_result.status == "failed" or task_result.runtime_result is None:
                self._sync_manage_record(status="running", last_action="data_processing_failed")
                return task_result.to_json()
            result = task_result.runtime_result

            self._data_results.append(result)
            self._latest_data_summary = result.data_summary
            self._decision_history.append("data_processing")
            self._linked_experiments["data_processing"] = result.experiment_id

            self._sync_manage_record(status="running", last_action="data_processing")
            self.agent_logger.append(
                "data_round_complete "
                f"manage_experiment_id={self._manage_experiment_id} "
                f"data_experiment_id={self._data_experiment_id}"
            )
            return task_result.to_json()

        @tool
        def LaunchHPORound(
            target_eer: float = 0.02,
            custom_objective: Optional[str] = None,
        ) -> str:
            """启动一轮超参数智能体，并同步写入统筹记录。"""
            self._ensure_orchestration_context()

            data_context = json.dumps(self._latest_data_summary or {}, ensure_ascii=False)
            objective_parts = [
                "请先参考最新数据处理智能体的摘要，再进行超参数优化。",
                f"管理实验 ID: {self._manage_experiment_id}",
                f"数据处理实验 ID: {self._data_experiment_id}",
                f"数据处理摘要: {data_context}",
            ]
            if custom_objective:
                objective_parts.append(custom_objective)

            hpo_agent = create_hpo_agent(
                model_name=self.model_name,
                temperature=self.temperature,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
                config_path=self.config_path,
                experiments_dir=get_hpo_experiments_dir(),
            )
            request = AgentTaskRequest(
                action="optimize_hyperparameters",
                objective="\n".join(objective_parts),
                context={
                    "target_eer": target_eer,
                    "data_summary": self._latest_data_summary,
                    "config_path": self.config_path,
                },
                budget={"max_iterations": self.max_iterations},
                experiment_ids={
                    "manage": self._manage_experiment_id,
                    "data_processing": self._data_experiment_id,
                },
            )
            request_message = self._message_service.send_task(
                "coordinator", "hpo_agent", request
            )
            task_result = hpo_agent.execute_task(request)
            self._message_service.send_result(
                "hpo_agent", "coordinator", request_message, task_result
            )
            if task_result.status == "failed" or task_result.runtime_result is None:
                self._sync_manage_record(status="running", last_action="hpo_failed")
                return task_result.to_json()
            hpo_result = task_result.runtime_result

            self._hpo_results.append(hpo_result)
            self._latest_hpo_result = {
                "experiment_id": hpo_result.experiment_id,
                "best_config": hpo_result.best_config,
                "execution_summary": hpo_result.execution_summary,
                "final_answer": hpo_result.final_answer,
            }
            self._decision_history.append("hpo")

            hpo_experiment_id = hpo_result.experiment_id
            hpo_ids = self._linked_experiments.setdefault("hpo", [])
            if hpo_experiment_id and hpo_experiment_id not in hpo_ids:
                hpo_ids.append(hpo_experiment_id)

            self._sync_manage_record(status="running", last_action="hpo")
            self.agent_logger.append(
                "hpo_round_complete "
                f"manage_experiment_id={self._manage_experiment_id} "
                f"data_experiment_id={self._data_experiment_id}"
            )
            return task_result.to_json()

        return [
            LaunchDataProcessingRound,
            LaunchHPORound,
            CompareOrchestrationExperiments,
            GetOrchestrationExperimentResults,
            ListOrchestrationExperiments,
        ]

    def _sync_manage_record(
        self,
        status: Optional[str] = None,
        last_action: Optional[str] = None,
        final_answer: Optional[str] = None,
    ) -> None:
        if self._manage_experiment_id is None:
            return

        record = self.manage_tracker.get_experiment(self._manage_experiment_id) or {}
        result_summary = dict((record.get("extensions") or {}).get("result_summary") or {})
        if final_answer is not None:
            result_summary["final_answer"] = final_answer
        result_summary["linked_experiments"] = self._linked_experiments
        if self._message_service is not None:
            result_summary["agent_messages"] = [
                msg.to_dict() for msg in self._message_service.history()
            ]

        orchestration_state = {
            "manage_experiment_id": self._manage_experiment_id,
            "data_experiment_id": self._data_experiment_id,
            "decision_history": self._decision_history,
            "rounds": max(len(self._data_results), len(self._hpo_results)),
            "last_action": last_action,
            "target_eer": self._target_eer,
            "custom_objective": self._custom_objective,
            "latest_data_summary": self._latest_data_summary,
            "latest_hpo_result": self._latest_hpo_result,
            "final_answer": final_answer,
            "data_processing_rounds": len(self._data_results),
            "hpo_rounds": len(self._hpo_results),
        }
        duration = None
        if status in {"success", "failed", "cancelled"} and self._run_started_at is not None:
            duration = (datetime.now() - self._run_started_at).total_seconds()

        self.manage_tracker.update_orchestration_experiment(
            experiment_id=self._manage_experiment_id,
            status=status,
            duration=duration,
            extensions={
                "result_summary": result_summary,
                "orchestration": orchestration_state,
            },
            linked_experiments=self._linked_experiments,
            agent_messages=result_summary.get("agent_messages", []),
        )
        self.memory_service.update_working_state(
            self._manage_experiment_id,
            {
                "status": status or "running",
                "current_stage": last_action,
                "target_eer": self._target_eer,
                "custom_objective": self._custom_objective,
                "decision_history": self._decision_history,
                "latest_data_summary": self._latest_data_summary,
                "latest_hpo_result": self._latest_hpo_result,
                "linked_experiments": self._linked_experiments,
                "rounds": orchestration_state["rounds"],
            },
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
        self._ensure_orchestration_context()

        objective_parts = [
            "你现在是声纹识别系统的统筹智能体，必须通过工具完成优化闭环。",
            f"目标 EER: {target_eer}",
            f"管理实验 ID: {self._manage_experiment_id}",
            f"数据处理实验 ID: {self._data_experiment_id}",
            "建议先调用数据处理工具，再根据结果调用 HPO 工具。",
            f"最多进行 {self.max_rounds} 轮核心迭代。",
        ]
        if custom_objective:
            objective_parts.append(custom_objective)
        objective_parts.append(self._get_memory_context())

        objective = "\n".join(objective_parts)

        if self.verbose:
            print("=" * 80)
            print("🧩 Tool-based coordinated pipeline start")
            print("=" * 80)
            print(f"Manage Experiment ID: {self._manage_experiment_id}")
            print(f"Data Experiment ID: {self._data_experiment_id}")
            print("=" * 80)

        self.agent_logger.append(f"objective={objective.strip()}")

        try:
            result = self._invoke(objective)
            messages_result = result.get("messages", [])
            final_answer = ""
            if messages_result:
                final_answer = self._extract_message_content(messages_result[-1])

            self._sync_manage_record(status="success", last_action="complete", final_answer=final_answer)
            self._persist_episode("success", final_answer)
            self.agent_logger.append("agent_run_end success")

            if self.verbose:
                print("\n" + "=" * 80)
                print("✅ Coordinated pipeline end")
                print("=" * 80)

            return OrchestrationResult(
                experiment_id=self._manage_experiment_id or "",
                rounds=len(self._hpo_results),
                data_processing=self._data_results,
                hpo=self._hpo_results,
            )
        except Exception as exc:
            self._sync_manage_record(status="failed", last_action="error", final_answer=str(exc))
            self._persist_episode("failed", str(exc))
            self.agent_logger.append(f"agent_run_end error={exc}")
            raise


class OrchestratedPipeline(CoordinatorAgent):
    """Backward-compatible alias."""


ManagerAgent = CoordinatorAgent


__all__ = ["CoordinatorAgent", "ManagerAgent", "OrchestratedPipeline", "OrchestrationResult"]
