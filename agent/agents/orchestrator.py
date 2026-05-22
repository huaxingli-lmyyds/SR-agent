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

from agent.agents.a2a import A2AChannel, A2AMessage
from agent.agents.base_agent import BaseLangChainAgent
from agent.agents.data_processing_agent import create_data_processing_agent, DataProcessingResult
from agent.agents.react_agent import create_react_agent, OptimizationResult
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
        self.config_path = str(config_path)
        self.data_iterations = data_iterations
        self.max_rounds = max_rounds

        self.manage_tracker = ExperimentTracker(get_manage_experiments_dir())
        self.data_tracker = ExperimentTracker(get_data_processing_experiments_dir())
        self.hpo_tracker = ExperimentTracker(get_hpo_experiments_dir())
        self.agent_logger = AgentLogger(get_agent_dir() / "logs" / "agent_manage.log")

        self._manage_experiment_id: Optional[str] = None
        self._data_experiment_id: Optional[str] = None
        self._channel: Optional[A2AChannel] = None
        self._target_eer: float = 0.02
        self._custom_objective: Optional[str] = None
        self._latest_data_summary: Optional[Dict[str, Any]] = None
        self._latest_hpo_result: Optional[Dict[str, Any]] = None
        self._decision_history: List[str] = []
        self._data_summary_history: List[Dict[str, Any]] = []
        self._hpo_feedback_history: List[Dict[str, Any]] = []
        self._data_results: List[DataProcessingResult] = []
        self._hpo_results: List[OptimizationResult] = []
        self._linked_experiments: Dict[str, str] = {}

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

    def _reset_runtime_state(self) -> None:
        self._manage_experiment_id = None
        self._data_experiment_id = None
        self._channel = None
        self._latest_data_summary = None
        self._latest_hpo_result = None
        self._decision_history = []
        self._data_summary_history = []
        self._hpo_feedback_history = []
        self._data_results = []
        self._hpo_results = []
        self._linked_experiments = {}

    def _ensure_orchestration_context(self) -> None:
        if self._manage_experiment_id is not None and self._data_experiment_id is not None:
            return

        config_data = self._load_config()
        data_folder = str(config_data.get("data_folder") or "./datasets/voxceleb1")
        output_folder = config_data.get("output_folder")

        if self._manage_experiment_id is None:
            self._manage_experiment_id = self.manage_tracker.create_orchestration_experiment(
                config_path=self.config_path,
                data_folder=data_folder,
                output_folder=str(output_folder) if output_folder else None,
                description="orchestrated run",
            )

        if self._data_experiment_id is None:
            self._data_experiment_id = self.data_tracker.create_data_processing_experiment(
                config_path=self.config_path,
                data_folder=data_folder,
                output_folder=str(output_folder) if output_folder else None,
                description="data processing run",
            )

        if self._channel is None and self._manage_experiment_id is not None:
            self._channel = A2AChannel(session_id=self._manage_experiment_id)

        self._linked_experiments = {
            "manage": self._manage_experiment_id,
            "data_processing": self._data_experiment_id,
        }

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
                experiments_dir=get_data_processing_experiments_dir(),
            )

            result = data_agent.optimize_data_processing(
                target_goal=target_goal or "提升数据质量并保持训练/验证分布稳定",
                custom_objective=custom_objective,
                experiment_id=self._data_experiment_id,
                hpo_feedback=feedback,
            )

            self._data_results.append(result)
            self._latest_data_summary = result.data_summary
            self._data_summary_history.append(result.data_summary)
            self._decision_history.append("data_processing")

            if self._channel is not None:
                self._channel.send(
                    A2AMessage(
                        sender="coordinator",
                        recipient="data_processing_agent",
                        type="data_processing_request",
                        payload={
                            "target_goal": target_goal,
                            "custom_objective": custom_objective,
                            "experiment_id": self._data_experiment_id,
                        },
                    )
                )
                self._channel.send(
                    A2AMessage(
                        sender="data_processing_agent",
                        recipient="coordinator",
                        type="data_processing_summary",
                        payload=result.data_summary,
                    )
                )

            self._sync_manage_record(status="running", last_action="data_processing")
            self.agent_logger.append(
                "data_round_complete "
                f"manage_experiment_id={self._manage_experiment_id} "
                f"data_experiment_id={self._data_experiment_id}"
            )
            return self._format_data_round_result(result)

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

            hpo_agent = create_react_agent(
                model_name=self.model_name,
                temperature=self.temperature,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
                experiments_dir=get_hpo_experiments_dir(),
            )
            hpo_result = hpo_agent.optimize_hyperparameters(
                target_eer=target_eer,
                custom_objective="\n".join(objective_parts),
            )

            self._hpo_results.append(hpo_result)
            self._latest_hpo_result = {
                "best_config": hpo_result.best_config,
                "execution_summary": hpo_result.execution_summary,
                "final_answer": hpo_result.final_answer,
            }
            self._decision_history.append("hpo")

            latest_hpo_record = self.hpo_tracker.list_experiments(limit=1)
            hpo_experiment_id = latest_hpo_record[0]["experiment_id"] if latest_hpo_record else None
            if hpo_experiment_id:
                self._linked_experiments["hpo"] = hpo_experiment_id

            feedback = {
                "timestamp": datetime.now().isoformat(),
                "target_eer": target_eer,
                "best_config": hpo_result.best_config,
                "summary": hpo_result.execution_summary,
                "final_answer": hpo_result.final_answer,
                "round": len(self._hpo_results),
                "experiment_id": hpo_experiment_id,
            }
            self._hpo_feedback_history.append(feedback)

            if self._channel is not None:
                self._channel.send(
                    A2AMessage(
                        sender="coordinator",
                        recipient="hpo_agent",
                        type="hpo_request",
                        payload={
                            "target_eer": target_eer,
                            "custom_objective": custom_objective,
                            "data_summary": self._latest_data_summary,
                        },
                    )
                )
                self._channel.send(
                    A2AMessage(
                        sender="hpo_agent",
                        recipient="coordinator",
                        type="hpo_feedback",
                        payload=feedback,
                    )
                )

            self._sync_manage_record(status="running", last_action="hpo")
            self.agent_logger.append(
                "hpo_round_complete "
                f"manage_experiment_id={self._manage_experiment_id} "
                f"data_experiment_id={self._data_experiment_id}"
            )
            return self._format_hpo_round_result(hpo_result, hpo_experiment_id)

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
        results = dict(record.get("results") or {})
        if final_answer is not None:
            results["final_answer"] = final_answer
        results["data_processing_summary_history"] = self._data_summary_history
        results["hpo_feedback_history"] = self._hpo_feedback_history
        results["linked_experiments"] = self._linked_experiments
        if self._channel is not None:
            results["a2a_messages"] = [msg.to_dict() for msg in self._channel.history()]

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
        }

        self.manage_tracker.update_orchestration_experiment(
            experiment_id=self._manage_experiment_id,
            status=status,
            results=results,
            orchestration=orchestration_state,
            linked_experiments=self._linked_experiments,
            a2a_messages=results.get("a2a_messages", []),
            data_processing_summary_history=self._data_summary_history,
            hpo_feedback_history=self._hpo_feedback_history,
        )

    def _format_data_round_result(self, result: DataProcessingResult) -> str:
        best_config = json.dumps(result.best_config, ensure_ascii=False, indent=2)
        data_summary = json.dumps(result.data_summary, ensure_ascii=False, indent=2)
        return (
            "✅ 数据处理智能体完成一轮优化\n"
            f"实验 ID: {self._data_experiment_id}\n"
            f"总步骤数: {result.total_steps}\n"
            f"最佳配置: {best_config}\n"
            f"数据摘要: {data_summary}\n"
        )

    def _format_hpo_round_result(self, result: OptimizationResult, experiment_id: Optional[str]) -> str:
        best_config = json.dumps(result.best_config, ensure_ascii=False, indent=2)
        return (
            "✅ 超参数智能体完成一轮优化\n"
            f"实验 ID: {experiment_id or 'N/A'}\n"
            f"总步骤数: {result.total_steps}\n"
            f"最佳配置: {best_config}\n"
            f"执行总结: {result.execution_summary}\n"
        )

    def run(
        self,
        target_eer: float = 0.02,
        custom_objective: Optional[str] = None,
    ) -> OrchestrationResult:
        self._reset_runtime_state()
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
            self.agent_logger.append(f"agent_run_end error={exc}")
            raise


class OrchestratedPipeline(CoordinatorAgent):
    """Backward-compatible alias."""


ManagerAgent = CoordinatorAgent


__all__ = ["CoordinatorAgent", "ManagerAgent", "OrchestratedPipeline", "OrchestrationResult"]
