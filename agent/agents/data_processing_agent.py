"""
基于 LangChain v1.0 架构的数据处理优化智能体
使用 create_agent 标准接口
"""

import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent.utils import ExperimentTracker, ConfigParser
from agent.utils import get_agent_dir
from agent.utils.logger import AgentLogger
from agent.utils.path_tool import get_config_file
from agent.utils.path_tool import get_data_processing_experiments_dir
from agent.utils.path_tool import get_logs_dir, resolve_config_path, resolve_data_path, resolve_config_value_path
from agent.memory import EpisodeMemory, MemoryQuery, MemoryScope, MemoryService
from agent.agents.communication import AgentTaskRequest, AgentTaskResult
from agent.agents.base_agent import BaseLangChainAgent


@dataclass
class DataProcessingResult:
    """数据处理优化结果"""

    objective: str
    final_answer: str
    total_steps: int
    intermediate_steps: List[Any]
    best_config: Dict[str, Any]
    execution_summary: str
    data_summary: Dict[str, Any]
    experiment_id: str


class DataProcessingAgent(BaseLangChainAgent):
    """基于 LangChain v1.0 的数据处理优化智能体"""

    def __init__(
        self,
        model_name: str = "GLM-4.7",
        temperature: float = 0.2,
        max_iterations: int = 6,
        verbose: bool = True,
        config_path: str = str(get_config_file("train_ecapa_tdnn.yaml")),
        experiments_dir: Optional[Union[str, Path]] = None,
    ):
        """
        初始化数据处理智能体

        参数:
            model_name: 使用的模型名称
            temperature: 温度参数
            max_iterations: 最大迭代次数（通过中间件实现）
            verbose: 是否显示详细输出
            config_path: 配置文件路径
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        self.config_path = str(resolve_config_path(config_path))
        self.experiments_dir = Path(experiments_dir).resolve() if experiments_dir else get_data_processing_experiments_dir()
        self.tracker = ExperimentTracker(self.experiments_dir)
        self.agent_logger = AgentLogger(get_logs_dir() / "agent_DP.log")
        self.memory_service = MemoryService()
        self.memory_scope = MemoryScope(
            agent_type="data_processing",
            task_type="speaker_verification",
            model_family=Path(config_path).stem,
            tags=["data_processing", "data_quality", "data_preparation"],
        )

        self.tools = self._load_tools()
        self.system_prompt = self._create_system_prompt()
        self.agent = self._build_agent(
            tools=self.tools,
            system_prompt=self.system_prompt,
            middleware=self.agent_logger.build_middleware(),
        )

    def _get_memory_context(self) -> str:
        return self.memory_service.format_context(
            MemoryQuery(
                agent_type="data_processing",
                task_type=self.memory_scope.task_type,
                model_family=self.memory_scope.model_family,
                dataset_key=self.memory_scope.dataset_key,
                limit=3,
            ),
            max_chars=1000,
        )

    def _persist_memory(
        self,
        objective: str,
        best_config: Dict[str, Any],
        data_summary: Dict[str, Any],
        experiment_id: Optional[str],
        status: str = "success",
        error: Optional[str] = None,
    ) -> None:
        try:
            outcome = {"data_summary": data_summary}
            if error:
                outcome["error"] = error
            self.memory_service.remember_episode(EpisodeMemory(
                agent_type="data_processing",
                objective=objective,
                action={"best_config": best_config},
                outcome=outcome,
                summary=str(data_summary.get("summary", "")),
                experiment_ids=[experiment_id] if experiment_id else [],
                scope=self.memory_scope,
                status=status,
                importance=0.9 if status == "failed" else 0.75,
            ))
        except Exception as exc:
            self.agent_logger.append(f"memory_update_failed error={exc}")

    def _create_system_prompt(self) -> str:
        """创建系统提示词"""
        prompt_path = get_agent_dir() / "prompts" / "dp_prompt.txt"
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except OSError:
            return "You are a model-agnostic data quality and preparation agent."

    def _load_tools(self):
        """加载已用 @tool 修饰的工具"""
        from agent.tools.config_tools import (
            ReadConfig,
            UpdateConfig,
            GetConfigStructure,
            ListConfigParameters,
            ResetConfig,
        )
        from agent.tools.experiment_history_tools import (
            CompareDataProcessingExperiments,
            GetDataProcessingExperimentResults,
            ListDataProcessingExperiments,
        )
        from agent.tools.data_processing_tools import PrepareVoxCelebData

        return [
            ReadConfig,
            UpdateConfig,
            GetConfigStructure,
            ListConfigParameters,
            ResetConfig,
            CompareDataProcessingExperiments,
            GetDataProcessingExperimentResults,
            ListDataProcessingExperiments,
            PrepareVoxCelebData,
        ]

    def optimize_data_processing(
        self,
        target_goal: str = "提升数据质量并保持训练/验证分布稳定",
        max_runs: Optional[int] = None,
        custom_objective: Optional[str] = None,
        experiment_id: Optional[str] = None,
        hpo_feedback: Optional[Dict[str, Any]] = None,
    ) -> DataProcessingResult:
        """
        优化数据处理

        参数:
            target_goal: 优化目标描述
            max_runs: 最大实验次数
            custom_objective: 自定义优化目标

        Returns:
            DataProcessingResult 实例
        """
        run_started_at = datetime.now()
        config_data = ConfigParser(self.config_path).load_config(resolve_references=True)
        data_folder = str(resolve_data_path(config_data.get("data_folder")))
        self.memory_scope.dataset_key = data_folder
        if custom_objective:
            objective = custom_objective
        else:
            objective = f"""
        优化 VoxCeleb 数据准备流程以提升数据质量。

        目标：
        - {target_goal}
        - 合理设置 split_ratio 与 sentence_len
        - 输出最佳数据处理配置总结

        请开始优化，使用可用工具完成准备与分析，并在最后提供最佳配置。
        规则：当 PrepareVoxCelebData 返回成功后，必须立即停止继续调用任何工具，直接输出 Final Answer。
        规则：根据传入信息进行分析，最多只允许一轮 ReadConfig/ListConfigParameters/UpdateConfig 组合和一次 PrepareVoxCelebData 调用，不要循环重复执行。
        """

        if hpo_feedback:
            feedback_text = json.dumps(hpo_feedback, ensure_ascii=False)
            objective += f"\n\n来自超参数智能体的反馈:\n{feedback_text}\n"

        objective += f"\n\n{self._get_memory_context()}\n"

        tracker = self.tracker
        if experiment_id is None:
            output_folder = config_data.get("save_folder") or config_data.get("output_folder")
            resolved_output = resolve_config_value_path(output_folder)
            experiment_id = tracker.create_data_processing_experiment(
                config_path=self.config_path,
                data_folder=data_folder,
                output_folder=str(resolved_output) if resolved_output else None,
                description="data processing run",
            )

        objective += (
            f"\n\n本轮数据处理实验 ID: {experiment_id}。"
            f"调用 PrepareVoxCelebData 时必须传入 experiment_id={experiment_id}。\n"
        )

        tracker.update_data_processing_experiment(
            experiment_id=experiment_id,
            status="running",
            extensions={"data_processing": {
                "objective": objective,
                "target_goal": target_goal,
                "hpo_feedback": hpo_feedback,
            }},
        )

        if self.verbose:
            print("=" * 80)
            print("🤖 数据处理智能体启动")
            print("=" * 80)
            print(f"目标: {target_goal}")
            print(f"最大迭代次数: {self.max_iterations}")
            print(f"可用工具数: {len(self.tools)}")
            print("=" * 80)
            print()

        self.agent_logger.append(f"objective={objective.strip()}")

        try:
            recursion_limit = max(50, self.max_iterations * 8)
            result = self._invoke_with_recursion_limit(objective, recursion_limit)

            messages_result = result.get("messages", [])
            final_answer = ""
            if messages_result:
                final_answer = self._extract_message_content(messages_result[-1])

            intermediate_steps = result.get("intermediate_steps", [])
            best_config = self._extract_best_config(final_answer, intermediate_steps)
            summary = self._generate_summary(intermediate_steps, best_config)
            data_summary = self._build_data_summary(best_config, summary, objective)
            current_record = tracker.get_experiment(experiment_id) or {}
            if current_record.get("status") == "failed":
                raise RuntimeError(current_record.get("error") or "data processing tool failed")
            prepared_data = (current_record.get("extensions") or {}).get("data_processing") or {}
            data_summary["prepared_data"] = {
                key: prepared_data.get(key)
                for key in (
                    "data_folder",
                    "save_folder",
                    "verification_file",
                    "split_ratio",
                    "sentence_len",
                    "splits",
                    "skip_prep",
                    "random_segment",
                    "split_speaker",
                    "stats",
                    "csv_files",
                    "prepare_status",
                )
            }
            duration = (datetime.now() - run_started_at).total_seconds()
            record = current_record
            run_history = list(((record.get("extensions") or {}).get("data_processing") or {}).get("run_history") or [])
            run_history.append(
                {
                    "started_at": run_started_at.isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "status": "success",
                    "duration_seconds": duration,
                    "target_goal": target_goal,
                    "objective": objective,
                    "hpo_feedback": hpo_feedback,
                    "best_config": best_config,
                    "summary": data_summary,
                    "final_answer": final_answer,
                    "total_steps": len(intermediate_steps),
                }
            )

            tracker.update_data_processing_experiment(
                experiment_id=experiment_id,
                duration=duration,
                extensions={"data_processing": {
                    "summary": data_summary,
                    "best_config": best_config,
                    "objective": objective,
                    "final_answer": final_answer,
                    "total_steps": len(intermediate_steps),
                    "run_history": run_history,
                    **best_config,
                }},
                parameters=best_config,
                status="success",
            )

            processing_result = DataProcessingResult(
                objective=objective,
                final_answer=final_answer,
                total_steps=len(intermediate_steps),
                intermediate_steps=intermediate_steps,
                best_config=best_config,
                execution_summary=summary,
                data_summary=data_summary,
                experiment_id=experiment_id,
            )

            if self.verbose:
                print("\n" + "=" * 80)
                print("📋 执行总结")
                print("=" * 80)
                print(f"总步骤数: {processing_result.total_steps}")
                print("\n最佳配置:")
                print(json.dumps(best_config, indent=2, ensure_ascii=False))
                print("\n最终答案:")
                print(final_answer)
                print("=" * 80)

            self.agent_logger.append("agent_run_end success")
            self._persist_memory(
                objective=objective,
                best_config=best_config,
                data_summary=data_summary,
                experiment_id=experiment_id,
            )

            return processing_result

        except Exception as e:
            duration = (datetime.now() - run_started_at).total_seconds()
            record = tracker.get_experiment(experiment_id) or {}
            run_history = list(((record.get("extensions") or {}).get("data_processing") or {}).get("run_history") or [])
            run_history.append(
                {
                    "started_at": run_started_at.isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "status": "failed",
                    "duration_seconds": duration,
                    "target_goal": target_goal,
                    "objective": objective,
                    "hpo_feedback": hpo_feedback,
                    "error": str(e),
                }
            )
            tracker.update_data_processing_experiment(
                experiment_id=experiment_id,
                status="failed",
                error=str(e),
                duration=duration,
                extensions={"data_processing": {
                    "objective": objective,
                    "target_goal": target_goal,
                    "hpo_feedback": hpo_feedback,
                    "run_history": run_history,
                }},
            )
            if self.verbose:
                print(f"\n❌ 执行失败: {str(e)}")
                import traceback
                traceback.print_exc()
            self.agent_logger.append(f"agent_run_end error={str(e)}")
            self._persist_memory(
                objective=objective,
                best_config={},
                data_summary={},
                experiment_id=experiment_id,
                status="failed",
                error=str(e),
            )
            raise

    def execute_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        """Execute a coordinator request through the structured agent protocol."""
        if request.action != "optimize_data_processing":
            return AgentTaskResult(
                status="failed",
                error=f"unsupported data processing action: {request.action}",
                request_id=request.request_id,
            )
        try:
            result = self.optimize_data_processing(
                target_goal=request.context.get("target_goal") or request.objective,
                max_runs=request.budget.get("max_runs"),
                custom_objective=request.objective,
                experiment_id=request.experiment_ids.get("data_processing"),
                hpo_feedback=request.context.get("hpo_feedback"),
            )
            return AgentTaskResult(
                status="success",
                summary={
                    "best_config": result.best_config,
                    "data_summary": result.data_summary,
                    "execution_summary": result.execution_summary,
                    "final_answer": result.final_answer,
                    "total_steps": result.total_steps,
                },
                recommendations=[{
                    "action": "run_hpo",
                    "reason": "data processing round completed",
                    "priority": "normal",
                }],
                experiment_ids={"data_processing": result.experiment_id},
                request_id=request.request_id,
                runtime_result=result,
            )
        except Exception as exc:
            return AgentTaskResult(
                status="failed",
                error=str(exc),
                experiment_ids=request.experiment_ids,
                request_id=request.request_id,
            )

    def _extract_best_config(
        self,
        final_answer: str,
        intermediate_steps: List[Any],
    ) -> Dict[str, Any]:
        """从执行历史中提取最佳数据处理配置"""
        best_config: Dict[str, Any] = {}
        keys = {
            "split_ratio",
            "sentence_len",
            "skip_prep",
            "random_segment",
            "amp_th",
            "split_speaker",
            "data_folder",
            "save_folder",
        }

        json_matches = re.findall(r"\{[^}]*\}", final_answer)
        for match in json_matches:
            try:
                config = json.loads(match)
                if any(key in config for key in keys):
                    best_config.update({k: v for k, v in config.items() if k in keys})
            except json.JSONDecodeError:
                pass

        prep_steps = []
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) > 0:
                tool_call = step[0]
                if hasattr(tool_call, "tool") and "prepare" in str(tool_call.tool).lower():
                    prep_steps.append(step)

        if prep_steps:
            obs_text = str(prep_steps[-1][1] if len(prep_steps[-1]) > 1 else "")
            ratio_match = re.search(r"拆分比例:\s*\[([^\]]+)\]", obs_text)
            if ratio_match:
                ratio_vals = [item.strip() for item in ratio_match.group(1).split(",")]
                try:
                    best_config["split_ratio"] = [int(v) for v in ratio_vals]
                except ValueError:
                    pass

            sentence_match = re.search(r"片段长度:\s*([\d.]+)s", obs_text)
            if sentence_match:
                best_config["sentence_len"] = float(sentence_match.group(1))

            for key, label in [
                ("random_segment", "随机切片"),
                ("split_speaker", "按说话人划分"),
                ("skip_prep", "跳过准备"),
            ]:
                bool_match = re.search(fr"{label}:\s*(True|False)", obs_text)
                if bool_match:
                    best_config[key] = bool_match.group(1) == "True"

        return best_config

    def _generate_summary(
        self,
        intermediate_steps: List[Any],
        best_config: Dict[str, Any],
    ) -> str:
        """生成执行总结"""
        if not intermediate_steps:
            return "暂无执行记录"

        tool_usage: Dict[str, int] = {}
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) > 0:
                tool_call = step[0]
                if hasattr(tool_call, "tool"):
                    tool_name = str(tool_call.tool)
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        summary_lines = [
            f"总执行步骤: {len(intermediate_steps)}",
            "",
            "工具使用统计:",
        ]

        for tool_name, count in sorted(tool_usage.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  - {tool_name}: {count} 次")

        summary_lines.append("")
        summary_lines.append("最佳配置:")
        for key, value in best_config.items():
            summary_lines.append(f"  - {key}: {value}")

        return "\n".join(summary_lines)

    def _build_data_summary(
        self,
        best_config: Dict[str, Any],
        summary: str,
        objective: str,
    ) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(),
            "objective": objective,
            "best_config": best_config,
            "summary": summary,
        }

    def run_custom_task(self, objective: str) -> DataProcessingResult:
        """
        运行自定义任务

        参数:
            objective: 任务描述

        Returns:
            DataProcessingResult 实例
        """
        objective = f"{self._get_memory_context()}\n\n{objective}"

        if self.verbose:
            print("=" * 80)
            print("🤖 执行自定义任务")
            print("=" * 80)
            print(f"任务: {objective}")
            print("=" * 80)
            print()

        self.agent_logger.append(f"objective={objective.strip()}")

        try:
            result = self._invoke(objective)

            messages_result = result.get("messages", [])
            final_answer = ""
            if messages_result:
                final_answer = self._extract_message_content(messages_result[-1])

            intermediate_steps = result.get("intermediate_steps", [])
            best_config = self._extract_best_config(final_answer, intermediate_steps)
            summary = self._generate_summary(intermediate_steps, best_config)

            processing_result = DataProcessingResult(
                objective=objective,
                final_answer=final_answer,
                total_steps=len(intermediate_steps),
                intermediate_steps=intermediate_steps,
                best_config=best_config,
                execution_summary=summary,
                data_summary=self._build_data_summary(best_config, summary, objective),
                experiment_id="",
            )

            self.agent_logger.append("agent_run_end success")
            self._persist_memory(
                objective=objective,
                best_config=best_config,
                data_summary=processing_result.data_summary,
                experiment_id=None,
            )
            return processing_result
        except Exception as exc:
            self.agent_logger.append(f"agent_run_end error={exc}")
            raise

    def get_execution_details(self) -> Dict[str, Any]:
        """获取智能体配置详情"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "verbose": self.verbose,
            "num_tools": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],
        }


def create_data_processing_agent(
    model_name: str = "GLM-4.7",
    temperature: float = 0.2,
    max_iterations: int = 6,
    verbose: bool = True,
    config_path: str = str(get_config_file("train_ecapa_tdnn.yaml")),
    experiments_dir: Optional[Union[str, Path]] = None,
) -> DataProcessingAgent:
    """
    创建数据处理智能体的便捷函数

    参数:
        model_name: 模型名称
        temperature: 温度参数
        max_iterations: 最大迭代次数
        verbose: 是否显示详细输出

    Returns:
        DataProcessingAgent 实例
    """
    return DataProcessingAgent(
        model_name=model_name,
        temperature=temperature,
        max_iterations=max_iterations,
        verbose=verbose,
        config_path=config_path,
        experiments_dir=experiments_dir,
    )
