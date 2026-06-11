"""
基于 LangChain v1.0 最新架构的声纹识别超参数优化智能体
使用 create_agent 标准接口，无需 AgentExecutor
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent.utils import ConfigParser, ExperimentTracker, get_agent_dir
from agent.utils.reward import compute_reward
from agent.utils.logger import AgentLogger
from agent.utils.path_tool import (
    get_config_file,
    get_hpo_experiments_dir,
    get_logs_dir,
    resolve_config_path,
    resolve_config_value_path,
    resolve_data_path,
    resolve_optional_project_path,
)
from agent.memory import EpisodeMemory, MemoryQuery, MemoryScope, MemoryService
from agent.agents.communication import AgentTaskRequest, AgentTaskResult
from agent.agents.base_agent import BaseLangChainAgent


@dataclass
class OptimizationResult:
    """优化结果"""
    objective: str
    final_answer: str
    total_steps: int
    intermediate_steps: List[Any]
    best_config: Dict[str, Any]
    execution_summary: str
    experiment_id: str


class HPOAgent(BaseLangChainAgent):
    """基于 LangChain v1.0 的超参数优化智能体"""
    
    def __init__(
        self,
        model_name: str = "GLM-4.7",
        temperature: float = 0.2,
        max_iterations: int = 10,
        verbose: bool = True,
        config_path: str = str(get_config_file("train_ecapa_tdnn.yaml")),
        experiments_dir: Optional[str] = None,
        memory_key: Optional[str] = None,
        memory_path: Optional[str] = None
    ):
        """
        初始化 LangChain v1.0 智能体
        
        参数:
            model_name: 使用的模型名称
            temperature: 温度参数
            max_iterations: 最大迭代次数（通过中间件实现）
            verbose: 是否显示详细输出
            config_path: 配置文件路径
            experiments_dir: HPO 实验目录（可选）
            memory_key: 记忆存储的模型标识（可选）
            memory_path: 记忆文件路径（可选）
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        self.config_path = str(resolve_config_path(config_path))
        self.agent_logger = AgentLogger(get_logs_dir() / "hpo_agent.log")
        self.experiments_dir = Path(experiments_dir).resolve() if experiments_dir else get_hpo_experiments_dir()
        self.memory_key = memory_key or Path(config_path).stem or "default_model"
        memory_root = resolve_optional_project_path(memory_path) if memory_path else None
        if memory_root is not None and memory_root.suffix:
            memory_root = memory_root.parent
        self.memory_service = MemoryService(root_dir=memory_root)
        self.memory_store = self.memory_service
        self.memory_scope = MemoryScope(
            agent_type="hpo_agent",
            task_type="speaker_verification",
            model_family=self.memory_key,
            tags=["optimization", "training", "evaluation"],
        )
        
        # 导入已修饰的工具
        self.tools = self._load_tools()
        
        # 创建系统提示词
        self.system_prompt = self._create_system_prompt()
        
        # 创建智能体（使用 create_agent，无需 AgentExecutor）
        self.agent = self._build_agent(
            tools=self.tools,
            system_prompt=self.system_prompt,
            middleware=self.agent_logger.build_middleware(),
        )

    def _persist_memory(
        self,
        objective: str,
        best_config: Dict[str, Any],
        summary: str,
        total_steps: int,
        intermediate_steps: Optional[List[Any]] = None,
        experiment_id: Optional[str] = None,
        status: str = "success",
        error: Optional[str] = None,
    ) -> None:
        """Persist optimization summary without interrupting execution."""
        try:
            metric_keys = {"eer", "accuracy", "loss", "min_dcf", "precision", "recall", "f1"}
            best_metrics = {k: v for k, v in best_config.items() if k in metric_keys}
            best_params = {k: v for k, v in best_config.items() if k not in metric_keys}
            changes, outcomes = self._extract_change_history(intermediate_steps or [])
            if error:
                outcomes["error"] = error
            self.memory_service.remember_episode(EpisodeMemory(
                agent_type="hpo_agent",
                objective=objective,
                action={
                    "best_config": best_params,
                    "changes": changes,
                    "total_steps": total_steps,
                },
                outcome={"best_metrics": best_metrics, **outcomes},
                summary=summary,
                experiment_ids=[experiment_id] if experiment_id else [],
                scope=self.memory_scope,
                status=status,
                importance=0.9 if status == "failed" or best_metrics else 0.7,
            ))
        except Exception as exc:
            self.agent_logger.append(f"memory_update_failed error={exc}")
    
    def _create_system_prompt(self) -> str:
        """创建系统提示词"""
        prompt_path = get_agent_dir() / "prompts" / "hpo_system_prompt.txt"
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except OSError:
            return "You are a model-agnostic hyperparameter optimization agent."

    def _get_history_context(self, metric: str = "eer") -> str:
        """从实验历史中提取基线信息。"""
        tracker = ExperimentTracker()
        best_list = tracker.find_best_experiment(metric=metric, minimize=True, top_n=1)
        if not best_list:
            return "历史实验: 暂无可用记录。"

        best = best_list[0]
        exp_id = best.get("experiment_id", "N/A")
        results = (best.get("metrics") or {}).get("test") or (best.get("metrics") or {}).get("validation") or {}
        config = best.get("config", {})

        key_fields = ["lr", "batch_size", "number_of_epochs"]
        config_summary = {k: config.get(k) for k in key_fields if k in config}
        reward, _ = compute_reward({
            "eer": results.get("eer"),
            "min_dcf": results.get("min_dcf"),
        })
        return (
            "历史最佳实验基线:\n"
            f"- 实验ID: {exp_id}\n"
            f"- {metric}: {results.get(metric, 'N/A')}\n"
            f"- 奖励分数: {reward if reward is not None else 'N/A'}\n"
            f"- 配置: {config_summary}\n"
        )

    def _get_memory_context(self, max_chars: int = 800) -> str:
        """从记忆存储中提取精简上下文。"""
        try:
            memory = self.memory_store.get_model(
                self.memory_key,
                dataset_key=self.memory_scope.dataset_key,
            )
            if not memory:
                return "记忆摘要: 暂无记录。"

            last_config = memory.get("last_best_config", {})
            last_metrics = memory.get("last_best_metrics", {})
            last_objective = memory.get("last_objective", "N/A")
            last_summary = memory.get("last_summary", "")
            last_changes = memory.get("last_changes", [])
            last_outcomes = memory.get("last_outcomes", {})

            summary_lines = [
                "记忆摘要:",
                f"- 最近目标: {last_objective}",
                f"- 最佳参数: {last_config}",
                f"- 最佳指标: {last_metrics}",
            ]

            if last_changes:
                summary_lines.append(f"- 最近改动: {last_changes[-1]}")

            if last_outcomes:
                summary_lines.append(f"- 最近结果: {last_outcomes}")

            if last_summary:
                summary_lines.append(f"- 总结: {last_summary}")

            text = "\n".join(summary_lines)
            if len(text) > max_chars:
                text = text[: max_chars - 3].rstrip() + "..."

            return text
        except Exception as exc:
            self.agent_logger.append(f"memory_read_failed error={exc}")
            return "记忆摘要: 读取失败。"
    
    def _load_tools(self):
        """加载已用 @tool 修饰的工具"""
        from agent.tools.config_tools import (
            ReadConfig,
            UpdateConfig,
            GetConfigStructure,
            ListConfigParameters,
            ResetConfig
        )
        from agent.tools.training_tools import (
            TrainModel,
            EvaluateModel,
            AnalyzeResults,
        )
        from agent.tools.experiment_history_tools import (
            CompareHPOExperiments,
            GetHPOExperimentResults,
            ListHPOExperiments,
        )
        from agent.tools.evaluation_tools import (
            RunEvaluation
        )
        from agent.tools.training_diagnostics_tools import (
            AnalyzeTrainingCurves,
            DiagnoseFitStatus,
        )
        from agent.tools.reward_tools import (
            ScoreExperiment,
        )
        
        # 直接返回已修饰的工具实例
        return [
            ReadConfig,
            UpdateConfig,
            GetConfigStructure,
            ListConfigParameters,
            ResetConfig,
            TrainModel,
            EvaluateModel,
            AnalyzeResults,
            AnalyzeTrainingCurves,
            DiagnoseFitStatus,
            ScoreExperiment,
            CompareHPOExperiments,
            RunEvaluation,
            GetHPOExperimentResults,
            ListHPOExperiments
        ]
    
    def optimize_hyperparameters(
        self,
        target_eer: float = 0.08,
        max_experiments: Optional[int] = None,
        custom_objective: Optional[str] = None
    ) -> OptimizationResult:
        """
        优化超参数
        
        参数:
            target_eer: 目标 EER 值
            max_experiments: 最大实验次数
            custom_objective: 自定义优化目标
            
        Returns:
            OptimizationResult 实例
        """
        run_started_at = datetime.now()
        tracker = ExperimentTracker(self.experiments_dir)
        config_data = ConfigParser(self.config_path).load_config(resolve_references=True)
        data_folder = str(resolve_data_path(config_data.get("data_folder")))
        self.memory_scope.dataset_key = data_folder
        output_folder = config_data.get("output_folder")
        resolved_output = resolve_config_value_path(output_folder)
        experiment_id = tracker.create_hpo_experiment(
            config_path=self.config_path,
            data_folder=data_folder,
            output_folder=str(resolved_output) if resolved_output else None,
            description="HPO agent optimization run",
            extra_fields={
                "extensions": {
                    "optimization": {
                        "target_eer": target_eer,
                        "max_experiments": max_experiments,
                    }
                }
            },
        )

        # 构建优化目标
        history_context = self._get_history_context(metric="eer")
        memory_context = self._get_memory_context(max_chars=800)

        if custom_objective:
            objective = (
                f"{history_context}\n{memory_context}\n"
                f"本轮 HPO 实验 ID: {experiment_id}。调用 TrainModel、RunEvaluation、"
                f"EvaluateModel 等实验工具时必须传入 experiment_id={experiment_id}。\n"
                f"{custom_objective}"
            )
        else:
            objective = f"""
        优化 ECAPA-TDNN 模型的超参数以提升性能。

        目标：
        - 将等错误率 (EER) 降低到 {target_eer} 以下
        - 提高模型准确率
        - 平衡训练时间和性能

        {history_context}

        {memory_context}

        本轮 HPO 实验 ID: {experiment_id}。
        调用 TrainModel、RunEvaluation、EvaluateModel 等实验工具时必须传入 experiment_id={experiment_id}。

        请开始优化，使用可用工具完成实验，并在最后提供最佳配置。
        约束：每轮最多只允许 1 次 UpdateConfig、1 次 TrainModel、1 次 RunEvaluation。
        约束：每次只改 1-2 个关键参数，不要反复尝试同一组参数。
        约束：如果训练收敛稳定且指标已经接近目标，或者连续 2 轮没有明显改善，立即停止并输出 Final Answer。
        """
        
        if self.verbose:
            print("=" * 80)
            print("🤖 LangChain v1.0 智能体启动")
            print("=" * 80)
            print(f"目标 EER: {target_eer}")
            print(f"最大迭代次数: {self.max_iterations}")
            print(f"可用工具数: {len(self.tools)}")
            print("=" * 80)
            print()

        self.agent_logger.append(f"objective={objective.strip()}")
        tracker.update_hpo_experiment(
            experiment_id=experiment_id,
            status="running",
            extensions={"optimization": {
                "target_eer": target_eer,
                "max_experiments": max_experiments,
                "objective": objective,
            }},
        )
        
        # 执行智能体（直接 invoke，无需 AgentExecutor）
        try:
            recursion_limit = max(60, self.max_iterations * 12)
            if max_experiments is not None:
                recursion_limit = max(recursion_limit, max_experiments * 20)
            result = self._invoke_with_recursion_limit(objective, recursion_limit)
            
            # 提取结果
            messages_result = result.get("messages", [])
            final_answer = ""
            if messages_result:
                final_answer = self._extract_message_content(messages_result[-1])
            
            intermediate_steps = result.get("intermediate_steps", [])
            
            # 分析结果
            best_config = self._extract_best_config(final_answer, intermediate_steps)
            
            # 生成总结
            summary = self._generate_summary(intermediate_steps, best_config)
            duration = (datetime.now() - run_started_at).total_seconds()
            record = tracker.get_experiment(experiment_id) or {}
            run_history = list(((record.get("extensions") or {}).get("optimization") or {}).get("run_history") or [])
            run_history.append(
                {
                    "started_at": run_started_at.isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "status": "success",
                    "duration_seconds": duration,
                    "target_eer": target_eer,
                    "objective": objective,
                    "best_config": best_config,
                    "execution_summary": summary,
                    "final_answer": final_answer,
                    "total_steps": len(intermediate_steps),
                }
            )
            
            optimization_result = OptimizationResult(
                objective=objective,
                final_answer=final_answer,
                total_steps=len(intermediate_steps),
                intermediate_steps=intermediate_steps,
                best_config=best_config,
                execution_summary=summary,
                experiment_id=experiment_id,
            )

            self._persist_memory(
                objective=objective,
                best_config=best_config,
                summary=summary,
                total_steps=optimization_result.total_steps,
                intermediate_steps=intermediate_steps,
                experiment_id=experiment_id,
            )

            tracker.update_hpo_experiment(
                experiment_id=experiment_id,
                status="success",
                duration=duration,
                extensions={"optimization": {
                    "target_eer": target_eer,
                    "objective": objective,
                    "best_config": best_config,
                    "execution_summary": summary,
                    "final_answer": final_answer,
                    "total_steps": len(intermediate_steps),
                    "run_history": run_history,
                }},
                parameters=best_config,
            )
            if self.verbose:
                print("\n" + "=" * 80)
                print("📋 执行总结")
                print("=" * 80)
                print(f"总步骤数: {optimization_result.total_steps}")
                print(f"\n最佳配置:")
                print(json.dumps(best_config, indent=2, ensure_ascii=False))
                print(f"\n最终答案:")
                print(final_answer)
                print("=" * 80)

            self.agent_logger.append("agent_run_end success")
            
            return optimization_result
        
        except Exception as e:
            duration = (datetime.now() - run_started_at).total_seconds()
            record = tracker.get_experiment(experiment_id) or {}
            run_history = list(((record.get("extensions") or {}).get("optimization") or {}).get("run_history") or [])
            run_history.append(
                {
                    "started_at": run_started_at.isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "status": "failed",
                    "duration_seconds": duration,
                    "target_eer": target_eer,
                    "objective": objective,
                    "error": str(e),
                }
            )
            tracker.update_hpo_experiment(
                experiment_id=experiment_id,
                status="failed",
                error=str(e),
                duration=duration,
                extensions={"optimization": {
                    "target_eer": target_eer,
                    "objective": objective,
                    "run_history": run_history,
                }},
            )
            self._persist_memory(
                objective=objective,
                best_config={},
                summary="HPO run failed.",
                total_steps=0,
                experiment_id=experiment_id,
                status="failed",
                error=str(e),
            )
            if self.verbose:
                print(f"\n❌ 执行失败: {str(e)}")
                import traceback
                traceback.print_exc()
            self.agent_logger.append(f"agent_run_end error={str(e)}")
            raise

    def execute_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        """Execute a coordinator request through the structured agent protocol."""
        if request.action != "optimize_hyperparameters":
            return AgentTaskResult(
                status="failed",
                error=f"unsupported HPO action: {request.action}",
                request_id=request.request_id,
            )
        try:
            result = self.optimize_hyperparameters(
                target_eer=float(request.context.get("target_eer", 0.02)),
                max_experiments=request.budget.get("max_experiments"),
                custom_objective=request.objective,
            )
            return AgentTaskResult(
                status="success",
                summary={
                    "best_config": result.best_config,
                    "execution_summary": result.execution_summary,
                    "final_answer": result.final_answer,
                    "total_steps": result.total_steps,
                },
                recommendations=[{
                    "action": "review_result",
                    "reason": "HPO round completed",
                    "priority": "normal",
                }],
                experiment_ids={"hpo": result.experiment_id},
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
        intermediate_steps: List[Any]
    ) -> Dict[str, Any]:
        """从执行历史中提取最佳配置"""
        best_config = {}
        
        # 从最终答案中提取
        json_matches = re.findall(r'\{[^}]*"lr"[^}]*\}', final_answer)
        for match in json_matches:
            try:
                config = json.loads(match)
                if 'eer' in config or 'lr' in config:
                    best_config.update(config)
            except json.JSONDecodeError:
                pass
        
        # 从训练/评估步骤中提取
        train_steps = []
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) > 0:
                tool_call = step[0]
                tool_name = str(getattr(tool_call, "tool", ""))
                if "train" in tool_name.lower() or "evaluate" in tool_name.lower():
                    train_steps.append(step)
        
        best_eer = float('inf')
        for step in train_steps:
            obs = step[1] if len(step) > 1 else ""
            obs_text = str(obs)

            exp_id_match = re.search(r"Experiment ID:\s*([\w\-]+)", obs_text)
            if exp_id_match:
                best_config["experiment_id"] = exp_id_match.group(1)
            
            eer_match = re.search(r'EER[:\s]+([\d.]+)', obs_text)
            valid_match = re.search(r'Valid ErrorRate[:\s]+([\d.eE+-]+)', obs_text)
            metric_val = None
            metric_key = None
            if eer_match:
                metric_val = float(eer_match.group(1))
                metric_key = "eer"
            elif valid_match:
                metric_val = float(valid_match.group(1))
                metric_key = "valid_error_rate"

            if metric_val is not None:
                if metric_val < best_eer:
                    best_eer = metric_val

                    if metric_key:
                        best_config[metric_key] = metric_val

                    lr_match = re.search(r'lr[:\s]+([\d.]+)', obs_text)
                    batch_match = re.search(r'batch[:\s]+(\d+)', obs_text)
                    
                    if lr_match:
                        best_config['lr'] = float(lr_match.group(1))
                    if batch_match:
                        best_config['batch_size'] = int(batch_match.group(1))
        
        return best_config

    def _extract_change_history(
        self, intermediate_steps: List[Any]
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        changes: List[Dict[str, Any]] = []
        outcomes: Dict[str, Any] = {}

        for step in intermediate_steps:
            if not isinstance(step, tuple) or len(step) < 1:
                continue
            tool_call = step[0]
            observation = step[1] if len(step) > 1 else ""
            tool_name = str(getattr(tool_call, "tool", ""))
            tool_input = getattr(tool_call, "tool_input", None)

            if "UpdateConfig" in tool_name:
                change_entry = {"tool": tool_name, "input": tool_input}
                changes.append(change_entry)

            obs_text = str(observation)
            if "TrainModel" in tool_name or "train" in tool_name.lower():
                match = re.search(r"Valid ErrorRate:\s*([\d.eE+-]+)", obs_text)
                if match:
                    outcomes["train_valid_error_rate"] = float(match.group(1))
            if "RunEvaluation" in tool_name or "EvaluateModel" in tool_name:
                match = re.search(r"EER[:\s]+([\d.]+)", obs_text)
                if match:
                    outcomes["eval_eer"] = float(match.group(1))

            exp_id_match = re.search(r"Experiment ID:\s*([\w\-]+)", obs_text)
            if exp_id_match:
                outcomes["experiment_id"] = exp_id_match.group(1)

        return changes, outcomes
    
    def _generate_summary(
        self,
        intermediate_steps: List[Any],
        best_config: Dict[str, Any]
    ) -> str:
        """生成执行总结"""
        if not intermediate_steps:
            return "暂无执行记录"
        
        tool_usage = {}
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) > 0:
                tool_call = step[0]
                if hasattr(tool_call, 'tool'):
                    tool_name = str(tool_call.tool)
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        summary_lines = [
            f"总执行步骤: {len(intermediate_steps)}",
            "",
            "工具使用统计:"
        ]
        
        for tool_name, count in sorted(tool_usage.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  - {tool_name}: {count} 次")
        
        summary_lines.append("")
        summary_lines.append("最佳配置:")
        for key, value in best_config.items():
            summary_lines.append(f"  - {key}: {value}")
        
        return "\n".join(summary_lines)
    
    def run_custom_task(self, objective: str) -> OptimizationResult:
        """
        运行自定义任务
        
        参数:
            objective: 任务描述
            
        Returns:
            OptimizationResult 实例
        """
        objective = f"{self._get_memory_context(max_chars=800)}\n\n{objective}"

        if self.verbose:
            print("=" * 80)
            print("🤖 执行自定义任务")
            print("=" * 80)
            print(f"任务: {objective}")
            print("=" * 80)
            print()
        
        # 执行智能体
        result = self._invoke(objective)
        
        # 提取结果
        messages_result = result.get("messages", [])
        final_answer = ""
        if messages_result:
            final_answer = self._extract_message_content(messages_result[-1])
        
        intermediate_steps = result.get("intermediate_steps", [])
        
        # 分析结果
        best_config = self._extract_best_config(final_answer, intermediate_steps)
        
        # 生成总结
        summary = self._generate_summary(intermediate_steps, best_config)
        
        optimization_result = OptimizationResult(
            objective=objective,
            final_answer=final_answer,
            total_steps=len(intermediate_steps),
            intermediate_steps=intermediate_steps,
            best_config=best_config,
            execution_summary=summary,
            experiment_id="",
        )

        self._persist_memory(
            objective=objective,
            best_config=best_config,
            summary=summary,
            total_steps=optimization_result.total_steps,
            intermediate_steps=intermediate_steps,
        )
        
        return optimization_result
    
    def get_execution_details(self) -> Dict[str, Any]:
        """获取智能体配置详情"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "verbose": self.verbose,
            "num_tools": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],
            "memory_key": self.memory_key,
            "memory_path": str(self.memory_service.root_dir)
        }


def create_hpo_agent(
    model_name: str = "GLM-4.7",
    temperature: float = 0.2,
    max_iterations: int = 10,
    verbose: bool = True,
    config_path: str = str(get_config_file("train_ecapa_tdnn.yaml")),
    experiments_dir: Optional[str] = None,
) -> HPOAgent:
    """
    创建 LangChain v1.0 智能体的便捷函数
    
    参数:
        model_name: 模型名称
        temperature: 温度参数
        max_iterations: 最大迭代次数
        verbose: 是否显示详细输出
        
    Returns:
        HPOAgent 实例
    """
    return HPOAgent(
        model_name=model_name,
        temperature=temperature,
        max_iterations=max_iterations,
        verbose=verbose,
        config_path=config_path,
        experiments_dir=experiments_dir,
    )
