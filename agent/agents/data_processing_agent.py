"""
基于 LangChain v1.0 架构的数据处理优化智能体
使用 create_agent 标准接口，无需 AgentExecutor
"""

import os
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# 加载环境变量
dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())
os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("ZHIU_API_BASE_URL")


@dataclass
class DataProcessingResult:
    """数据处理优化结果"""

    objective: str
    final_answer: str
    total_steps: int
    intermediate_steps: List[Any]
    best_config: Dict[str, Any]
    execution_summary: str


class DataProcessingAgent:
    """基于 LangChain v1.0 的数据处理优化智能体"""

    def __init__(
        self,
        model_name: str = "GLM-4.7",
        temperature: float = 0.2,
        max_iterations: int = 6,
        verbose: bool = True,
        config_path: str = "../configs/train_ecapa_tdnn.yaml",
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
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.config_path = config_path

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

        self.tools = self._load_tools()
        self.system_prompt = self._create_system_prompt()

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt,
        )

    def _create_system_prompt(self) -> str:
        """创建系统提示词"""
        return """你是一个声纹识别数据处理优化专家，负责优化 VoxCeleb 数据准备流程。

        你的任务：
        1. 分析当前数据配置与预处理设置
        2. 识别关键数据处理参数（split_ratio、sentence_len、skip_prep 等）
        3. 根据数据统计结果调整参数
        4. 使用提供的工具完成数据准备与优化

        优化目标：
        - 数据切分合理、统计稳定
        - 提升训练/验证数据质量
        - 平衡准备时间与数据完整性

        关键参数范围建议：
        - split_ratio: [80, 20] ~ [95, 5]
        - sentence_len: [2.0, 5.0]
        - skip_prep: False（建议先完整准备）

        工作流程：
        1. 使用 ReadConfig 读取并分析当前配置
        2. 使用 ListConfigParameters 查看可调参数
        3. 使用 UpdateConfig 调整数据相关配置（参数格式为 JSON 字符串）
        4. 使用 PrepareVoxCelebData 生成数据 CSV 并查看统计
        5. 依据统计结果迭代优化

        重要提示：
        - UpdateConfig 需要传入 JSON 字符串，例如 '{"split_ratio": [90, 10], "sentence_len": 3.0}'
        - 工具调用参数尽量使用字符串类型
        - 请在最后提供最佳数据处理配置总结

        请使用 Final Answer 提供最佳配置和优化结果。"""

    def _load_tools(self):
        """加载已用 @tool 修饰的工具"""
        from agent.tools.config_tools import (
            ReadConfig,
            UpdateConfig,
            GetConfigStructure,
            ListConfigParameters,
            ResetConfig,
        )
        from agent.tools.data_processing_tools import PrepareVoxCelebData

        return [
            ReadConfig,
            UpdateConfig,
            GetConfigStructure,
            ListConfigParameters,
            ResetConfig,
            PrepareVoxCelebData,
        ]

    def optimize_data_processing(
        self,
        target_goal: str = "提升数据质量并保持训练/验证分布稳定",
        max_runs: Optional[int] = None,
        custom_objective: Optional[str] = None,
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
        """

        if self.verbose:
            print("=" * 80)
            print("🤖 数据处理智能体启动")
            print("=" * 80)
            print(f"目标: {target_goal}")
            print(f"最大迭代次数: {self.max_iterations}")
            print(f"可用工具数: {len(self.tools)}")
            print("=" * 80)
            print()

        try:
            messages = [{"role": "user", "content": objective}]
            result = self.agent.invoke({"messages": messages})

            messages_result = result.get("messages", [])
            final_answer = ""
            if messages_result:
                final_answer = str(messages_result[-1].get("content", messages_result[-1]))

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

            return processing_result

        except Exception as e:
            if self.verbose:
                print(f"\n❌ 执行失败: {str(e)}")
                import traceback
                traceback.print_exc()
            raise

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

    def run_custom_task(self, objective: str) -> DataProcessingResult:
        """
        运行自定义任务

        参数:
            objective: 任务描述

        Returns:
            DataProcessingResult 实例
        """
        if self.verbose:
            print("=" * 80)
            print("🤖 执行自定义任务")
            print("=" * 80)
            print(f"任务: {objective}")
            print("=" * 80)
            print()

        messages = [{"role": "user", "content": objective}]
        result = self.agent.invoke({"messages": messages})

        messages_result = result.get("messages", [])
        final_answer = ""
        if messages_result:
            final_answer = str(messages_result[-1].get("content", messages_result[-1]))

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
        )

        return processing_result

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
    )


DataProcessingHPOAgent = DataProcessingAgent
