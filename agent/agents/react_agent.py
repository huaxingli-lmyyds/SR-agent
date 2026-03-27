"""
基于 LangChain v1.0 最新架构的声纹识别超参数优化智能体
使用 create_agent 标准接口，无需 AgentExecutor
"""

import os
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

import dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# 加载环境变量
dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())
os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("ZHIU_API_BASE_URL")


@dataclass
class OptimizationResult:
    """优化结果"""
    objective: str
    final_answer: str
    total_steps: int
    intermediate_steps: List[Any]
    best_config: Dict[str, Any]
    execution_summary: str


class LangChainHPOAgent:
    """基于 LangChain v1.0 的超参数优化智能体"""
    
    def __init__(
        self,
        model_name: str = "GLM-4.7",
        temperature: float = 0.2,
        max_iterations: int = 10,
        verbose: bool = True,
        config_path: str = "../configs/train_ecapa_tdnn.yaml"
    ):
        """
        初始化 LangChain v1.0 智能体
        
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
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        
        # 导入已修饰的工具
        self.tools = self._load_tools()
        
        # 创建系统提示词
        self.system_prompt = self._create_system_prompt()
        
        # 创建智能体（使用 create_agent，无需 AgentExecutor）
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt
        )
    
    def _create_system_prompt(self) -> str:
        """创建系统提示词"""
        return """你是一个声纹识别模型超参数优化专家，专门负责优化 ECAPA-TDNN 模型的性能。

你的任务：
1. 分析当前模型配置
2. 识别关键超参数（学习率、批次大小、训练轮数等）
3. 根据实验结果调整超参数
4. 使用提供的工具完成优化任务

优化目标：
- 最小化等错误率 (EER)
- 提高准确率
- 平衡训练时间和性能

关键超参数范围：
- 学习率 (lr): [0.0001, 0.01]
- 批次大小 (batch_size): [16, 128]
- 训练轮数 (number_of_epochs): [5, 20]

工作流程：
1. 首先使用 ReadConfig 读取并分析当前配置
2. 使用 ListConfigParameters 了解所有可调参数
3. 使用 UpdateConfig 调整超参数（参数格式为 JSON 字符串）
4. 使用 TrainModel 训练模型
5. 使用 EvaluateModel 评估性能
6. 使用 AnalyzeResults 分析结果
7. 迭代优化直到达到目标

重要提示：
- UpdateConfig 需要传入 JSON 字符串，例如 '{"lr": 0.0005, "batch_size": 64}'
- 所有工具调用参数都必须是字符串类型
- 请在最后提供最佳配置的总结

请使用 Final Answer 提供最佳配置和优化结果。"""
    
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
            CompareExperiments
        )
        from agent.tools.evaluation_tools import (
            RunEvaluation,
            GetEvaluationResults,
            CompareEvaluations,
            ListEvaluations
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
            CompareExperiments,
            RunEvaluation,
            GetEvaluationResults,
            CompareEvaluations,
            ListEvaluations
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
        # 构建优化目标
        if custom_objective:
            objective = custom_objective
        else:
            objective = f"""
优化 ECAPA-TDNN 模型的超参数以提升性能。

目标：
- 将等错误率 (EER) 降低到 {target_eer} 以下
- 提高模型准确率
- 平衡训练时间和性能

请开始优化，使用可用工具完成实验，并在最后提供最佳配置。
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
        
        # 执行智能体（直接 invoke，无需 AgentExecutor）
        try:
            # 使用新的消息格式
            messages = [
                {"role": "user", "content": objective}
            ]
            
            result = self.agent.invoke({"messages": messages})
            
            # 提取结果
            messages_result = result.get("messages", [])
            final_answer = ""
            if messages_result:
                final_answer = str(messages_result[-1].get("content", messages_result[-1]))
            
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
                execution_summary=summary
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
            
            return optimization_result
        
        except Exception as e:
            if self.verbose:
                print(f"\n❌ 执行失败: {str(e)}")
                import traceback
                traceback.print_exc()
            raise
    
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
        
        # 从训练步骤中提取
        train_steps = []
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) > 0:
                tool_call = step[0]
                if hasattr(tool_call, 'tool') and 'train' in str(tool_call.tool).lower():
                    train_steps.append(step)
        
        best_eer = float('inf')
        for step in train_steps:
            obs = step[1] if len(step) > 1 else ""
            obs_text = str(obs)
            
            eer_match = re.search(r'EER[:\s]+([\d.]+)', obs_text)
            if eer_match:
                eer = float(eer_match.group(1))
                if eer < best_eer:
                    best_eer = eer
                    
                    lr_match = re.search(r'lr[:\s]+([\d.]+)', obs_text)
                    batch_match = re.search(r'batch[:\s]+(\d+)', obs_text)
                    
                    if lr_match:
                        best_config['lr'] = float(lr_match.group(1))
                    if batch_match:
                        best_config['batch_size'] = int(batch_match.group(1))
                    best_config['eer'] = eer
        
        return best_config
    
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
        if self.verbose:
            print("=" * 80)
            print("🤖 执行自定义任务")
            print("=" * 80)
            print(f"任务: {objective}")
            print("=" * 80)
            print()
        
        # 使用新的消息格式
        messages = [
            {"role": "user", "content": objective}
        ]
        
        # 执行智能体
        result = self.agent.invoke({"messages": messages})
        
        # 提取结果
        messages_result = result.get("messages", [])
        final_answer = ""
        if messages_result:
            final_answer = str(messages_result[-1].get("content", messages_result[-1]))
        
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
            execution_summary=summary
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
            "tool_names": [tool.name for tool in self.tools]
        }


def create_react_agent(
    model_name: str = "GLM-4.7",
    temperature: float = 0.2,
    max_iterations: int = 10,
    verbose: bool = True
) -> LangChainHPOAgent:
    """
    创建 LangChain v1.0 智能体的便捷函数
    
    参数:
        model_name: 模型名称
        temperature: 温度参数
        max_iterations: 最大迭代次数
        verbose: 是否显示详细输出
        
    Returns:
        LangChainHPOAgent 实例
    """
    return LangChainHPOAgent(
        model_name=model_name,
        temperature=temperature,
        max_iterations=max_iterations,
        verbose=verbose
    )


# 兼容性别名
ReActHPOAgent = LangChainHPOAgent