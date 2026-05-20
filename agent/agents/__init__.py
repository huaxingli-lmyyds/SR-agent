"""
智能体模块
包含基于 ReAct 框架的声纹识别超参数优化智能体
"""

from .react_agent import (
    LangChainHPOAgent,
    ReActHPOAgent,  # 兼容性别名
    create_react_agent,
    OptimizationResult
)
from .data_processing_agent import (
    DataProcessingAgent,
    DataProcessingHPOAgent,
    DataProcessingResult,
    create_data_processing_agent,
)
from .orchestrator import (
    CoordinatorAgent,
    OrchestratedPipeline,
    OrchestrationResult,
)

__all__ = [
    'LangChainHPOAgent',
    'ReActHPOAgent',
    'create_react_agent',
    'OptimizationResult',
    'DataProcessingAgent',
    'DataProcessingHPOAgent',
    'DataProcessingResult',
    'create_data_processing_agent',
    'CoordinatorAgent',
    'OrchestratedPipeline',
    'OrchestrationResult',
]
