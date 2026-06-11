"""
智能体模块
包含模型无关的超参数优化、数据处理和协调智能体
"""

from .hpo_agent import (
    HPOAgent,
    create_hpo_agent,
    OptimizationResult
)
from .data_processing_agent import (
    DataProcessingAgent,
    DataProcessingResult,
    create_data_processing_agent,
)
from .orchestrator import (
    CoordinatorAgent,
    OrchestratedPipeline,
    OrchestrationResult,
)
from .communication import (
    AgentMessage,
    AgentTaskRequest,
    AgentTaskResult,
    MessageService,
    MessageType,
)
from .coordination import (
    AgentRegistration,
    AgentRegistry,
    CompletionDecision,
    CompletionPolicy,
    CoordinatedAgent,
    TaskDispatcher,
    TaskExecutionRecord,
)

__all__ = [
    'HPOAgent',
    'create_hpo_agent',
    'OptimizationResult',
    'DataProcessingAgent',
    'DataProcessingResult',
    'create_data_processing_agent',
    'CoordinatorAgent',
    'OrchestratedPipeline',
    'OrchestrationResult',
    'AgentMessage',
    'AgentTaskRequest',
    'AgentTaskResult',
    'MessageService',
    'MessageType',
    'AgentRegistration',
    'AgentRegistry',
    'CompletionDecision',
    'CompletionPolicy',
    'CoordinatedAgent',
    'TaskDispatcher',
    'TaskExecutionRecord',
]
