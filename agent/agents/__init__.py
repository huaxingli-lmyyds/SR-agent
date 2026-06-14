"""Agent implementations and coordination primitives with lazy exports."""

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "AdvisoryAgentBase": ("agent.agents.base_agent", "AdvisoryAgentBase"),
    "LangGraphAgent": ("agent.agents.base_agent", "LangGraphAgent"),
    "HPOAgent": ("agent.agents.hpo_agent", "HPOAgent"),
    "create_hpo_agent": ("agent.agents.hpo_agent", "create_hpo_agent"),
    "DataProcessingAgent": ("agent.agents.data_processing_agent", "DataProcessingAgent"),
    "create_data_processing_agent": ("agent.agents.data_processing_agent", "create_data_processing_agent"),
    "CoordinatorAgent": ("agent.agents.orchestrator", "CoordinatorAgent"),
    "OrchestrationResult": ("agent.agents.orchestrator", "OrchestrationResult"),
    "AgentMessage": ("agent.agents.communication", "AgentMessage"),
    "AgentTaskRequest": ("agent.agents.communication", "AgentTaskRequest"),
    "AgentTaskResult": ("agent.agents.communication", "AgentTaskResult"),
    "MessageService": ("agent.agents.communication", "MessageService"),
    "MessageType": ("agent.agents.communication", "MessageType"),
    "AgentRegistration": ("agent.agents.coordination", "AgentRegistration"),
    "AgentRegistry": ("agent.agents.coordination", "AgentRegistry"),
    "CompletionDecision": ("agent.agents.coordination", "CompletionDecision"),
    "CompletionPolicy": ("agent.agents.coordination", "CompletionPolicy"),
    "CoordinatedAgent": ("agent.agents.coordination", "CoordinatedAgent"),
    "TaskDispatcher": ("agent.agents.coordination", "TaskDispatcher"),
    "TaskExecutionRecord": ("agent.agents.coordination", "TaskExecutionRecord"),
    "OrchestrationWorkflow": ("agent.agents.orchestration_workflow", "OrchestrationWorkflow"),
    "OrchestrationDecisionPolicy": ("agent.agents.orchestration_workflow", "OrchestrationDecisionPolicy"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
