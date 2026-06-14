"""Registry-driven LangGraph workflow for coordinating specialized agents."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .communication import AgentTaskRequest
from .coordination import AgentRegistry, CompletionPolicy, TaskDispatcher, TaskExecutionRecord


RequestFactory = Callable[[Dict[str, Any], Dict[str, Any]], AgentTaskRequest]
CoordinationAdvisor = Callable[[List[Dict[str, Any]], Dict[str, Any]], Dict[str, Any]]


class OrchestrationGraphState(TypedDict, total=False):
    context: Dict[str, Any]
    budget: Dict[str, Any]
    pending_agents: List[str]
    current_agent: Optional[str]
    records: List[TaskExecutionRecord]
    latest_results: Dict[str, Dict[str, Any]]
    route: str
    completion: Dict[str, Any]
    advice: Dict[str, Any]


class OrchestrationDecisionPolicy:
    """Choose registered agents deterministically; registry additions are automatic."""

    def order(self, registry: AgentRegistry) -> List[str]:
        preferred = ["data_processing_agent", "hpo_agent"]
        available = [item["agent_type"] for item in registry.describe()]
        return [item for item in preferred if item in available] + [
            item for item in available if item not in preferred
        ]


class OrchestrationWorkflow:
    def __init__(
        self,
        registry: AgentRegistry,
        dispatcher: TaskDispatcher,
        completion_policy: CompletionPolicy,
        request_factory: RequestFactory,
        *,
        decision_policy: Optional[OrchestrationDecisionPolicy] = None,
        advisor: Optional[CoordinationAdvisor] = None,
    ) -> None:
        self.registry = registry
        self.dispatcher = dispatcher
        self.completion_policy = completion_policy
        self.request_factory = request_factory
        self.decision_policy = decision_policy or OrchestrationDecisionPolicy()
        self.advisor = advisor
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(OrchestrationGraphState)
        graph.add_node("plan", self._plan)
        graph.add_node("select_agent", self._select_agent)
        graph.add_node("dispatch", self._dispatch)
        graph.add_node("complete", self._complete)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "select_agent")
        graph.add_conditional_edges(
            "select_agent",
            lambda state: state["route"],
            {"dispatch": "dispatch", "complete": "complete"},
        )
        graph.add_edge("dispatch", "select_agent")
        graph.add_edge("complete", END)
        return graph.compile()

    def run(self, context: Dict[str, Any], budget: Dict[str, Any]) -> OrchestrationGraphState:
        return self.graph.invoke(
            {
                "context": context,
                "budget": budget,
                "records": [],
                "latest_results": {},
            }
        )

    def _plan(self, state: OrchestrationGraphState) -> Dict[str, Any]:
        descriptions = self.registry.describe()
        try:
            advice = self.advisor(descriptions, state.get("context") or {}) if self.advisor else {}
        except Exception as exc:
            advice = {"advice_error": f"{type(exc).__name__}: {exc}"}
        return {
            "pending_agents": self.decision_policy.order(self.registry),
            "advice": dict(advice or {}),
        }

    @staticmethod
    def _select_agent(state: OrchestrationGraphState) -> Dict[str, Any]:
        pending = list(state.get("pending_agents") or [])
        if not pending:
            return {"route": "complete", "current_agent": None}
        return {
            "route": "dispatch",
            "current_agent": pending[0],
            "pending_agents": pending[1:],
        }

    def _dispatch(self, state: OrchestrationGraphState) -> Dict[str, Any]:
        agent_type = str(state["current_agent"])
        registration = self.registry.get(agent_type)
        action = registration.actions[0]
        request = self.request_factory(dict(state.get("context") or {}), dict(state.get("budget") or {}))
        request.action = action
        record = self.dispatcher.dispatch(agent_type, request)
        records = list(state.get("records") or [])
        records.append(record)
        latest = dict(state.get("latest_results") or {})
        latest[agent_type] = record.result
        context = dict(state.get("context") or {})
        context["previous_results"] = latest
        return {"records": records, "latest_results": latest, "context": context}

    def _complete(self, state: OrchestrationGraphState) -> Dict[str, Any]:
        decision = self.completion_policy.evaluate(state.get("records") or [])
        return {"completion": decision.to_dict()}


__all__ = [
    "CoordinationAdvisor",
    "OrchestrationDecisionPolicy",
    "OrchestrationGraphState",
    "OrchestrationWorkflow",
    "RequestFactory",
]
