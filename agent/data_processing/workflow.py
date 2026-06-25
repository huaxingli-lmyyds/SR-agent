"""LangGraph workflow for deterministic dataset processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .service import (
    build_processing_plan,
    execute_plan,
    infer_dataset_spec,
    profile_dataset,
    publish_dataset_version,
)


DataStrategyAdvisor = Callable[[Dict[str, Any]], Dict[str, Any]]


class DataProcessingGraphState(TypedDict, total=False):
    dataset_uri: str
    dataset_type: str
    task_type: str
    target_goal: str
    output_path: str
    requested_operations: list[Dict[str, Any]]
    advice: Dict[str, Any]
    profile: Dict[str, Any]
    plan: Dict[str, Any]
    results: list[Dict[str, Any]]
    published_version: Dict[str, Any]
    status: str
    error: Optional[str]
    route: str


class DataProcessingDecisionPolicy:
    """Convert requested/advised operations into a bounded deterministic plan."""

    max_operations = 20

    def requested_operations(self, state: DataProcessingGraphState) -> list[Dict[str, Any]]:
        explicit = [
            {**item, "_advisory": False}
            for item in state.get("requested_operations") or []
            if isinstance(item, dict)
        ]
        advised = [
            {**item, "_advisory": True}
            for item in (state.get("advice") or {}).get("suggested_operations") or []
            if isinstance(item, dict)
        ]
        candidates = explicit + advised
        return [
            {
                "operation": str(item.get("operation") or ""),
                "parameters": dict(item.get("parameters") or {}),
                "reason": str(item.get("reason") or ""),
                "expected_effect": dict(item.get("expected_effect") or {}),
                "_advisory": bool(item.get("_advisory")),
            }
            for item in candidates[: self.max_operations]
            if isinstance(item, dict) and item.get("operation")
        ]

    def after_execute(self, results: list[Any]) -> str:
        return "fail" if not results or any(item.status == "failed" for item in results) else "publish"


class DataProcessingWorkflow:
    """Inspect, plan, execute, validate, and publish through LangGraph."""

    def __init__(
        self,
        *,
        decision_policy: Optional[DataProcessingDecisionPolicy] = None,
        strategy_advisor: Optional[DataStrategyAdvisor] = None,
    ) -> None:
        self.decision_policy = decision_policy or DataProcessingDecisionPolicy()
        self.strategy_advisor = strategy_advisor
        self._dataset = None
        self._profile = None
        self._plan = None
        self._results = []
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(DataProcessingGraphState)
        graph.add_node("advise", self._advise)
        graph.add_node("inspect", self._inspect)
        graph.add_node("plan", self._build_plan)
        graph.add_node("execute", self._execute)
        graph.add_node("publish", self._publish)
        graph.add_node("fail", self._fail)
        graph.add_edge(START, "advise")
        graph.add_edge("advise", "inspect")
        graph.add_edge("inspect", "plan")
        graph.add_edge("plan", "execute")
        graph.add_conditional_edges(
            "execute",
            lambda state: state["route"],
            {"publish": "publish", "fail": "fail"},
        )
        graph.add_edge("publish", END)
        graph.add_edge("fail", END)
        return graph.compile()

    def run(self, state: DataProcessingGraphState) -> DataProcessingGraphState:
        return self.graph.invoke(state)

    def _advise(self, state: DataProcessingGraphState) -> Dict[str, Any]:
        try:
            advice = self.strategy_advisor(dict(state)) if self.strategy_advisor else {}
        except Exception as exc:
            advice = {"advice_error": f"{type(exc).__name__}: {exc}"}
        return {"advice": dict(advice or {})}

    def _inspect(self, state: DataProcessingGraphState) -> Dict[str, Any]:
        self._dataset = infer_dataset_spec(
            state["dataset_uri"],
            dataset_type=state.get("dataset_type", "auto"),
            task_type=state.get("task_type", "generic"),
        )
        self._profile = profile_dataset(self._dataset)
        return {"profile": self._profile.to_dict()}

    def _build_plan(self, state: DataProcessingGraphState) -> Dict[str, Any]:
        requested = self.decision_policy.requested_operations(state)
        self._plan = build_processing_plan(self._profile, state.get("target_goal", ""), requested)
        return {"plan": self._plan.to_dict()}

    def _execute(self, state: DataProcessingGraphState) -> Dict[str, Any]:
        output_root = Path(state["output_path"]).parent / "processed"
        self._results = execute_plan(self._plan, output_root=output_root)
        route = self.decision_policy.after_execute(self._results)
        error = next((item.error for item in self._results if item.status == "failed"), None)
        return {
            "results": [item.to_dict() for item in self._results],
            "route": route,
            "error": error,
        }

    def _publish(self, state: DataProcessingGraphState) -> Dict[str, Any]:
        version = publish_dataset_version(
            self._dataset,
            self._results,
            Path(state["output_path"]),
        )
        return {"published_version": version.to_dict(), "status": "success", "error": None}

    @staticmethod
    def _fail(state: DataProcessingGraphState) -> Dict[str, Any]:
        return {"status": "failed", "error": state.get("error") or "data processing failed"}


__all__ = [
    "DataProcessingDecisionPolicy",
    "DataProcessingGraphState",
    "DataProcessingWorkflow",
    "DataStrategyAdvisor",
]
