"""LangGraph workflow for deterministic dataset processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .contracts import QualityPolicy
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
    quality_policy: Dict[str, Any]
    profile: Dict[str, Any]
    advice: Dict[str, Any]
    plan: Dict[str, Any]
    results: list[Dict[str, Any]]
    published_version: Dict[str, Any]
    status: str
    error: Optional[str]
    route: str


MAX_OPERATIONS = 20


def _requested_operations(
    state: DataProcessingGraphState,
) -> list[Dict[str, Any]]:
    """Bound and label explicit and LLM-advised operations."""

    explicit = [
        (item, False)
        for item in state.get("requested_operations") or []
        if isinstance(item, dict) and item.get("operation")
    ]
    advised = [
        (item, True)
        for item in (state.get("advice") or {}).get(
            "suggested_operations", []
        )
        if isinstance(item, dict) and item.get("operation")
    ]
    return [
        {
            "operation": str(item["operation"]),
            "parameters": dict(item.get("parameters") or {}),
            "reason": str(item.get("reason") or ""),
            "_advisory": advisory,
        }
        for item, advisory in (explicit + advised)[:MAX_OPERATIONS]
    ]


class DataProcessingWorkflow:
    """Inspect, plan, execute, validate, and publish through LangGraph."""

    def __init__(
        self,
        *,
        strategy_advisor: Optional[DataStrategyAdvisor] = None,
    ) -> None:
        self.strategy_advisor = strategy_advisor
        self._dataset = None
        self._profile = None
        self._policy = QualityPolicy()
        self._plan = None
        self._results = []
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(DataProcessingGraphState)
        graph.add_node("inspect", self._inspect)
        graph.add_node("advise", self._advise)
        graph.add_node("plan", self._build_plan)
        graph.add_node("execute", self._execute)
        graph.add_node("publish", self._publish)
        graph.add_node("fail", self._fail)
        graph.add_edge(START, "inspect")
        graph.add_edge("inspect", "advise")
        graph.add_edge("advise", "plan")
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

    def _inspect(self, state: DataProcessingGraphState) -> Dict[str, Any]:
        self._dataset = infer_dataset_spec(
            state["dataset_uri"],
            dataset_type=state.get("dataset_type", "auto"),
            task_type=state.get("task_type", "generic"),
        )
        self._policy = QualityPolicy.from_dict(state.get("quality_policy"))
        self._profile = profile_dataset(self._dataset, self._policy)
        return {"profile": self._profile.to_dict()}

    def _advise(self, state: DataProcessingGraphState) -> Dict[str, Any]:
        if self.strategy_advisor is None:
            return {"advice": {}}
        try:
            advice = self.strategy_advisor(dict(state))
        except Exception as exc:
            advice = {"advice_error": f"{type(exc).__name__}: {exc}"}
        return {"advice": dict(advice or {})}

    def _build_plan(self, state: DataProcessingGraphState) -> Dict[str, Any]:
        requested = _requested_operations(state)
        self._plan = build_processing_plan(
            self._profile,
            state.get("target_goal", ""),
            requested,
            self._policy,
        )
        return {"plan": self._plan.to_dict()}

    def _execute(self, state: DataProcessingGraphState) -> Dict[str, Any]:
        output_root = Path(state["output_path"]).parent / "processed"
        self._results = execute_plan(
            self._plan,
            output_root=output_root,
            initial_profile=self._profile,
        )
        route = (
            "fail"
            if not self._results
            or any(item.status == "failed" for item in self._results)
            else "publish"
        )
        error = next(
            (item.error for item in self._results if item.status == "failed"),
            None,
        )
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
        return {
            "published_version": version.to_dict(),
            "status": "success",
            "error": None,
        }

    @staticmethod
    def _fail(state: DataProcessingGraphState) -> Dict[str, Any]:
        return {
            "status": "failed",
            "error": state.get("error") or "data processing failed",
        }


__all__ = [
    "DataProcessingGraphState",
    "DataProcessingWorkflow",
    "DataStrategyAdvisor",
]
