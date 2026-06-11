"""Generic registration, dispatch, and completion policies for coordinated agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from agent.agents.communication import AgentTaskRequest, AgentTaskResult, MessageService


class CoordinatedAgent(Protocol):
    """Minimal interface required by the coordinator."""

    def execute_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        ...


AgentFactory = Callable[[], CoordinatedAgent]


@dataclass(frozen=True)
class AgentRegistration:
    """An agent factory and the actions it is allowed to execute."""

    agent_type: str
    actions: Sequence[str]
    factory: AgentFactory = field(repr=False, compare=False)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "actions": list(self.actions),
            "description": self.description,
        }


class AgentRegistry:
    """Explicit registry used to discover and construct specialized agents."""

    def __init__(self) -> None:
        self._registrations: Dict[str, AgentRegistration] = {}

    def register(self, registration: AgentRegistration) -> None:
        if not registration.agent_type:
            raise ValueError("agent_type is required")
        if not registration.actions:
            raise ValueError(f"agent '{registration.agent_type}' must expose at least one action")
        self._registrations[registration.agent_type] = registration

    def get(self, agent_type: str) -> AgentRegistration:
        try:
            return self._registrations[agent_type]
        except KeyError as exc:
            available = ", ".join(sorted(self._registrations)) or "none"
            raise KeyError(f"unknown agent_type '{agent_type}'; available: {available}") from exc

    def create(self, agent_type: str, action: str) -> CoordinatedAgent:
        registration = self.get(agent_type)
        if action not in registration.actions:
            allowed = ", ".join(registration.actions)
            raise ValueError(
                f"agent '{agent_type}' does not support action '{action}'; allowed: {allowed}"
            )
        return registration.factory()

    def describe(self) -> List[Dict[str, Any]]:
        return [
            self._registrations[key].to_dict()
            for key in sorted(self._registrations)
        ]


@dataclass
class TaskExecutionRecord:
    """Serializable record of one dispatched agent task."""

    agent_type: str
    action: str
    request: Dict[str, Any]
    result: Dict[str, Any]
    started_at: str
    completed_at: str

    @property
    def status(self) -> str:
        return str(self.result.get("status", "unknown"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TaskDispatcher:
    """Dispatch structured requests and record correlated messages."""

    def __init__(
        self,
        registry: AgentRegistry,
        message_service: MessageService,
        sender: str = "coordinator",
    ) -> None:
        self.registry = registry
        self.message_service = message_service
        self.sender = sender

    def dispatch(
        self,
        agent_type: str,
        request: AgentTaskRequest,
    ) -> TaskExecutionRecord:
        started_at = datetime.now().isoformat()
        request_message = self.message_service.send_task(self.sender, agent_type, request)
        try:
            agent = self.registry.create(agent_type, request.action)
            result = agent.execute_task(request)
        except Exception as exc:
            result = AgentTaskResult(
                status="failed",
                error=str(exc),
                experiment_ids=request.experiment_ids,
                request_id=request.request_id,
            )
        self.message_service.send_result(agent_type, self.sender, request_message, result)
        return TaskExecutionRecord(
            agent_type=agent_type,
            action=request.action,
            request=request.to_dict(),
            result=result.to_dict(),
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
        )


@dataclass(frozen=True)
class CompletionDecision:
    """Result of evaluating whether an orchestration run may be completed."""

    complete: bool
    status: str
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CompletionPolicy:
    """Require successful execution of configured agents and reject failures."""

    def __init__(
        self,
        required_agent_types: Optional[Iterable[str]] = None,
        fail_on_any_error: bool = True,
    ) -> None:
        self.required_agent_types = set(required_agent_types or [])
        self.fail_on_any_error = fail_on_any_error

    def evaluate(self, records: Sequence[TaskExecutionRecord]) -> CompletionDecision:
        failed = [
            f"{record.agent_type}:{record.action}"
            for record in records
            if record.status == "failed"
        ]
        succeeded_agents = {
            record.agent_type
            for record in records
            if record.status == "success"
        }
        missing = sorted(self.required_agent_types - succeeded_agents)

        if failed and self.fail_on_any_error:
            return CompletionDecision(
                complete=False,
                status="failed",
                reasons=[f"failed tasks: {', '.join(failed)}"],
            )
        if missing:
            return CompletionDecision(
                complete=False,
                status="incomplete",
                reasons=[f"required agents not completed: {', '.join(missing)}"],
            )
        return CompletionDecision(complete=True, status="success")


__all__ = [
    "AgentRegistration",
    "AgentRegistry",
    "CompletionDecision",
    "CompletionPolicy",
    "CoordinatedAgent",
    "TaskDispatcher",
    "TaskExecutionRecord",
]
