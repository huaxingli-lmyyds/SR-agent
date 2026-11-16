"""Structured synchronous communication primitives for agent coordination."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import uuid


def json_safe(value: Any) -> Any:
    """Convert protocol payloads to JSON-native values at the boundary."""
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (datetime, Path)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class MessageType:
    TASK_REQUEST = "task.request"
    TASK_RESULT = "task.result"
    STATUS_UPDATE = "status.update"
    ERROR = "error"


@dataclass
class AgentTaskRequest:
    """Task contract passed from the coordinator to a specialized agent."""

    action: str
    objective: str
    context: Dict[str, Any] = field(default_factory=dict)
    budget: Dict[str, Any] = field(default_factory=dict)
    experiment_ids: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"request_{uuid.uuid4().hex}")

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class AgentTaskResult:
    """Fully serializable task result returned by a specialized agent."""

    status: str
    summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    experiment_ids: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return json_safe({
            "status": self.status,
            "summary": self.summary,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "recommendations": self.recommendations,
            "experiment_ids": self.experiment_ids,
            "error": self.error,
            "request_id": self.request_id,
        })

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


@dataclass
class AgentMessage:
    """Auditable message envelope for requests, results, and status updates."""

    session_id: str
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    status: str = "created"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    message_id: str = field(default_factory=lambda: f"message_{uuid.uuid4().hex}")

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


class MessageService:
    """In-process synchronous message service with request/result correlation."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._messages: List[AgentMessage] = []

    def send_task(self, sender: str, recipient: str, request: AgentTaskRequest) -> AgentMessage:
        message = AgentMessage(
            session_id=self.session_id,
            sender=sender,
            recipient=recipient,
            message_type=MessageType.TASK_REQUEST,
            correlation_id=request.request_id,
            payload=request.to_dict(),
            status="sent",
        )
        self._messages.append(message)
        return message

    def send_result(
        self,
        sender: str,
        recipient: str,
        request_message: AgentMessage,
        result: AgentTaskResult,
    ) -> AgentMessage:
        result.request_id = result.request_id or request_message.correlation_id
        message = AgentMessage(
            session_id=self.session_id,
            sender=sender,
            recipient=recipient,
            message_type=MessageType.TASK_RESULT if result.status != "failed" else MessageType.ERROR,
            correlation_id=request_message.correlation_id,
            reply_to=request_message.message_id,
            payload=result.to_dict(),
            status=result.status,
        )
        self._messages.append(message)
        return message

    def send_status(self, sender: str, recipient: str, payload: Dict[str, Any]) -> AgentMessage:
        message = AgentMessage(
            session_id=self.session_id,
            sender=sender,
            recipient=recipient,
            message_type=MessageType.STATUS_UPDATE,
            payload=payload,
            status=str(payload.get("status", "running")),
        )
        self._messages.append(message)
        return message

    def history(self) -> List[AgentMessage]:
        return list(self._messages)

    def conversation(self, correlation_id: str) -> List[AgentMessage]:
        return [
            message
            for message in self._messages
            if message.correlation_id == correlation_id
        ]
