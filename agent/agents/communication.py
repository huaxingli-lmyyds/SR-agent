"""Structured synchronous communication primitives for agent coordination."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import uuid


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
        return asdict(self)


@dataclass
class AgentTaskResult:
    """Task result returned by a specialized agent."""

    status: str
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    experiment_ids: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    request_id: Optional[str] = None
    runtime_result: Any = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "experiment_ids": self.experiment_ids,
            "error": self.error,
            "request_id": self.request_id,
        }

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
        return asdict(self)


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
