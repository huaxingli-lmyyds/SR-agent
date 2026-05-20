"""
Simple A2A-style messaging primitives for agent coordination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


@dataclass
class A2AMessage:
    sender: str
    recipient: str
    type: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.type,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
        }


class A2AChannel:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._messages: List[A2AMessage] = []

    def send(self, message: A2AMessage) -> None:
        self._messages.append(message)

    def history(self) -> List[A2AMessage]:
        return list(self._messages)

    def last_of_type(self, message_type: str) -> Optional[A2AMessage]:
        for message in reversed(self._messages):
            if message.type == message_type:
                return message
        return None
