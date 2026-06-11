"""Model-agnostic contracts shared by tools, adapters, and experiment records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class Artifact:
    type: str
    name: str
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OperationRequest:
    stage: str
    task_type: str
    model_family: str
    runner: str
    config_path: str
    data_path: Optional[str] = None
    output_dir: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationResult:
    status: str
    stage: str
    task: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    artifacts: List[Artifact] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    experiment_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)
