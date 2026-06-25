"""Model-agnostic data processing contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetSpec:
    dataset_id: str
    dataset_type: str
    source_uri: str
    format: str = "directory"
    task_type: str = "generic"
    version: Optional[str] = None
    splits: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataIssue:
    code: str
    severity: str
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_operation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataProfile:
    dataset: DatasetSpec
    sample_count: int
    schema: Dict[str, Any] = field(default_factory=dict)
    distributions: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[DataIssue] = field(default_factory=list)
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["dataset"] = self.dataset.to_dict()
        data["issues"] = [issue.to_dict() for issue in self.issues]
        return data


@dataclass
class DataOperation:
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    expected_effect: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataProcessingPlan:
    dataset: DatasetSpec
    operations: List[DataOperation] = field(default_factory=list)
    rejected_operations: List[Dict[str, Any]] = field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    target_goal: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["dataset"] = self.dataset.to_dict()
        data["operations"] = [operation.to_dict() for operation in self.operations]
        return data


@dataclass
class DataOperationResult:
    status: str
    operation: str
    input_dataset_version: Optional[str] = None
    output_dataset_version: Optional[str] = None
    output_dataset_uri: Optional[str] = None
    consumer_ready: bool = False
    before_metrics: Dict[str, Any] = field(default_factory=dict)
    after_metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetVersion:
    dataset_id: str
    version: str
    source_uri: str
    output_uri: Optional[str] = None
    consumer_uri: Optional[str] = None
    consumption_status: str = "source_unchanged"
    consumption_reason: Optional[str] = None
    parent_version: Optional[str] = None
    operations: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    created_by: str = "data_processing_agent"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
