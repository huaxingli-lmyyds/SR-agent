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
        return asdict(self)


@dataclass
class QualityPolicy:
    """Bounded defaults for deterministic dataset quality checks."""

    max_audio_files: int = 1000
    max_signal_files: int = 200
    max_signal_seconds: float = 30.0
    min_duration_seconds: float = 0.1
    max_duration_seconds: float = 30.0
    expected_sample_rates: List[int] = field(default_factory=lambda: [16000])
    expected_channels: List[int] = field(default_factory=lambda: [1])
    max_silence_ratio: float = 0.95
    max_clipping_ratio: float = 0.01
    near_zero_rms: float = 1e-5
    min_samples_per_speaker: int = 2
    require_disjoint_speakers: bool = True
    require_disjoint_files: bool = True
    require_valid_trials: bool = True
    hash_file_limit: int = 10000

    def __post_init__(self) -> None:
        for name in ("max_audio_files", "max_signal_files", "hash_file_limit"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.max_signal_seconds <= 0:
            raise ValueError("max_signal_seconds must be positive")
        if (
            self.min_duration_seconds < 0
            or self.max_duration_seconds <= self.min_duration_seconds
        ):
            raise ValueError("duration policy bounds are invalid")
        for name in ("max_silence_ratio", "max_clipping_ratio"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.near_zero_rms < 0:
            raise ValueError("near_zero_rms cannot be negative")
        if self.min_samples_per_speaker <= 0:
            raise ValueError("min_samples_per_speaker must be positive")
        for name in (
            "require_disjoint_speakers",
            "require_disjoint_files",
            "require_valid_trials",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("expected_sample_rates", "expected_channels"):
            values = getattr(self, name)
            if (
                not isinstance(values, (list, tuple))
                or not values
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in values
                )
            ):
                raise ValueError(f"{name} must contain positive integers")

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "QualityPolicy":
        if not value:
            return cls()
        allowed = cls.__dataclass_fields__
        unknown = sorted(set(value) - set(allowed))
        if unknown:
            raise ValueError(
                f"unknown quality policy fields: {', '.join(unknown)}"
            )
        return cls(**value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityDecision:
    status: str
    training_allowed: bool
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OperationImpact:
    operation: str
    scanned_samples: int = 0
    affected_samples: int = 0
    estimated_output_bytes: int = 0
    creates_new_version: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataOperation:
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataProcessingPlan:
    dataset: DatasetSpec
    operations: List[DataOperation] = field(default_factory=list)
    rejected_operations: List[Dict[str, Any]] = field(default_factory=list)
    quality_policy: Dict[str, Any] = field(default_factory=dict)
    target_goal: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataOperationResult:
    status: str
    operation: str
    output_dataset_uri: Optional[str] = None
    consumer_ready: bool = False
    before_metrics: Dict[str, Any] = field(default_factory=dict)
    after_metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    impact: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
            and value != {}
            and value != []
            and (key != "consumer_ready" or value)
        }


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
    quality_policy: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    quality_decision: Dict[str, Any] = field(default_factory=dict)
    file_hashes: Dict[str, str] = field(default_factory=dict)
    hashes_complete: bool = True
    created_at: str = ""
    created_by: str = "data_processing_agent"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
