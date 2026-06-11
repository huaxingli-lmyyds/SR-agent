"""Model-agnostic data processing domain APIs."""

from .contracts import (
    DataIssue,
    DataOperation,
    DataOperationResult,
    DataProcessingPlan,
    DataProfile,
    DatasetSpec,
    DatasetVersion,
)
from .registry import PROCESSORS, DataProcessorRegistry, register_processor
from .service import (
    build_processing_plan,
    execute_plan,
    infer_dataset_spec,
    plan_from_dict,
    profile_dataset,
    profile_from_dict,
    publish_dataset_version,
)

__all__ = [
    "DatasetSpec",
    "DataIssue",
    "DataProfile",
    "DataOperation",
    "DataProcessingPlan",
    "DataOperationResult",
    "DatasetVersion",
    "DataProcessorRegistry",
    "PROCESSORS",
    "register_processor",
    "infer_dataset_spec",
    "profile_dataset",
    "build_processing_plan",
    "execute_plan",
    "publish_dataset_version",
    "profile_from_dict",
    "plan_from_dict",
]
