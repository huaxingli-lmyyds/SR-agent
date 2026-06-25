"""Model-agnostic data processing domain APIs."""

from importlib import import_module

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
from .handoff import build_data_handoff, resolve_data_handoff
_WORKFLOW_EXPORTS = {
    "DataProcessingDecisionPolicy": (
        "agent.data_processing.workflow",
        "DataProcessingDecisionPolicy",
    ),
    "DataProcessingWorkflow": (
        "agent.data_processing.workflow",
        "DataProcessingWorkflow",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _WORKFLOW_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

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
    "resolve_data_handoff",
    "build_data_handoff",
    "DataProcessingDecisionPolicy",
    "DataProcessingWorkflow",
]
