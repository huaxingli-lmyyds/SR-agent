"""Model optimization agent package with lazy public exports."""

from importlib import import_module
from typing import Dict, Tuple

__version__ = "1.0.0"

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "HPOAgent": ("agent.agents.hpo_agent", "HPOAgent"),
    "create_hpo_agent": ("agent.agents.hpo_agent", "create_hpo_agent"),
    "ReadConfig": ("agent.tools.config_tools", "ReadConfig"),
    "UpdateConfig": ("agent.tools.config_tools", "UpdateConfig"),
    "ListConfigParameters": ("agent.tools.config_tools", "ListConfigParameters"),
    "GetConfigStructure": ("agent.tools.config_tools", "GetConfigStructure"),
    "ResetConfig": ("agent.tools.config_tools", "ResetConfig"),
    "TrainModel": ("agent.tools.training_tools", "TrainModel"),
    "EvaluateModel": ("agent.tools.training_tools", "EvaluateModel"),
    "AnalyzeResults": ("agent.tools.training_tools", "AnalyzeResults"),
    "CompareExperiments": ("agent.tools.experiment_history_tools", "CompareExperiments"),
    "RunEvaluation": ("agent.tools.evaluation_tools", "RunEvaluation"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
