"""LangChain tool package with lazy public exports."""

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ReadConfig": ("agent.tools.config_tools", "ReadConfig"),
    "UpdateConfig": ("agent.tools.config_tools", "UpdateConfig"),
    "ListConfigParameters": ("agent.tools.config_tools", "ListConfigParameters"),
    "GetConfigStructure": ("agent.tools.config_tools", "GetConfigStructure"),
    "ResetConfig": ("agent.tools.config_tools", "ResetConfig"),
    "TrainModel": ("agent.tools.training_tools", "TrainModel"),
    "EvaluateModel": ("agent.tools.training_tools", "EvaluateModel"),
    "AnalyzeResults": ("agent.tools.training_tools", "AnalyzeResults"),
    "RunEvaluation": ("agent.tools.evaluation_tools", "RunEvaluation"),
    "GetExperimentResults": ("agent.tools.experiment_history_tools", "GetExperimentResults"),
    "ListExperiments": ("agent.tools.experiment_history_tools", "ListExperiments"),
    "CompareExperiments": ("agent.tools.experiment_history_tools", "CompareExperiments"),
    "CompareHPOExperiments": ("agent.tools.experiment_history_tools", "CompareHPOExperiments"),
    "GetHPOExperimentResults": ("agent.tools.experiment_history_tools", "GetHPOExperimentResults"),
    "ListHPOExperiments": ("agent.tools.experiment_history_tools", "ListHPOExperiments"),
    "CompareDataProcessingExperiments": (
        "agent.tools.experiment_history_tools",
        "CompareDataProcessingExperiments",
    ),
    "GetDataProcessingExperimentResults": (
        "agent.tools.experiment_history_tools",
        "GetDataProcessingExperimentResults",
    ),
    "ListDataProcessingExperiments": (
        "agent.tools.experiment_history_tools",
        "ListDataProcessingExperiments",
    ),
    "CompareOrchestrationExperiments": (
        "agent.tools.experiment_history_tools",
        "CompareOrchestrationExperiments",
    ),
    "GetOrchestrationExperimentResults": (
        "agent.tools.experiment_history_tools",
        "GetOrchestrationExperimentResults",
    ),
    "ListOrchestrationExperiments": (
        "agent.tools.experiment_history_tools",
        "ListOrchestrationExperiments",
    ),
    "PrepareVoxCelebData": ("agent.tools.speechbrain_data_tools", "PrepareVoxCelebData"),
    "InspectDataset": ("agent.tools.dataset_tools", "InspectDataset"),
    "BuildDataProcessingPlan": ("agent.tools.dataset_tools", "BuildDataProcessingPlan"),
    "ExecuteDataProcessingPlan": ("agent.tools.dataset_tools", "ExecuteDataProcessingPlan"),
    "PublishDatasetVersion": ("agent.tools.dataset_tools", "PublishDatasetVersion"),
    "ListDataProcessors": ("agent.tools.dataset_tools", "ListDataProcessors"),
    "AnalyzeTrainingCurves": ("agent.tools.training_diagnostics_tools", "AnalyzeTrainingCurves"),
    "DiagnoseFitStatus": ("agent.tools.training_diagnostics_tools", "DiagnoseFitStatus"),
    "ScoreExperiment": ("agent.tools.reward_tools", "ScoreExperiment"),
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
