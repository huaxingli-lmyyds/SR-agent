"""Infrastructure utilities exposed through lazy compatibility exports."""

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {}

for name in (
    "get_project_root",
    "get_agent_dir",
    "get_configs_dir",
    "get_datasets_dir",
    "get_recipes_dir",
    "get_experiments_dir",
    "get_hpo_experiments_dir",
    "get_data_processing_experiments_dir",
    "get_manage_experiments_dir",
    "get_results_dir",
    "get_logs_dir",
    "get_prep_cache_dir",
    "get_memory_dir",
    "get_experiment_type_dir",
    "get_experiment_dir",
    "get_experiment_artifact_dir",
    "get_experiment_configs_dir",
    "resolve_project_path",
    "resolve_optional_project_path",
    "resolve_config_path",
    "resolve_data_path",
    "resolve_config_value_path",
    "is_remote_path",
    "ensure_dir",
    "find_result_dirs",
    "find_log_files",
    "find_scores_file",
    "get_experiment_log_path",
    "list_directories",
    "copy_file",
    "get_config_file",
    "get_train_script",
    "get_eval_script",
    "backup_file",
    "yaml_to_dict",
    "format_nested_dict",
):
    _EXPORTS[name] = ("agent.utils.path_tool", name)

for name in ("ConfigParser", "load_config", "validate_config", "compare_configs"):
    _EXPORTS[name] = ("agent.utils.config_parser", name)

for name in (
    "BaseExperimentRecord",
    "HPOExperimentRecord",
    "DataProcessingExperimentRecord",
    "OrchestrationExperimentRecord",
    "ExperimentTracker",
    "create_experiment",
    "list_experiments",
    "find_best_experiment",
    "get_experiment_stats",
):
    _EXPORTS[name] = ("agent.utils.experiment_tracker", name)

for name in (
    "MetricsExtractor",
    "MetricsCalculator",
    "MetricsComparator",
    "MetricsVisualizer",
    "extract_log_metrics",
    "extract_scores_data",
    "compute_metrics_from_scores",
    "compare_experiments",
):
    _EXPORTS[name] = ("agent.utils.metrics", name)

for name in ("compute_reward", "compute_objective_reward"):
    _EXPORTS[name] = ("agent.utils.reward", name)

for name in ("Logger", "get_logger", "AgentLogger"):
    _EXPORTS[name] = ("agent.utils.logger", name)

_EXPORTS["build_agent_logging_middleware"] = (
    "agent.utils.agent_middleware",
    "build_agent_logging_middleware",
)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
