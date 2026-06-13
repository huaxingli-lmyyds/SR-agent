"""
工具模块
提供路径管理、配置解析、实验跟踪、日志记录和性能指标分析等功能
"""

from importlib import import_module

from .path_tool import (
    get_project_root,
    get_agent_dir,
    get_configs_dir,
    get_datasets_dir,
    get_recipes_dir,
    get_experiments_dir,
    get_hpo_experiments_dir,
    get_data_processing_experiments_dir,
    get_manage_experiments_dir,
    get_results_dir,
    get_logs_dir,
    get_prep_cache_dir,
    get_memory_dir,
    get_experiment_type_dir,
    get_experiment_dir,
    get_experiment_artifact_dir,
    get_experiment_configs_dir,
    resolve_project_path,
    resolve_optional_project_path,
    resolve_config_path,
    resolve_data_path,
    resolve_config_value_path,
    is_remote_path,
    ensure_dir,
    find_result_dirs,
    find_log_files,
    find_scores_file,
    get_experiment_log_path,
    list_directories,
    copy_file,
    get_config_file,
    get_train_script,
    get_eval_script,
    backup_file,
    yaml_to_dict,
    format_nested_dict
)

from .config_parser import (
    ConfigParser,
    load_config,
    validate_config,
    compare_configs
)

from .experiment_tracker import (
    BaseExperimentRecord,
    HPOExperimentRecord,
    DataProcessingExperimentRecord,
    OrchestrationExperimentRecord,
    ExperimentTracker,
    create_experiment,
    list_experiments,
    find_best_experiment,
    get_experiment_stats
)

from .metrics import (
    MetricsExtractor,
    MetricsCalculator,
    MetricsComparator,
    MetricsVisualizer,
    extract_log_metrics,
    extract_scores_data,
    compute_metrics_from_scores,
    compare_experiments
)

from .reward import (
    compute_reward,
    compute_objective_reward,
)

_OPTIONAL_EXPORTS = {
    "Logger": ("agent.utils.logger", "Logger"),
    "get_logger": ("agent.utils.logger", "get_logger"),
    "AgentLogger": ("agent.utils.logger", "AgentLogger"),
    "build_agent_logging_middleware": (
        "agent.utils.agent_middleware",
        "build_agent_logging_middleware",
    ),
}


def __getattr__(name):
    try:
        module_name, attribute = _OPTIONAL_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    # path_tool
    'get_project_root',
    'get_agent_dir',
    'get_configs_dir',
    'get_datasets_dir',
    'get_recipes_dir',
    'get_experiments_dir',
    'get_hpo_experiments_dir',
    'get_data_processing_experiments_dir',
    'get_manage_experiments_dir',
    'get_results_dir',
    'get_logs_dir',
    'get_prep_cache_dir',
    'get_memory_dir',
    'get_experiment_type_dir',
    'get_experiment_dir',
    'get_experiment_artifact_dir',
    'get_experiment_configs_dir',
    'resolve_project_path',
    'resolve_optional_project_path',
    'resolve_config_path',
    'resolve_data_path',
    'resolve_config_value_path',
    'is_remote_path',
    'ensure_dir',
    'find_result_dirs',
    'find_log_files',
    'find_scores_file',
    'get_experiment_log_path',
    'list_directories',
    'copy_file',
    'get_config_file',
    'get_train_script',
    'get_eval_script',
    'backup_file',
    'yaml_to_dict',
    'format_nested_dict',
    
    # config_parser
    'ConfigParser',
    'load_config',
    'validate_config',
    'compare_configs',
    
    # experiment_tracker
    'BaseExperimentRecord',
    'HPOExperimentRecord',
    'DataProcessingExperimentRecord',
    'OrchestrationExperimentRecord',
    'ExperimentTracker',
    'create_experiment',
    'list_experiments',
    'find_best_experiment',
    'get_experiment_stats',
    
    # logger
    'Logger',
    'get_logger',
    'AgentLogger',
    'build_agent_logging_middleware',
    
    # metrics
    'MetricsExtractor',
    'MetricsCalculator',
    'MetricsComparator',
    'MetricsVisualizer',
    'extract_log_metrics',
    'extract_scores_data',
    'compute_metrics_from_scores',
    'compare_experiments',
    'compute_reward',
    'compute_objective_reward',
]
