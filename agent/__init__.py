"""
智能体主模块
提供超参数优化智能体的所有功能和工具
"""

# 导入智能体
from .agents import (
    ReActHPOAgent,
    create_react_agent
)

# 导入工具函数
from .tools import (
    # 配置工具
    ReadConfig,
    UpdateConfig,
    ListConfigParameters,
    GetConfigStructure,
    ResetConfig,
    
    # 训练工具
    TrainModel,
    EvaluateModel,
    AnalyzeResults,
    CompareExperiments,
    
    # 评估工具
    RunEvaluation,
    GetEvaluationResults,
    CompareEvaluations,
    ListEvaluations,
)

# 导入实用函数
from .utils import (
    # 路径工具
    get_project_root,
    get_agent_dir,
    get_configs_dir,
    get_datasets_dir,
    get_recipes_dir,
    get_experiments_dir,
    get_results_dir,
    get_config_file,
    get_train_script,
    get_eval_script,
    
    # 配置解析
    load_config,
    validate_config,
    compare_configs,
    
    # 实验跟踪
    create_experiment,
    list_experiments,
    find_best_experiment,
    get_experiment_stats,
    
    # 日志记录
    ExperimentLogger,
)

__version__ = "1.0.0"

__all__ = [
    # 智能体
    'ReActHPOAgent',
    'create_react_agent',
    # 工具
    'ReadConfig',
    'UpdateConfig',
    'ListConfigParameters',
    'GetConfigStructure',
    'ResetConfig',
    'TrainModel',
    'EvaluateModel',
    'AnalyzeResults',
    'CompareExperiments',
    'RunEvaluation',
    'GetEvaluationResults',
    'CompareEvaluations',
    'ListEvaluations',
    # 路径工具
    'get_project_root',
    'get_agent_dir',
    'get_configs_dir',
    'get_datasets_dir',
    'get_recipes_dir',
    'get_experiments_dir',
    'get_results_dir',
    'get_config_file',
    'get_train_script',
    'get_eval_script',
    # 配置解析
    'load_config',
    'validate_config',
    'compare_configs',
    # 实验跟踪
    'create_experiment',
    'list_experiments',
    'find_best_experiment',
    'get_experiment_stats',
    # 日志记录
    'ExperimentLogger'
]