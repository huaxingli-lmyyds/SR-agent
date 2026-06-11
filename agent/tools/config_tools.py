"""
配置管理工具集合
提供配置文件的读取、修改、验证和比较等功能
使用 LangChain 工具接口，基于 utils 模块实现
"""

from langchain_core.tools import tool
from typing import Union, Dict, List, Optional
import json
import os
from pathlib import Path
from threading import Lock

# 导入 utils 模块
from agent.utils import (
    ConfigParser,
    backup_file,
    get_experiment_configs_dir,
    resolve_config_path,
    yaml_to_dict,
    format_nested_dict
)

# 全局配置路径
DEFAULT_CONFIG_NAME = "train_ecapa_tdnn.yaml"
CONFIG_ENV_VAR = "SR_AGENT_CONFIG_PATH"
DATA_PROCESSING_KEYS = [
    "data_folder",
    "split_ratio",
    "skip_prep",
    "sentence_len",
    "random_chunk",
    "random_segment",
    "split_speaker",
    "splits",
    "amp_th",
    "save_folder",
    "output_folder",
]

HPO_KEYS = [
    "number_of_epochs",
    "batch_size",
    "lr",
    "base_lr",
    "step_size",
    "num_workers",
]

MODEL_SECTION_KEYS = [
    "embedding_model",
    "classifier",
    "compute_cost",
    "mean_var_norm",
]


def _append_key_section(summary: str, title: str, config_dict: Dict, keys: List[str]) -> str:
    summary += f"{title}:\n"
    any_value = False
    for key in keys:
        if key in config_dict:
            summary += f"  {key}: {config_dict[key]}\n"
            any_value = True
    if not any_value:
        summary += "  (无)\n"
    summary += "\n"
    return summary

# 轻量缓存：按配置路径复用 ConfigParser，避免每次工具调用都重新实例化
_PARSER_CACHE: Dict[str, ConfigParser] = {}
_PARSER_CACHE_LOCK = Lock()


def _get_parser(config_path: Optional[Union[str, Path]] = None) -> ConfigParser:
    """获取（或创建）指定配置路径的 ConfigParser 实例。"""
    path_str = str(resolve_config_path(config_path, default_name=DEFAULT_CONFIG_NAME))

    with _PARSER_CACHE_LOCK:
        parser = _PARSER_CACHE.get(path_str)
        if parser is None:
            parser = ConfigParser(path_str)
            _PARSER_CACHE[path_str] = parser
        return parser


def _invalidate_parser_cache(config_path: Optional[Union[str, Path]] = None) -> None:
    """在配置被外部修改后使缓存失效。"""
    path_str = str(resolve_config_path(config_path, default_name=DEFAULT_CONFIG_NAME))
    with _PARSER_CACHE_LOCK:
        _PARSER_CACHE.pop(path_str, None)


def _resolve_config_path(config_path: Optional[Union[str, Path]] = None) -> str:
    """解析配置路径：参数优先，其次环境变量，最后默认路径。"""
    if config_path:
        return str(resolve_config_path(config_path))

    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        return str(resolve_config_path(env_path))

    return str(resolve_config_path(default_name=DEFAULT_CONFIG_NAME))


@tool
def ReadConfig(config_path: Optional[str] = None) -> str:
    """
    读取当前模型配置文件的关键内容。
    
    Returns:
        str: 配置摘要
    """
    try:
        # 使用 ConfigParser 读取配置
        path = _resolve_config_path(config_path)
        parser = _get_parser(path)
        config_dict = parser.load_config()
        
        # 返回关键配置、数据处理参数和模型超参数
        summary = "当前配置 (数据处理 + 训练超参数 + 模型结构):\n"
        summary += "=" * 80 + "\n\n"
        
        summary = _append_key_section(summary, "🧹 数据处理参数", config_dict, DATA_PROCESSING_KEYS)
        summary = _append_key_section(summary, "🎯 训练超参数", config_dict, HPO_KEYS)
        
        # 模型结构参数
        summary += "🏗️  模型结构参数:\n"

        for section in MODEL_SECTION_KEYS:
            if section in config_dict:
                value = config_dict[section]
                if isinstance(value, dict):
                    summary += f"\n  {section}:\n"
                    # 格式化显示嵌套结构
                    formatted = format_nested_dict(value, indent=4)
                    summary += formatted
                else:
                    summary += f"  {section}: {value}\n"
        
        return summary
    except Exception as e:
        return f"❌ 读取配置失败: {str(e)}"


@tool
def UpdateConfig(
    config_json: Union[str, Dict],
    persist: bool = True,
    create_backup: bool = True,
    config_path: Optional[str] = None,
) -> str:
    """
    使用 JSON 形式的配置更新 ECAPA-TDNN YAML 文件。

    这个工具同时服务于数据处理智能体和超参数智能体，通常用于修改：
    - 数据处理相关字段，例如 split_ratio, sentence_len, skip_prep, random_chunk, split_speaker
    - 训练超参数相关字段，例如 number_of_epochs, batch_size, lr, base_lr, max_lr, step_size
    - 模型结构相关字段，例如 embedding_model, classifier, compute_cost
    
    参数:
        config_json: JSON 字符串或 dict，表示要更新的字段
            例如: '{"lr": 0.0005, "classifier": {"input_size": 200}}'
        persist: 是否写回文件（True）或仅预览（False），默认 True
        create_backup: 是否在修改前创建备份，默认 True
        config_path: 可选配置文件路径（优先于环境变量与默认路径）
    
    Returns:
        str: 操作结果描述
    """
    try:
        if isinstance(config_json, str):
            updates = json.loads(config_json)
        elif isinstance(config_json, dict):
            updates = config_json
        else:
            return json.dumps({"status": "failed", "error": "config_json must be a JSON object"}, ensure_ascii=False)
        if not isinstance(updates, dict) or not updates:
            return json.dumps({"status": "failed", "error": "config_json must be a non-empty JSON object"}, ensure_ascii=False)
        
        # 使用 ConfigParser 更新配置
        path = _resolve_config_path(config_path)
        parser = _get_parser(path)
        
        # 创建备份
        backup_path = None
        if create_backup and persist:
            backup_path = backup_file(path, backup_dir=get_experiment_configs_dir())
        
        # 更新配置
        parser.update_config(updates, persist=persist, create_backup=False)

        # 写回后主动失效，避免其他调用拿到旧缓存
        if persist:
            _invalidate_parser_cache(path)
        
        backup_info = f"\n✅ 配置已备份到: {backup_path}" if backup_path else ""
        return (
            "✅ 配置已更新\n"
            f"修改内容: {updates}\n"
            f"persist={persist}{backup_info}"
        )
    
    except Exception as e:
        return f"❌ 修改配置失败: {str(e)}"


@tool
def ListConfigParameters(path: Optional[str] = None) -> str:
    """
    列出配置文件中的所有参数及其当前值。
    
    参数:
        path: 配置文件路径，如果为 None 则使用当前配置
    
    Returns:
        str: 参数列表
    """
    try:
        config_path = Path(_resolve_config_path(path))
        parser = _get_parser(config_path)
        config_dict = parser.load_config()
        
        summary = f"\n📋 配置参数列表 - {config_path.name}\n"
        summary += "=" * 80 + "\n\n"

        summary += "🧹 数据处理参数候选:\n"
        for key in DATA_PROCESSING_KEYS:
            if key in config_dict:
                summary += f"  {key}: {config_dict[key]}\n"
        summary += "\n"

        summary += "🎯 训练超参数候选:\n"
        for key in HPO_KEYS:
            if key in config_dict:
                summary += f"  {key}: {config_dict[key]}\n"
        summary += "\n"
        
        def _list_params(d, prefix="", indent=2):
            """递归列出参数"""
            items = []
            indent_str = " " * indent
            for key, value in d.items():
                full_key = f"{prefix}{key}" if prefix else key
                if isinstance(value, dict):
                    items.append(f"\n{indent_str}{full_key}:")
                    items.append(_list_params(value, f"{full_key}.", indent + 2))
                elif isinstance(value, (list, tuple)):
                    items.append(f"{indent_str}{full_key}: {list(value)}")
                else:
                    items.append(f"{indent_str}{full_key}: {value}")
            return "\n".join(items)
        
        summary += _list_params(config_dict)
        return summary
    
    except Exception as e:
        return f"❌ 列出参数失败: {str(e)}"


@tool
def GetConfigStructure(config_path: Optional[str] = None) -> str:
    """
    获取配置文件的结构信息，展示所有可配置的参数及其类型。
    
    Returns:
        str: 配置结构说明
    """
    try:
        path = _resolve_config_path(config_path)
        parser = _get_parser(path)
        config_dict = parser.load_config()
        
        summary = "\n📋 ECAPA-TDNN 配置文件结构\n"
        summary += "=" * 80 + "\n\n"
        
        def _describe_structure(d, prefix="", indent=0):
            """递归描述结构"""
            items = []
            indent_str = " " * indent
            for key, value in d.items():
                full_key = f"{prefix}{key}" if prefix else key
                if isinstance(value, dict):
                    items.append(f"{indent_str}{full_key} (dict)")
                    items.append(_describe_structure(value, f"{full_key}.", indent + 2))
                elif isinstance(value, list):
                    items.append(f"{indent_str}{full_key} (list[{len(value)}])")
                elif isinstance(value, tuple):
                    items.append(f"{indent_str}{full_key} (tuple[{len(value)}])")
                elif isinstance(value, bool):
                    items.append(f"{indent_str}{full_key} (bool) = {value}")
                elif isinstance(value, int):
                    items.append(f"{indent_str}{full_key} (int) = {value}")
                elif isinstance(value, float):
                    items.append(f"{indent_str}{full_key} (float) = {value}")
                elif isinstance(value, str):
                    items.append(f"{indent_str}{full_key} (str) = {value}")
                else:
                    items.append(f"{indent_str}{full_key} ({type(value).__name__}) = {value}")
            return "\n".join(items)
        
        summary += _describe_structure(config_dict)
        return summary
    
    except Exception as e:
        return f"❌ 获取配置结构失败: {str(e)}"


@tool
def ResetConfig(config_path: Optional[str] = None) -> str:
    """
    重置配置文件到默认值（恢复到最初的备份）。
    
    Returns:
        str: 重置操作结果
    """
    try:
        path = _resolve_config_path(config_path)
        # 查找最初的备份
        backup_dir = get_experiment_configs_dir()
        if not backup_dir.exists():
            return "❌ 没有找到备份目录"
        
        config_stem = Path(path).stem
        backups = sorted(
            backup_dir.glob(f"{config_stem}.backup_*{Path(path).suffix}"),
            key=lambda p: p.stat().st_mtime,
        )
        if not backups:
            return "❌ 没有找到备份文件"
        
        # 使用最早的备份
        original_backup = backups[0]
        
        # 备份当前配置
        current_backup = backup_file(path, backup_dir=get_experiment_configs_dir())
        
        # 恢复原始配置
        import shutil
        shutil.copy2(original_backup, path)

        # 文件被外部 copy 覆盖，主动失效缓存
        _invalidate_parser_cache(path)
        
        return f"""✅ 配置已重置
            恢复自: {original_backup}
            当前配置已备份到: {current_backup}"""
    
    except Exception as e:
        return f"❌ 重置配置失败: {str(e)}"


# 导出所有工具
__all__ = [
    'ReadConfig',
    'UpdateConfig',
    'ListConfigParameters',
    'GetConfigStructure',
    'ResetConfig',
]
