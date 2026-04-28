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
    get_config_file,
    backup_file,
    ensure_dir,
    get_experiments_dir,
    yaml_to_dict,
    format_nested_dict
)

# 全局配置路径
DEFAULT_CONFIG_NAME = "train_ecapa_tdnn.yaml"
CONFIG_ENV_VAR = "SR_AGENT_CONFIG_PATH"
CONFIG_PATH = str(get_config_file(DEFAULT_CONFIG_NAME))

# 轻量缓存：按配置路径复用 ConfigParser，避免每次工具调用都重新实例化
_PARSER_CACHE: Dict[str, ConfigParser] = {}
_PARSER_CACHE_LOCK = Lock()


def _get_parser(config_path: Optional[Union[str, Path]] = None) -> ConfigParser:
    """获取（或创建）指定配置路径的 ConfigParser 实例。"""
    path_str = str(Path(config_path or CONFIG_PATH).resolve())

    with _PARSER_CACHE_LOCK:
        parser = _PARSER_CACHE.get(path_str)
        if parser is None:
            parser = ConfigParser(path_str)
            _PARSER_CACHE[path_str] = parser
        return parser


def _invalidate_parser_cache(config_path: Optional[Union[str, Path]] = None) -> None:
    """在配置被外部修改后使缓存失效。"""
    path_str = str(Path(config_path or CONFIG_PATH).resolve())
    with _PARSER_CACHE_LOCK:
        _PARSER_CACHE.pop(path_str, None)


def _resolve_config_path(config_path: Optional[Union[str, Path]] = None) -> str:
    """解析配置路径：参数优先，其次环境变量，最后默认路径。"""
    if config_path:
        return str(Path(config_path).resolve())

    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        return str(Path(env_path).resolve())

    return str(Path(CONFIG_PATH).resolve())


@tool
def ReadConfig(config_path: Optional[str] = None) -> str:
    """
    读取当前 ECAPA-TDNN 配置文件的完整内容，包括模型结构信息。
    
    Returns:
        str: 配置文件的完整 YAML 内容（包括模型结构）
    """
    try:
        # 使用 ConfigParser 读取配置
        path = _resolve_config_path(config_path)
        parser = _get_parser(path)
        config_dict = parser.load_config()
        
        # 返回关键配置和模型结构
        summary = "当前配置 (包括模型结构):\n"
        summary += "=" * 80 + "\n\n"
        
        # 基础训练参数
        basic_params = ['lr', 'batch_size', 'number_of_epochs', 'step_size', 'seed']
        summary += "📊 基础训练参数:\n"
        for param in basic_params:
            if param in config_dict:
                summary += f"  {param}: {config_dict[param]}\n"
        summary += "\n"
        
        # 模型结构参数
        model_sections = ['embedding_model', 'classifier', 'compute_cost', 'opt_class']
        summary += "🏗️  模型结构参数:\n"
        
        for section in model_sections:
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
            return "config_json 必须是 JSON 字符串或 dict 类型"
        
        # 使用 ConfigParser 更新配置
        path = _resolve_config_path(config_path)
        parser = _get_parser(path)
        
        # 创建备份
        backup_path = None
        if create_backup and persist:
            backup_path = backup_file(path)
        
        # 更新配置
        parser.update_config(updates, persist=persist, create_backup=False)

        # 写回后主动失效，避免其他调用拿到旧缓存
        if persist:
            _invalidate_parser_cache(path)
        
        backup_info = f"\n✅ 配置已备份到: {backup_path}" if backup_path else ""
        return f"✅ 配置已更新: {updates} (persist={persist}){backup_info}"
    
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
                elif isinstance(value, int):
                    items.append(f"{indent_str}{full_key} (int) = {value}")
                elif isinstance(value, float):
                    items.append(f"{indent_str}{full_key} (float) = {value}")
                elif isinstance(value, str):
                    items.append(f"{indent_str}{full_key} (str) = {value}")
                elif isinstance(value, bool):
                    items.append(f"{indent_str}{full_key} (bool) = {value}")
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
        backup_dir = get_experiments_dir() / "configs"
        if not backup_dir.exists():
            return "❌ 没有找到备份目录"
        
        backups = sorted(backup_dir.glob("*.yaml"), key=lambda p: p.stat().st_mtime)
        if not backups:
            return "❌ 没有找到备份文件"
        
        # 使用最早的备份
        original_backup = backups[0]
        
        # 备份当前配置
        current_backup = backup_file(path)
        
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