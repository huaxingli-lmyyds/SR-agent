"""
配置解析和管理模块
提供 YAML 配置文件的加载、解析、验证和比较等功能
"""

from pathlib import Path
from typing import Union, Dict, List, Optional, Any
from datetime import datetime
import json

# 导入路径工具
from .path_tool import (
    get_config_file,
    backup_file,
    yaml_to_dict,
    format_nested_dict
)


def parse_yaml_tags(value: Any, config: Dict) -> Any:
    """
    解析 YAML 中的特殊标签（!ref, !new:, !apply: 等）
    
    参数:
        value: 要解析的值
        config: 完整的配置字典
    
    Returns:
        解析后的值
    """
    # 处理 !ref <key> 引用（TaggedScalar 已经被转换）
    if isinstance(value, str) and value.startswith("!ref <") and value.endswith(">"):
        ref_key = value[6:-1].strip()  # 移除 "!ref <" 和 ">"
        
        # 递归解析嵌套引用
        if ref_key in config:
            resolved = parse_yaml_tags(config[ref_key], config)
            return resolved
        else:
            return value
    
    # 处理 <key> 形式的引用（简化版引用语法）
    elif isinstance(value, str) and value.startswith("<") and value.endswith(">"):
        ref_key = value[1:-1].strip()  # 移除 "<" 和 ">"
        
        # 递归解析嵌套引用
        if ref_key in config:
            resolved = parse_yaml_tags(config[ref_key], config)
            return resolved
        else:
            return value
    
    # 处理 !apply:function [args] 或 !new:class {params}
    elif isinstance(value, str) and (value.startswith("!apply:") or value.startswith("!new:")):
        # 保留原始字符串，不实例化，只保留参数结构
        return value
    
    # 处理 !PLACEHOLDER
    elif isinstance(value, str) and value == "!PLACEHOLDER":
        return "!PLACEHOLDER"
    
    # 处理包含 <key> 的字符串（如 "results/ecapa_augment/<seed>"）
    elif isinstance(value, str) and "<" in value and ">" in value:
        # 简单的字符串替换
        result = value
        import re
        # 查找所有 <key> 模式
        matches = re.findall(r'<([^>]+)>', value)
        for match in matches:
            if match in config:
                resolved = parse_yaml_tags(config[match], config)
                result = result.replace(f"<{match}>", str(resolved))
        return result
    
    # 处理 TaggedScalar 类型
    elif hasattr(value, 'value'):
        return parse_yaml_tags(value.value, config)
    
    # 处理字典
    elif isinstance(value, dict):
        resolved_dict = {}
        for key, val in value.items():
            resolved_dict[key] = parse_yaml_tags(val, config)
        return resolved_dict
    
    # 处理列表
    elif isinstance(value, list):
        return [parse_yaml_tags(item, config) for item in value]
    
    # 其他类型直接返回
    else:
        return value


def resolve_yaml_references(config: Dict) -> Dict:
    """
    解析配置中的所有 YAML 引用（!ref, !new:, !apply: 等）
    
    参数:
        config: 配置字典
    
    Returns:
        解析后的配置字典
    """
    resolved_config = {}
    
    for key, value in config.items():
        # 解析每个值
        resolved_value = parse_yaml_tags(value, config)
        resolved_config[key] = resolved_value
    
    return resolved_config


class ConfigParser:
    """配置解析器类"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置解析器
        
        参数:
            config_path: 配置文件路径，如果为 None 则使用默认配置
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = get_config_file("train_ecapa_tdnn.yaml")

        self._config_cache_raw = None
        self._config_cache = None
        self._data_folder = None  # 用于在解析前设置 data_folder

    @property
    def data_folder(self) -> Optional[str]:
        """获取 data_folder 覆盖值"""
        return self._data_folder

    @data_folder.setter
    def data_folder(self, value: Optional[str]):
        """设置 data_folder 覆盖值，并清理缓存以便重新解析"""
        self._data_folder = value
        self._config_cache = None
    
    def load_config(self, convert_to_dict: bool = True, 
                   resolve_references: bool = True) -> Union[Dict, Any]:
        """
        加载配置文件
        
        参数:
            convert_to_dict: 是否转换为标准字典，如果为 False 则保留 ruamel.yaml 的特殊类型
            resolve_references: 是否解析 YAML 引用（!ref, !new:, !apply: 等）
        
        Returns:
            配置数据（字典或原始数据）
        """
        try:
            from ruamel.yaml import YAML
            
            yaml_parser = YAML()
            yaml_parser.preserve_quotes = True
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml_parser.load(f)
            
            if config is None:
                config = {}

            # 缓存未转换、未解析的原始配置
            self._config_cache_raw = config

            # 如果设置了 data_folder，先更新配置
            if self._data_folder is not None and isinstance(config, dict):
                config['data_folder'] = self._data_folder

            if convert_to_dict:
                config_dict = yaml_to_dict(config)

                # 如果设置了 data_folder，先更新配置
                if self._data_folder is not None:
                    config_dict['data_folder'] = self._data_folder

                # 解析 YAML 引用时，缓存解析后的字典
                if resolve_references:
                    resolved = resolve_yaml_references(config_dict)
                    self._config_cache = resolved
                    return resolved

                # 缓存未解析引用的字典配置
                self._config_cache = config_dict
                return config_dict

            self._config_cache = None
            return config
        
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}") from e
    
    def save_config(self, config: Optional[Dict] = None, create_backup: bool = True) -> Path:
        """
        保存配置到文件
        
        参数:
            config: 要保存的配置字典
            create_backup: 是否在保存前创建备份
        
        Returns:
            备份文件路径（如果创建了备份）
        """
        try:
            from ruamel.yaml import YAML
            
            # 创建备份
            backup_path = None
            if create_backup:
                backup_path = backup_file(self.config_path)
            
            # 保存配置
            yaml_parser = YAML()
            yaml_parser.preserve_quotes = True

            if config is None:
                if self._config_cache_raw is None:
                    self.load_config(convert_to_dict=False, resolve_references=False)
                config_to_save = self._config_cache_raw
            else:
                config_to_save = config

            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml_parser.dump(config_to_save, f)
            
            # 清除缓存
            self._config_cache_raw = None
            self._config_cache = None
            
            return backup_path
        
        except Exception as e:
            raise RuntimeError(f"保存配置文件失败: {e}") from e
    
    def get_config(self, key: Optional[str] = None, default: Any = None) -> Any:
        """
        获取配置值
        
        参数:
            key: 配置键，支持点号分隔的嵌套键（如 "embedding_model.channels"）
            default: 默认值
        
        Returns:
            配置值
        """
        if self._config_cache is None:
            self.load_config()
        
        if key is None:
            return self._config_cache
        
        # 支持嵌套键
        keys = key.split('.')
        value = self._config_cache
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update_config(self, updates: Dict, persist: bool = True, 
                     create_backup: bool = False) -> None:
        """
        更新配置
        
        参数:
            updates: 更新的键值对字典
            persist: 是否立即保存到文件
            create_backup: 是否在保存前创建备份
        """
        if self._config_cache_raw is None:
            self.load_config(convert_to_dict=False, resolve_references=False)

        if self._config_cache is None:
            self.load_config(convert_to_dict=True, resolve_references=False)
        
        def _deep_update(orig, upd):
            """深度更新字典"""
            for k, v in upd.items():
                if k in orig and isinstance(orig[k], dict) and isinstance(v, dict):
                    _deep_update(orig[k], v)
                else:
                    orig[k] = v
        
        def _deep_update_raw(orig, upd):
            """深度更新原始配置对象"""
            for k, v in upd.items():
                if k in orig and isinstance(orig[k], dict) and isinstance(v, dict):
                    _deep_update_raw(orig[k], v)
                else:
                    orig[k] = v

        if isinstance(self._config_cache_raw, dict):
            _deep_update_raw(self._config_cache_raw, updates)

        _deep_update(self._config_cache, updates)

        if persist:
            self.save_config(None, create_backup)
    
    def validate_config(self, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        验证配置的正确性和完整性
        
        参数:
            config: 要验证的配置，如果为 None 则使用当前配置
        
        Returns:
            验证结果字典，包含:
            - valid: 是否验证通过
            - errors: 错误列表
            - warnings: 警告列表
            - info: 信息列表
        """
        if config is None:
            config = self.load_config()
        
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # 检查必需字段
        required_fields = {
            'lr': (int, float),
            'batch_size': int,
            'number_of_epochs': int,
            'embedding_model': dict,
            'classifier': dict,
        }
        
        for field, expected_type in required_fields.items():
            if field not in config:
                result["errors"].append(f"缺少必需字段: {field}")
                result["valid"] = False
            elif not isinstance(config[field], expected_type):
                if isinstance(expected_type, tuple):
                    expected_type_name = " 或 ".join(t.__name__ for t in expected_type)
                else:
                    expected_type_name = expected_type.__name__
                result["errors"].append(
                    f"字段 {field} 类型错误: 期望 {expected_type_name}, "
                    f"实际 {type(config[field]).__name__}"
                )
                result["valid"] = False
        
        # 检查数值范围
        range_checks = {
            'lr': (1e-6, 1.0, "学习率"),
            'batch_size': (1, 512, "批次大小"),
            'number_of_epochs': (1, 1000, "训练轮数"),
        }
        
        for field, (min_val, max_val, name) in range_checks.items():
            if field in config:
                value = config[field]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        result["warnings"].append(
                            f"{name} {value} 超出推荐范围 [{min_val}, {max_val}]"
                        )
        
        # 检查模型结构
        if 'embedding_model' in config and isinstance(config['embedding_model'], dict):
            emb_fields = ['channels', 'kernel_sizes', 'lin_neurons']
            for field in emb_fields:
                if field not in config['embedding_model']:
                    result["warnings"].append(
                        f"embedding_model 中缺少字段: {field}"
                    )
        
        # 检查数据类型一致性
        if 'embedding_model' in config and isinstance(config['embedding_model'], dict):
            list_fields = ['channels', 'kernel_sizes', 'dilations', 'groups']
            for field in list_fields:
                if field in config['embedding_model']:
                    if not isinstance(config['embedding_model'][field], list):
                        result["errors"].append(
                            f"embedding_model.{field} 应该是列表类型"
                        )
                        result["valid"] = False
        
        return result
    
    def compare_configs(self, other_config: Union[str, Path, Dict]) -> Dict[str, Any]:
        """
        比较当前配置与另一个配置的差异
        
        参数:
            other_config: 另一个配置（文件路径或配置字典）
        
        Returns:
            比较结果字典，包含:
            - added: 新增的键
            - removed: 删除的键
            - modified: 修改的键
            - unchanged: 未改变的键
        """
        # 加载当前配置
        config1 = self.load_config()
        
        # 加载另一个配置
        if isinstance(other_config, (str, Path)):
            other_parser = ConfigParser(other_config)
            config2 = other_parser.load_config()
        else:
            config2 = other_config
        
        result = {
            "added": [],
            "removed": [],
            "modified": {},
            "unchanged": []
        }
        
        def _compare_dicts(d1, d2, prefix=""):
            """递归比较字典"""
            all_keys = set(d1.keys()) | set(d2.keys())
            
            for key in all_keys:
                full_key = f"{prefix}.{key}" if prefix else key
                
                if key not in d1:
                    result["added"].append(full_key)
                elif key not in d2:
                    result["removed"].append(full_key)
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    _compare_dicts(d1[key], d2[key], full_key)
                elif d1[key] != d2[key]:
                    result["modified"][full_key] = {
                        "old": d1[key],
                        "new": d2[key]
                    }
                else:
                    result["unchanged"].append(full_key)
        
        _compare_dicts(config1, config2)
        
        return result
    
    def get_config_summary(self) -> str:
        """
        获取配置摘要
        
        Returns:
            配置摘要字符串
        """
        config = self.load_config()
        
        summary = f"配置文件: {self.config_path.name}\n"
        summary += "=" * 80 + "\n\n"
        
        # 基础训练参数
        basic_params = ['lr', 'batch_size', 'number_of_epochs', 'step_size', 'seed']
        summary += "📊 基础训练参数:\n"
        for param in basic_params:
            if param in config:
                summary += f"  {param}: {config[param]}\n"
        summary += "\n"
        
        # 模型结构参数
        model_sections = ['embedding_model', 'classifier', 'compute_cost', 'opt_class']
        summary += "🏗️  模型结构参数:\n"
        
        for section in model_sections:
            if section in config:
                value = config[section]
                if isinstance(value, dict):
                    summary += f"\n  {section}:\n"
                    formatted = format_nested_dict(value, indent=4)
                    summary += formatted
                else:
                    summary += f"  {section}: {value}\n"
        
        return summary
    
    def export_to_json(self, json_path: Optional[Union[str, Path]] = None) -> Path:
        """
        导出配置为 JSON 文件
        
        参数:
            json_path: JSON 文件路径，如果为 None 则使用配置文件同名但扩展名为 .json
        
        Returns:
            导出的 JSON 文件路径
        """
        if json_path is None:
            json_path = self.config_path.with_suffix('.json')
        else:
            json_path = Path(json_path)
        
        config = self.load_config()
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return json_path
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        提取模型相关的配置参数
        
        Returns:
            模型参数字典
        """
        config = self.load_config()
        
        model_params = {}
        
        # 提取 embedding_model 参数
        if 'embedding_model' in config:
            model_params['embedding_model'] = config['embedding_model']
        
        # 提取 classifier 参数
        if 'classifier' in config:
            model_params['classifier'] = config['classifier']
        
        # 提取损失函数参数
        if 'compute_cost' in config:
            model_params['loss'] = config['compute_cost']
        
        return model_params
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """
        提取训练相关的配置参数
        
        Returns:
            训练参数字典
        """
        config = self.load_config()
        
        training_params = {}
        
        # 基础训练参数
        training_fields = [
            'lr', 'batch_size', 'number_of_epochs', 'step_size', 'seed',
            'data_folder', 'output_folder', 'train_log'
        ]
        
        for field in training_fields:
            if field in config:
                training_params[field] = config[field]
        
        # 优化器参数
        if 'opt_class' in config:
            training_params['optimizer'] = config['opt_class']
        
        # 学习率调度器参数
        if 'lr_annealing' in config:
            training_params['lr_scheduler'] = config['lr_annealing']
        
        return training_params


# 便捷函数
def load_config(config_path: Optional[Union[str, Path]] = None,
               resolve_references: bool = True) -> Dict:
    """
    快速加载配置文件的便捷函数
    
    参数:
        config_path: 配置文件路径
        resolve_references: 是否解析 YAML 引用（!ref, !new:, !apply: 等）
    
    Returns:
        配置字典
    """
    parser = ConfigParser(config_path)
    return parser.load_config(resolve_references=resolve_references)


def validate_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    快速验证配置文件的便捷函数
    
    参数:
        config_path: 配置文件路径
    
    Returns:
        验证结果字典
    """
    parser = ConfigParser(config_path)
    return parser.validate_config()


def compare_configs(config1: Union[str, Path, Dict], 
                    config2: Union[str, Path, Dict]) -> Dict[str, Any]:
    """
    比较两个配置文件的便捷函数
    
    参数:
        config1: 第一个配置
        config2: 第二个配置
    
    Returns:
        比较结果字典
    """
    if isinstance(config1, dict) and isinstance(config2, dict):
        result = {
            "added": [],
            "removed": [],
            "modified": {},
            "unchanged": []
        }

        def _compare_dicts(d1, d2, prefix=""):
            all_keys = set(d1.keys()) | set(d2.keys())
            for key in all_keys:
                full_key = f"{prefix}.{key}" if prefix else key
                if key not in d1:
                    result["added"].append(full_key)
                elif key not in d2:
                    result["removed"].append(full_key)
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    _compare_dicts(d1[key], d2[key], full_key)
                elif d1[key] != d2[key]:
                    result["modified"][full_key] = {"old": d1[key], "new": d2[key]}
                else:
                    result["unchanged"].append(full_key)

        _compare_dicts(config1, config2)
        return result

    parser1 = ConfigParser(config1) if not isinstance(config1, dict) else None
    parser2 = ConfigParser(config2) if not isinstance(config2, dict) else None
    
    if parser1:
        return parser1.compare_configs(config2)
    else:
        # 交换结果中的 old 和 new
        result = parser2.compare_configs(config1)
        return {
            "added": result["removed"],
            "removed": result["added"],
            "modified": {k: {"old": v["new"], "new": v["old"]} 
                        for k, v in result["modified"].items()},
            "unchanged": result["unchanged"]
        }