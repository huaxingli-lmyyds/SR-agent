"""
路径操作工具函数
为整个工程提供统一的绝对路径管理和路径操作功能
"""

import os
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from datetime import datetime


# ============================================================================
# 项目基础路径
# ============================================================================

def get_project_root() -> Path:
    """
    获取项目根目录（SR-agent 目录）
    
    Returns:
        Path: 项目根目录的绝对路径
    """
    # 从当前文件向上两级到项目根目录
    # agent/utils/path_tool.py -> agent/ -> SR-agent/
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def get_agent_dir() -> Path:
    """
    获取 agent 目录
    
    Returns:
        Path: agent 目录的绝对路径
    """
    return get_project_root() / "agent"


def get_configs_dir() -> Path:
    """
    获取配置文件目录
    
    Returns:
        Path: configs 目录的绝对路径
    """
    return get_project_root() / "configs"

def get_datasets_dir() -> Path:
    """
    获取数据集目录
    
    Returns:
        Path: datasets 目录的绝对路径
    """
    return get_project_root() / "datasets"


def get_recipes_dir() -> Path:
    """
    获取训练脚本目录
    
    Returns:
        Path: recipes 目录的绝对路径
    """
    return get_project_root() / "recipes"


def get_experiments_dir() -> Path:
    """
    获取实验记录目录
    
    Returns:
        Path: agent/experiments 目录的绝对路径
    """
    return get_agent_dir() / "experiments"


def get_results_dir() -> Path:
    """
    获取结果目录
    
    Returns:
        Path: agent/results 目录的绝对路径
    """
    return get_agent_dir() / "results"


# ============================================================================
# 特定文件路径
# ============================================================================

def get_config_file(config_name: str = "train_ecapa_tdnn.yaml") -> Path:
    """
    获取配置文件路径
    
    Args:
        config_name: 配置文件名，默认为 train_ecapa_tdnn.yaml
    
    Returns:
        Path: 配置文件的绝对路径
    """
    return get_configs_dir() / config_name


def get_train_script(script_name: str = "voxceleb/train_speaker_embeddings.py") -> Path:
    """
    获取训练脚本路径
    
    Args:
        script_name: 训练脚本相对路径，默认为 voxceleb/train_speaker_embeddings.py
    
    Returns:
        Path: 训练脚本的绝对路径
    """
    return get_recipes_dir() / script_name


def get_eval_script(script_name: str = "voxceleb/speaker_verification_cosine.py") -> Path:
    """
    获取评估脚本路径
    
    Args:
        script_name: 评估脚本相对路径，默认为 voxceleb/speaker_verification_cosine.py
    
    Returns:
        Path: 评估脚本的绝对路径
    """
    return get_recipes_dir() / script_name


def get_system_prompt(prompt_name: str = "hpo_prompt.txt") -> Path:
    """
    获取系统提示词文件路径
    
    Args:
        prompt_name: 提示词文件名，默认为 hpo_prompt.txt
    
    Returns:
        Path: 提示词文件的绝对路径
    """
    return get_agent_dir() / "prompts" / prompt_name


def get_experiments_history_file() -> Path:
    """
    获取实验历史记录文件路径
    
    Returns:
        Path: experiments_history.json 的绝对路径
    """
    return get_experiments_dir() / "experiments_history.json"


def get_experiment_configs_dir() -> Path:
    """
    获取实验配置备份目录
    
    Returns:
        Path: experiments/configs 目录的绝对路径
    """
    return get_experiments_dir() / "configs"


# ============================================================================
# 路径验证和创建
# ============================================================================

def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    
    Returns:
        Path: 目录的绝对路径
    """
    dir_path = Path(directory).resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def path_exists(path: Union[str, Path]) -> bool:
    """
    检查路径是否存在（文件或目录）
    
    Args:
        path: 要检查的路径
    
    Returns:
        bool: 路径是否存在
    """
    return Path(path).exists()


def file_exists(file_path: Union[str, Path]) -> bool:
    """
    检查文件是否存在
    
    Args:
        file_path: 文件路径
    
    Returns:
        bool: 文件是否存在
    """
    return Path(file_path).is_file()


def dir_exists(dir_path: Union[str, Path]) -> bool:
    """
    检查目录是否存在
    
    Args:
        dir_path: 目录路径
    
    Returns:
        bool: 目录是否存在
    """
    return Path(dir_path).is_dir()


def is_absolute_path(path: Union[str, Path]) -> bool:
    """
    检查是否为绝对路径
    
    Args:
        path: 要检查的路径
    
    Returns:
        bool: 是否为绝对路径
    """
    return Path(path).is_absolute()


def to_absolute_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    将路径转换为绝对路径
    
    Args:
        path: 要转换的路径
        base_dir: 基础目录，如果为 None 则使用项目根目录
    
    Returns:
        Path: 绝对路径
    """
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    
    base = Path(base_dir) if base_dir else get_project_root()
    return (base / path).resolve()


def to_relative_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    将路径转换为相对路径
    
    Args:
        path: 要转换的路径
        base_dir: 基础目录，如果为 None 则使用项目根目录
    
    Returns:
        Path: 相对路径
    """
    path = Path(path).resolve()
    base = Path(base_dir).resolve() if base_dir else get_project_root()
    
    try:
        return path.relative_to(base)
    except ValueError:
        # 如果无法转换为相对路径，返回绝对路径
        return path


# ============================================================================
# 文件和目录操作
# ============================================================================

def list_files(directory: Union[str, Path], 
               pattern: Optional[str] = None,
               recursive: bool = False) -> List[Path]:
    """
    列出目录中的文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式（如 *.yaml），如果为 None 则匹配所有文件
        recursive: 是否递归查找子目录
    
    Returns:
        List[Path]: 文件路径列表
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    
    if recursive:
        files = dir_path.rglob(pattern if pattern else "*")
    else:
        files = dir_path.glob(pattern if pattern else "*")
    
    # 只返回文件，不包括目录
    return [f for f in files if f.is_file()]


def list_directories(directory: Union[str, Path], 
                     recursive: bool = False) -> List[Path]:
    """
    列出目录中的子目录
    
    Args:
        directory: 目录路径
        recursive: 是否递归查找子目录
    
    Returns:
        List[Path]: 目录路径列表
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    
    if recursive:
        dirs = dir_path.rglob("*")
    else:
        dirs = dir_path.glob("*")
    
    # 只返回目录，不包括文件
    return [d for d in dirs if d.is_dir()]


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    获取文件大小（字节）
    
    Args:
        file_path: 文件路径
    
    Returns:
        int: 文件大小（字节），如果文件不存在返回 -1
    """
    path = Path(file_path)
    if path.is_file():
        return path.stat().st_size
    return -1


def get_file_mtime(file_path: Union[str, Path]) -> datetime:
    """
    获取文件最后修改时间
    
    Args:
        file_path: 文件路径
    
    Returns:
        datetime: 最后修改时间，如果文件不存在返回 None
    """
    path = Path(file_path)
    if path.is_file():
        timestamp = path.stat().st_mtime
        return datetime.fromtimestamp(timestamp)
    return None


def backup_file(file_path: Union[str, Path],
                backup_dir: Optional[Union[str, Path]] = None,
                suffix: Optional[str] = None) -> Path:
    """
    备份文件
    
    Args:
        file_path: 要备份的文件路径
        backup_dir: 备份目录，如果为 None 则使用原文件所在目录
        suffix: 备份文件后缀，如果为 None 则使用时间戳
    
    Returns:
        Path: 备份文件的路径
    """
    src_path = Path(file_path).resolve()
    
    if not src_path.is_file():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 确定备份目录
    if backup_dir:
        backup_dir = Path(backup_dir).resolve()
        ensure_dir(backup_dir)
    else:
        backup_dir = src_path.parent
    
    # 生成备份文件名
    if suffix:
        backup_name = f"{src_path.stem}.{suffix}{src_path.suffix}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{src_path.stem}.backup_{timestamp}{src_path.suffix}"
    
    backup_path = backup_dir / backup_name
    
    # 复制文件
    import shutil
    shutil.copy2(src_path, backup_path)
    
    return backup_path


# ============================================================================
# 路径信息获取
# ============================================================================

def get_filename(file_path: Union[str, Path]) -> str:
    """
    获取文件名（带扩展名）
    
    Args:
        file_path: 文件路径
    
    Returns:
        str: 文件名
    """
    return Path(file_path).name


def get_filename_without_ext(file_path: Union[str, Path]) -> str:
    """
    获取文件名（不带扩展名）
    
    Args:
        file_path: 文件路径
    
    Returns:
        str: 文件名（不带扩展名）
    """
    return Path(file_path).stem


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    获取文件扩展名
    
    Args:
        file_path: 文件路径
    
    Returns:
        str: 文件扩展名（包含点号，如 .yaml）
    """
    return Path(file_path).suffix


def get_parent_dir(file_path: Union[str, Path]) -> Path:
    """
    获取父目录
    
    Args:
        file_path: 文件或目录路径
    
    Returns:
        Path: 父目录路径
    """
    return Path(file_path).parent.resolve()


# ============================================================================
# 日志和结果路径
# ============================================================================

def get_experiment_log_path(experiment_id: str) -> Path:
    """
    获取实验日志文件路径
    
    Args:
        experiment_id: 实验 ID
    
    Returns:
        Path: 日志文件路径（位于实验目录下）
    """
    return get_experiments_dir() / experiment_id / "experiment.log"


def get_model_save_path(experiment_id: str, model_name: str = "model.ckpt") -> Path:
    """
    获取模型保存路径
    
    Args:
        experiment_id: 实验 ID
        model_name: 模型文件名
    
    Returns:
        Path: 模型文件路径
    """
    exp_dir = ensure_dir(get_experiments_dir() / experiment_id)
    return exp_dir / model_name


def get_training_result_path(experiment_id: str, result_name: str = "experiment_record.json") -> Path:
    """
    获取训练结果文件路径
    
    Args:
        experiment_id: 实验 ID
        result_name: 结果文件名
    
    Returns:
        Path: 结果文件路径
    """
    exp_dir = ensure_dir(get_experiments_dir() / experiment_id)
    return exp_dir / result_name


# ============================================================================
# 实用工具
# ============================================================================

def join_paths(*paths: Union[str, Path]) -> Path:
    """
    连接多个路径
    
    Args:
        *paths: 要连接的路径
    
    Returns:
        Path: 连接后的路径
    """
    result = Path(paths[0])
    for path in paths[1:]:
        result = result / Path(path)
    return result


def normalize_path(path: Union[str, Path]) -> Path:
    """
    规范化路径（解析 . 和 .. 等）
    
    Args:
        path: 要规范化的路径
    
    Returns:
        Path: 规范化后的路径
    """
    return Path(path).resolve()


def get_path_info(path: Union[str, Path]) -> dict:
    """
    获取路径的详细信息
    
    Args:
        path: 文件或目录路径
    
    Returns:
        dict: 包含路径信息的字典
    """
    path = Path(path).resolve()
    exists = path.exists()
    stat = path.stat() if exists else None
    
    info = {
        "path": str(path),
        "name": path.name,
        "exists": exists,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "is_absolute": path.is_absolute(),
        "size": stat.st_size if stat and path.is_file() else 0,
        "mtime": datetime.fromtimestamp(stat.st_mtime) if stat else None,
        "extension": path.suffix if path.is_file() else None,
    }
    
    return info


def format_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 文件大小（字节）
    
    Returns:
        str: 格式化后的大小字符串（如 1.5 MB）
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


# ============================================================================
# 查找文件和目录
# ============================================================================

def find_result_dirs(base_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    """
    查找所有结果目录
    
    Args:
        base_dir: 基础目录，如果为 None 则使用 results 目录
    
    Returns:
        List[Path]: 结果目录列表
    """
    if base_dir is None:
        base_dir = get_results_dir()
    return list_directories(base_dir)


def find_log_files(base_dir: Optional[Union[str, Path]] = None,
                    pattern: str = "*.log") -> List[Path]:
    """
    查找所有日志文件
    
    Args:
        base_dir: 基础目录，如果为 None 则使用 experiments 目录
        pattern: 文件匹配模式
    
    Returns:
        List[Path]: 日志文件路径列表
    """
    if base_dir is None:
        base_dir = get_experiments_dir()
    return list_files(base_dir, pattern=pattern, recursive=True)


def find_scores_file(directory: Union[str, Path]) -> Optional[Path]:
    """
    在指定目录中查找 scores.txt 文件
    
    Args:
        directory: 要搜索的目录
    
    Returns:
        Optional[Path]: scores.txt 文件路径，如果不存在返回 None
    """
    dir_path = Path(directory)
    scores_file = dir_path / "scores.txt"
    return scores_file if scores_file.is_file() else None


def copy_file(src_path: Union[str, Path],
              dst_path: Union[str, Path]) -> Path:
    """
    复制文件
    
    Args:
        src_path: 源文件路径
        dst_path: 目标文件或目录路径
    
    Returns:
        Path: 目标文件路径
    """
    src = Path(src_path).resolve()
    dst = Path(dst_path).resolve()
    
    if not src.is_file():
        raise FileNotFoundError(f"源文件不存在: {src}")
    
    import shutil
    if dst.is_dir():
        dst = dst / src.name
    
    shutil.copy2(src, dst)
    return dst


def yaml_to_dict(yaml_obj: Any) -> Union[Dict, List, Any]:
    """
    将 ruamel.yaml 的特殊对象转换为标准 Python 类型
    
    Args:
        yaml_obj: ruamel.yaml 加载的对象
    
    Returns:
        转换后的标准 Python 对象
    """
    # 处理 TaggedScalar 类型
    try:
        from ruamel.yaml.scalarstring import TaggedScalar
        from ruamel.yaml.comments import TaggedScalarCommented
        if isinstance(yaml_obj, (TaggedScalar, TaggedScalarCommented)):
            # 返回实际值
            return yaml_to_dict(yaml_obj.value if hasattr(yaml_obj, 'value') else str(yaml_obj))
    except ImportError:
        pass
    
    # 处理注释对象
    try:
        from ruamel.yaml.comments import CommentedMap, CommentedSeq
        if isinstance(yaml_obj, CommentedMap):
            return {key: yaml_to_dict(value) for key, value in yaml_obj.items()}
        elif isinstance(yaml_obj, CommentedSeq):
            return [yaml_to_dict(item) for item in yaml_obj]
    except ImportError:
        pass
    
    # 处理标准 Python 类型
    if isinstance(yaml_obj, dict):
        return {key: yaml_to_dict(value) for key, value in yaml_obj.items()}
    elif isinstance(yaml_obj, list):
        return [yaml_to_dict(item) for item in yaml_obj]
    elif isinstance(yaml_obj, (str, int, float, bool)) or yaml_obj is None:
        return yaml_obj
    else:
        # 其他类型尝试转换为字符串
        return str(yaml_obj)


def format_nested_dict(d: dict, indent: int = 0) -> str:
    """
    格式化嵌套字典为字符串
    
    Args:
        d: 要格式化的字典
        indent: 缩进级别
    
    Returns:
        格式化后的字符串
    """
    result = []
    indent_str = " " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            result.append(f"{indent_str}{key}:")
            result.append(format_nested_dict(value, indent + 2))
        elif isinstance(value, (list, tuple)):
            result.append(f"{indent_str}{key}: {list(value)}")
        else:
            result.append(f"{indent_str}{key}: {value}")
    return "\n".join(result)


# ============================================================================
# 调试和日志
# ============================================================================

def print_path_info(path: Union[str, Path]) -> None:
    """
    打印路径的详细信息（用于调试）
    
    Args:
        path: 文件或目录路径
    """
    info = get_path_info(path)
    
    print(f"路径: {info['path']}")
    print(f"  名称: {info['name']}")
    print(f"  存在: {'是' if info['exists'] else '否'}")
    if info['exists']:
        print(f"  类型: {'文件' if info['is_file'] else '目录'}")
        print(f"  绝对路径: {'是' if info['is_absolute'] else '否'}")
        if info['is_file']:
            print(f"  大小: {format_size(info['size'])}")
            print(f"  扩展名: {info['extension']}")
        if info['mtime']:
            print(f"  修改时间: {info['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# 便捷函数
# ============================================================================

def get_all_config_files() -> List[Path]:
    """
    获取所有配置文件
    
    Returns:
        List[Path]: 配置文件路径列表
    """
    return list_files(get_configs_dir(), pattern="*.yaml")


def get_all_experiment_dirs() -> List[Path]:
    """
    获取所有实验目录
    
    Returns:
        List[Path]: 实验目录路径列表
    """
    return list_directories(get_experiments_dir())


def cleanup_old_backups(directory: Union[str, Path], 
                        pattern: str = "*.backup_*",
                        keep_last_n: int = 5) -> int:
    """
    清理旧的备份文件，保留最近的 n 个
    
    Args:
        directory: 要清理的目录
        pattern: 备份文件匹配模式
        keep_last_n: 保留最近的备份数量
    
    Returns:
        int: 删除的文件数量
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return 0
    
    # 获取所有匹配的备份文件
    backups = list_files(dir_path, pattern=pattern)
    
    # 按修改时间排序（从旧到新）
    backups.sort(key=lambda f: get_file_mtime(f) or datetime.min)
    
    # 删除旧的备份（保留最近的 n 个）
    to_delete = backups[:-keep_last_n] if len(backups) > keep_last_n else []
    
    deleted_count = 0
    for backup in to_delete:
        try:
            backup.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"删除备份文件失败: {backup}, 错误: {e}")
    
    return deleted_count


# ============================================================================
# 导出的主要路径常量（兼容性）
# ============================================================================

# 主配置文件路径
CONFIG_PATH = str(get_config_file("train_ecapa_tdnn.yaml"))

# 训练脚本路径
TRAIN_SCRIPT = str(get_train_script())

# 评估脚本路径
EVAL_SCRIPT = str(get_eval_script())

# 系统提示词路径
SYSTEM_PROMPT_PATH = str(get_system_prompt())

# 实验记录文件路径
EXPERIMENTS_FILE = str(get_experiments_history_file())


if __name__ == "__main__":
    # 测试代码
    print("=" * 80)
    print("路径工具测试")
    print("=" * 80)
    
    print(f"\n项目根目录: {get_project_root()}")
    print(f"Agent 目录: {get_agent_dir()}")
    print(f"配置目录: {get_configs_dir()}")
    print(f"训练脚本: {get_train_script()}")
    
    print(f"\n配置文件存在: {file_exists(get_config_file())}")
    print(f"训练脚本存在: {file_exists(get_train_script())}")
    
    print(f"\n所有配置文件:")
    for config in get_all_config_files():
        print(f"  - {config.name}")
    
    print(f"\n路径信息示例:")
    print_path_info(get_config_file())