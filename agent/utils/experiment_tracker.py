"""
实验跟踪模块
提供实验记录管理、状态跟踪、结果比较等功能
"""

from pathlib import Path
from typing import Union, Dict, List, Optional, Any
from datetime import datetime
import json
import shutil

# 导入路径工具
from .path_tool import (
    get_experiments_dir,
    ensure_dir,
    get_experiment_log_path,
    get_experiment_configs_dir,
    list_directories
)


class ExperimentTracker:
    """实验跟踪器类"""
    
    def __init__(self, experiments_dir: Optional[Union[str, Path]] = None):
        """
        初始化实验跟踪器
        
        参数:
            experiments_dir: 实验目录，如果为 None 则使用默认目录
        """
        if experiments_dir:
            self.experiments_dir = Path(experiments_dir)
        else:
            self.experiments_dir = get_experiments_dir()
        
        # 确保目录存在
        ensure_dir(self.experiments_dir)
        
        # 历史记录文件
        self.history_file = self.experiments_dir / "experiments_history.json"
        
        # 计数器，用于在同一秒内创建多个实验时保证 ID 唯一
        self._counter = 0
    
    def create_experiment(self, config: Dict, config_path: str, 
                         data_folder: str, description: str = "") -> str:
        """
        创建新实验
        
        参数:
            config: 配置字典
            config_path: 配置文件路径
            data_folder: 数据文件夹路径
            description: 实验描述
        
        Returns:
            实验 ID
        """
        # 生成实验 ID（使用计数器避免同一秒内的冲突）
        base_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{base_id}_{self._counter}"
        self._counter += 1
        
        # 创建实验目录
        exp_dir = ensure_dir(self.experiments_dir / experiment_id)
        
        # 备份配置文件
        config_backup = exp_dir / "config.yaml"
        shutil.copy2(config_path, config_backup)
        
        # 使用 ConfigParser 加载配置以解析占位符
        from . import ConfigParser
        parser = ConfigParser(config_path)
        
        # 临时设置 data_folder 以确保占位符被正确解析
        parser.data_folder = data_folder
        
        config_resolved = parser.load_config()
        
        # 确保 data_folder 使用传入的参数
        config_resolved['data_folder'] = data_folder
        
        # 转换配置为可序列化的格式
        from .path_tool import yaml_to_dict
        config_serializable = yaml_to_dict(config_resolved)
        
        # 创建实验记录
        experiment_record = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "status": "created",
            "duration_seconds": 0,
            "config": config_serializable,
            "training": {
                "config_path": config_path,
                "config_backup_path": str(config_backup),
                "data_folder": data_folder,
                "log_path": None,
                "output_folder": config_serializable.get("output_folder"),
                "metrics": None,
                "model_paths": []
            },
            "evaluation": None,
            "results": None,
            "error": None
        }
        
        # 保存实验记录
        record_path = exp_dir / "experiment_record.json"
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_record, f, indent=2, ensure_ascii=False)
        
        # 更新历史记录
        self._update_history(experiment_record)
        
        return experiment_id
    
    def update_experiment(self, experiment_id: str, 
                         results: Optional[Dict] = None,
                         status: Optional[str] = None,
                         error: Optional[str] = None,
                         duration: Optional[float] = None,
                         training: Optional[Dict] = None,
                         evaluation: Optional[Dict] = None) -> bool:
        """
        更新实验信息
        
        参数:
            experiment_id: 实验 ID
            results: 实验结果
            status: 实验状态
            error: 错误信息
            duration: 训练时长（秒）
        
        Returns:
            是否更新成功
        """
        # 读取实验记录
        record = self.get_experiment(experiment_id)
        if record is None:
            return False
        
        # 更新字段
        if results is not None:
            record["results"] = results
        if status is not None:
            record["status"] = status
        if error is not None:
            record["error"] = error
        if duration is not None:
            record["duration_seconds"] = duration
        if training is not None:
            current_training = record.get("training") or {}
            current_training.update(training)
            record["training"] = current_training
        if evaluation is not None:
            record["evaluation"] = evaluation
        
        # 保存更新后的记录
        record_path = self.experiments_dir / experiment_id / "experiment_record.json"
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        
        # 更新历史记录
        self._update_history(record)
        
        return True
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """
        获取实验记录
        
        参数:
            experiment_id: 实验 ID
        
        Returns:
            实验记录字典，如果不存在则返回 None
        """
        exp_dir = self.experiments_dir / experiment_id
        if not exp_dir.exists():
            return None
        
        record_path = exp_dir / "experiment_record.json"
        if not record_path.exists():
            return None
        
        with open(record_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_experiments(self, 
                       status: Optional[str] = None,
                       limit: Optional[int] = None,
                       sort_by: str = "timestamp",
                       reverse: bool = True) -> List[Dict]:
        """
        列出实验
        
        参数:
            status: 筛选状态（如 "success", "failed"）
            limit: 限制返回数量
            sort_by: 排序字段（如 "timestamp", "duration_seconds"）
            reverse: 是否倒序排列
        
        Returns:
            实验记录列表
        """
        # 读取历史记录
        history = self._load_history()
        
        # 筛选状态
        if status:
            history = [exp for exp in history if exp.get('status') == status]
        
        # 排序
        if sort_by:
            try:
                history.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
            except Exception:
                pass
        
        # 限制数量
        if limit:
            history = history[:limit]
        
        return history
    
    def find_best_experiment(self, 
                           metric: str = "eer",
                           minimize: bool = True,
                           top_n: int = 1) -> List[Dict]:
        """
        查找最佳实验
        
        参数:
            metric: 评估指标（如 "eer", "accuracy", "loss"）
            minimize: 是否最小化指标（True 表示越小越好）
            top_n: 返回前 N 个最佳实验
        
        Returns:
            最佳实验记录列表
        """
        # 获取所有成功的实验
        successful_exps = self.list_experiments(status="success")
        
        # 过滤出有该指标的实验
        valid_exps = []
        for exp in successful_exps:
            results = exp.get('results', {})
            if results and metric in results and results[metric] is not None:
                valid_exps.append(exp)
        
        if not valid_exps:
            return []
        
        # 排序
        valid_exps.sort(
            key=lambda x: x['results'][metric],
            reverse=not minimize
        )
        
        return valid_exps[:top_n]
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        比较多个实验
        
        参数:
            experiment_ids: 实验 ID 列表
        
        Returns:
            比较结果字典
        """
        experiments = []
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                experiments.append(exp)
        
        if not experiments:
            return {"error": "没有找到有效的实验"}
        
        # 提取所有可能的指标
        all_metrics = set()
        for exp in experiments:
            if exp.get('results'):
                all_metrics.update(exp['results'].keys())
        
        # 构建比较结果
        comparison = {
            "experiments": {},
            "metrics_comparison": {},
            "best_by_metric": {}
        }
        
        # 添加实验基本信息
        for exp in experiments:
            exp_id = exp['experiment_id']
            comparison["experiments"][exp_id] = {
                "status": exp.get('status'),
                "timestamp": exp.get('timestamp'),
                "duration_seconds": exp.get('duration_seconds'),
                "results": exp.get('results', {})
            }
        
        # 比较每个指标
        for metric in sorted(all_metrics):
            values = []
            for exp in experiments:
                if exp.get('results') and metric in exp['results']:
                    value = exp['results'][metric]
                    if value is not None:
                        values.append((exp['experiment_id'], value))
            
            if values:
                # 判断指标类型
                if all(isinstance(v[1], (int, float)) for v in values):
                    # 数值型指标，找出最佳值
                    sorted_values = sorted(values, key=lambda x: x[1])
                    comparison["metrics_comparison"][metric] = {
                        "type": "numeric",
                        "values": {exp_id: val for exp_id, val in values},
                        "best": sorted_values[0][0],
                        "best_value": sorted_values[0][1]
                    }
                else:
                    # 非数值型指标
                    comparison["metrics_comparison"][metric] = {
                        "type": "other",
                        "values": {exp_id: val for exp_id, val in values}
                    }
        
        # 为每个指标找出最佳实验
        for metric, comp in comparison["metrics_comparison"].items():
            if comp["type"] == "numeric":
                comparison["best_by_metric"][metric] = comp["best"]
        
        return comparison
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取实验统计信息
        
        Returns:
            统计信息字典
        """
        history = self._load_history()
        
        if not history:
            return {"total": 0, "message": "暂无实验记录"}
        
        # 统计各状态数量
        status_counts = {}
        for exp in history:
            status = exp.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # 统计训练时长
        durations = [exp.get('duration_seconds', 0) for exp in history if exp.get('duration_seconds')]
        avg_duration = sum(durations) / len(durations) if durations else 0
        total_duration = sum(durations)
        
        # 统计成功率
        success_count = status_counts.get('success', 0)
        success_rate = success_count / len(history) if history else 0
        
        # 时间范围
        timestamps = [datetime.fromisoformat(exp['timestamp']) for exp in history if exp.get('timestamp')]
        if timestamps:
            time_range = {
                "first": min(timestamps).isoformat(),
                "last": max(timestamps).isoformat()
            }
        else:
            time_range = None
        
        return {
            "total": len(history),
            "status_counts": status_counts,
            "success_rate": f"{success_rate * 100:.1f}%",
            "average_duration_seconds": avg_duration,
            "total_duration_hours": total_duration / 3600,
            "time_range": time_range
        }
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        删除实验记录
        
        参数:
            experiment_id: 实验 ID
        
        Returns:
            是否删除成功
        """
        exp_dir = self.experiments_dir / experiment_id
        if not exp_dir.exists():
            return False
        
        # 删除目录
        shutil.rmtree(exp_dir)
        
        # 更新历史记录
        self._remove_from_history(experiment_id)
        
        return True
    
    def cleanup_old_experiments(self, keep_n: int = 10, 
                               status_filter: Optional[str] = None) -> int:
        """
        清理旧的实验
        
        参数:
            keep_n: 保留最近 N 个实验
            status_filter: 只清理特定状态的实验
        
        Returns:
            删除的实验数量
        """
        # 获取实验列表
        exps = self.list_experiments(status=status_filter, sort_by="timestamp", reverse=True)
        
        # 确定要删除的实验
        to_delete = exps[keep_n:] if len(exps) > keep_n else []
        
        deleted_count = 0
        for exp in to_delete:
            if self.delete_experiment(exp['experiment_id']):
                deleted_count += 1
        
        return deleted_count
    
    def export_experiments(self, output_path: Union[str, Path], 
                         experiment_ids: Optional[List[str]] = None) -> Path:
        """
        导出实验数据
        
        参数:
            output_path: 输出文件路径
            experiment_ids: 要导出的实验 ID 列表，如果为 None 则导出所有实验
        
        Returns:
            导出文件路径
        """
        if experiment_ids:
            experiments = [self.get_experiment(exp_id) for exp_id in experiment_ids]
            experiments = [exp for exp in experiments if exp is not None]
        else:
            experiments = self._load_history()
        
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(experiments, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _load_history(self) -> List[Dict]:
        """加载历史记录"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except (json.JSONDecodeError, Exception):
            # 文件损坏或格式错误，返回空列表
            return []
    
    def _update_history(self, experiment: Dict):
        """更新历史记录"""
        history = self._load_history()
        
        # 查找并更新或添加
        found = False
        for i, exp in enumerate(history):
            if exp['experiment_id'] == experiment['experiment_id']:
                history[i] = experiment
                found = True
                break
        
        if not found:
            history.append(experiment)
        
        # 保存
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def _remove_from_history(self, experiment_id: str):
        """从历史记录中移除实验"""
        history = self._load_history()
        history = [exp for exp in history if exp['experiment_id'] != experiment_id]
        
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


# 便捷函数
def create_experiment(config: Dict, config_path: str, 
                    data_folder: str, description: str = "") -> str:
    """快速创建实验的便捷函数"""
    tracker = ExperimentTracker()
    return tracker.create_experiment(config, config_path, data_folder, description)


def list_experiments(status: Optional[str] = None, 
                    limit: Optional[int] = None) -> List[Dict]:
    """快速列出实验的便捷函数"""
    tracker = ExperimentTracker()
    return tracker.list_experiments(status=status, limit=limit)


def find_best_experiment(metric: str = "eer", 
                        minimize: bool = True) -> Optional[Dict]:
    """快速查找最佳实验的便捷函数"""
    tracker = ExperimentTracker()
    best = tracker.find_best_experiment(metric=metric, minimize=minimize, top_n=1)
    return best[0] if best else None


def get_experiment_stats() -> Dict[str, Any]:
    """快速获取统计信息的便捷函数"""
    tracker = ExperimentTracker()
    return tracker.get_statistics()