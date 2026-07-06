"""
实验跟踪模块
提供实验记录管理、状态跟踪、结果比较等功能
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union, Dict, List, Optional, Any
from datetime import datetime
import json
import shutil

# 导入路径工具
from .path_tool import (
    get_hpo_experiments_dir,
    get_experiment_type_dir,
    ensure_dir,
    resolve_config_path,
    resolve_config_value_path,
)
from .experiment_versioning import sync_experiment_catalog


@dataclass
class BaseExperimentRecord:
    """所有智能体实验记录的公共基类。"""

    experiment_type: str
    schema_version: str = "2.0"
    stage: str = ""
    experiment_id: str = ""
    timestamp: str = ""
    updated_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    description: str = ""
    status: str = "created"
    duration_seconds: float = 0.0
    config_path: str = ""
    actor: Dict[str, Any] = field(default_factory=dict)
    task: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HPOExperimentRecord(BaseExperimentRecord):
    """超参数智能体实验记录。"""

    experiment_type: str = "hpo"


@dataclass
class DataProcessingExperimentRecord(BaseExperimentRecord):
    """数据处理智能体实验记录。"""

    experiment_type: str = "data_processing"


@dataclass
class OrchestrationExperimentRecord(BaseExperimentRecord):
    """统筹智能体实验记录。"""

    experiment_type: str = "orchestration"
    linked_experiments: Dict[str, Any] = field(default_factory=dict)
    agent_messages: List[Dict[str, Any]] = field(default_factory=list)


ExperimentRecordDict = Dict[str, Any]


class ExperimentTracker:
    """实验跟踪器类"""
    
    def __init__(self, experiments_dir: Optional[Union[str, Path]] = None):
        """
        初始化实验跟踪器
        
        参数:
            experiments_dir: 实验目录，如果为 None 则使用默认目录
        """
        if experiments_dir is not None:
            self.experiments_dir = Path(experiments_dir).resolve()
            self._custom_experiments_dir = True
        else:
            self.experiments_dir = get_hpo_experiments_dir()
            self._custom_experiments_dir = False
        
        # 确保目录存在
        ensure_dir(self.experiments_dir)
        
        # 历史记录文件
        self.history_file = self.experiments_dir / "experiments_history.json"
        
        # 计数器，用于在同一秒内创建多个实验时保证 ID 唯一
        self._counter = 0
    
    def create_experiment(self, config_path: str,
                         data_folder: str,
                         output_folder: Optional[str] = None,
                         description: str = "",
                         experiment_type: str = "hpo",
                         extra_fields: Optional[Dict[str, Any]] = None,
                         stage: Optional[str] = None,
                         actor: Optional[Dict[str, Any]] = None,
                         task: Optional[Dict[str, Any]] = None,
                         model: Optional[Dict[str, Any]] = None,
                         execution: Optional[Dict[str, Any]] = None) -> str:
        """
        创建新实验
        
        参数:
            config_path: 配置文件路径
            data_folder: 数据文件夹路径
            output_folder: 输出文件夹路径
            description: 实验描述
            experiment_type: 实验类型，默认 hpo
            extra_fields: 额外要写入记录的字段
        
        Returns:
            实验 ID
        """
        config_path = str(resolve_config_path(config_path))
        resolved_data_folder = resolve_config_value_path(data_folder)
        resolved_output_folder = resolve_config_value_path(output_folder)
        data_folder = str(resolved_data_folder) if resolved_data_folder is not None else ""
        output_folder = str(resolved_output_folder) if resolved_output_folder is not None else None

        # 生成实验 ID（使用计数器避免同一秒内的冲突）
        experiments_dir = self._record_dir(experiment_type)
        base_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        while True:
            experiment_id = f"{base_id}_{self._counter}"
            self._counter += 1
            if not (experiments_dir / experiment_id).exists():
                break

        # 创建实验目录
        exp_dir = ensure_dir(experiments_dir / experiment_id)
        
        # 备份配置文件
        config_backup = exp_dir / "config.yaml"
        shutil.copy2(config_path, config_backup)
        
        # 创建实验记录
        experiment_record = self._build_record(
            experiment_type=experiment_type,
            experiment_id=experiment_id,
            config_path=config_path,
            data_folder=data_folder,
            output_folder=output_folder,
            description=description,
            config_backup_path=str(config_backup),
            extra_fields=extra_fields,
            stage=stage,
            actor=actor,
            task=task,
            model=model,
            execution=execution,
        )
        
        # 保存实验记录
        record_path = exp_dir / "experiment_record.json"
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_record, f, indent=2, ensure_ascii=False)
        sync_experiment_catalog(experiments_dir, exp_dir, experiment_record)
        
        # 更新历史记录
        self._update_history(experiment_record)
        
        return experiment_id

    def create_hpo_experiment(self, config_path: str, data_folder: str,
                              output_folder: Optional[str] = None,
                              description: str = "",
                              extra_fields: Optional[Dict[str, Any]] = None,
                              **metadata) -> str:
        """创建超参数智能体实验记录。"""
        return self.create_experiment(
            config_path=config_path,
            data_folder=data_folder,
            output_folder=output_folder,
            description=description,
            experiment_type="hpo",
            extra_fields=extra_fields,
            **metadata,
        )

    def create_data_processing_experiment(self, config_path: str, data_folder: str,
                                          output_folder: Optional[str] = None,
                                          description: str = "",
                                          extra_fields: Optional[Dict[str, Any]] = None,
                                          **metadata) -> str:
        """创建数据处理智能体实验记录。"""
        return self.create_experiment(
            config_path=config_path,
            data_folder=data_folder,
            output_folder=output_folder,
            description=description,
            experiment_type="data_processing",
            extra_fields=extra_fields,
            **metadata,
        )

    def create_orchestration_experiment(self, config_path: str, data_folder: str,
                                        output_folder: Optional[str] = None,
                                        description: str = "",
                                        extra_fields: Optional[Dict[str, Any]] = None,
                                        **metadata) -> str:
        """创建统筹智能体实验记录。"""
        return self.create_experiment(
            config_path=config_path,
            data_folder=data_folder,
            output_folder=output_folder,
            description=description,
            experiment_type="orchestration",
            extra_fields=extra_fields,
            **metadata,
        )
    
    def update_experiment(self, experiment_id: str, 
                         status: Optional[str] = None,
                         error: Optional[str] = None,
                         duration: Optional[float] = None,
                         linked_experiments: Optional[Dict] = None,
                         agent_messages: Optional[List[Dict[str, Any]]] = None,
                         stage: Optional[str] = None,
                         actor: Optional[Dict[str, Any]] = None,
                         task: Optional[Dict[str, Any]] = None,
                         model: Optional[Dict[str, Any]] = None,
                         execution: Optional[Dict[str, Any]] = None,
                         metrics: Optional[Dict[str, Dict[str, Any]]] = None,
                         artifacts: Optional[List[Dict[str, Any]]] = None,
                         parameters: Optional[Dict[str, Any]] = None,
                         extensions: Optional[Dict[str, Any]] = None,
                         experiment_type: Optional[str] = None) -> bool:
        """
        更新实验信息
        
        参数:
            experiment_id: 实验 ID
            metrics: 按数据切分保存的通用指标
            status: 实验状态
            error: 错误信息
            duration: 训练时长（秒）
        
        Returns:
            是否更新成功
        """
        # 读取实验记录
        record, record_dir = self._load_record(experiment_id, experiment_type=experiment_type)
        if record is None:
            return False
        
        # 更新字段
        now = datetime.now().isoformat()
        record["updated_at"] = now
        if status is not None:
            record["status"] = status
            if status == "running" and not record.get("started_at"):
                record["started_at"] = now
            if status in {"success", "failed", "cancelled"}:
                record["completed_at"] = now
        if error is not None:
            record["error"] = error
        elif status == "success":
            record["error"] = None
        if duration is not None:
            record["duration_seconds"] = duration
        extensions = dict(extensions or {})
        if stage is not None:
            record["stage"] = stage
        for key, value in (
            ("actor", actor),
            ("task", task),
            ("model", model),
            ("execution", execution),
            ("parameters", parameters),
        ):
            if value:
                current = dict(record.get(key) or {})
                current.update(value)
                record[key] = current
        if extensions:
            record["extensions"] = self._deep_merge(
                dict(record.get("extensions") or {}),
                extensions,
            )
        if metrics:
            current_metrics = dict(record.get("metrics") or {})
            for split, values in metrics.items():
                current_split = dict(current_metrics.get(split) or {})
                current_split.update(values)
                current_metrics[split] = current_split
            record["metrics"] = current_metrics
        if artifacts:
            existing = list(record.get("artifacts") or [])
            by_key = {(item.get("type"), item.get("name"), item.get("path")): item for item in existing}
            for item in artifacts:
                by_key[(item.get("type"), item.get("name"), item.get("path"))] = item
            record["artifacts"] = list(by_key.values())
        if linked_experiments is not None:
            record["linked_experiments"] = linked_experiments
        if agent_messages is not None:
            record["agent_messages"] = agent_messages
        
        # 保存更新后的记录
        record_path = record_dir / experiment_id / "experiment_record.json"
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        sync_experiment_catalog(record_dir, record_dir / experiment_id, record)
        
        # 更新历史记录
        self._update_history(record)
        
        return True

    @classmethod
    def _deep_merge(cls, current: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(current.get(key), dict):
                current[key] = cls._deep_merge(dict(current[key]), value)
            else:
                current[key] = value
        return current

    def update_hpo_experiment(self, experiment_id: str, **kwargs) -> bool:
        """更新超参数智能体实验记录。"""
        return self.update_experiment(experiment_id, experiment_type="hpo", **kwargs)

    def update_data_processing_experiment(self, experiment_id: str, **kwargs) -> bool:
        """更新数据处理智能体实验记录。"""
        return self.update_experiment(
            experiment_id,
            experiment_type="data_processing",
            **kwargs,
        )

    def update_orchestration_experiment(self, experiment_id: str,
                                        linked_experiments: Optional[Dict] = None,
                                        agent_messages: Optional[List[Dict[str, Any]]] = None,
                                        **kwargs) -> bool:
        """更新统筹智能体实验记录。"""
        return self.update_experiment(
            experiment_id,
            linked_experiments=linked_experiments,
            agent_messages=agent_messages,
            experiment_type="orchestration",
            **kwargs,
        )
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """
        获取实验记录
        
        参数:
            experiment_id: 实验 ID
        
        Returns:
            实验记录字典，如果不存在则返回 None
        """
        record, _ = self._load_record(experiment_id)
        return record
    
    def list_experiments(self, 
                       status: Optional[str] = None,
                       limit: Optional[int] = None,
                       sort_by: str = "timestamp",
                       reverse: bool = True,
                       experiment_type: Optional[str] = None) -> List[Dict]:
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
        history = self._load_history(experiment_type=experiment_type)
        
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
                           top_n: int = 1,
                           experiment_type: Optional[str] = None,
                           task_type: Optional[str] = None,
                           model_family: Optional[str] = None,
                           dataset: Optional[str] = None,
                           implementation: Optional[str] = None,
                           runner: Optional[str] = None) -> List[Dict]:
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
        successful_exps = self.list_experiments(status="success", experiment_type=experiment_type)
        successful_exps = [
            exp for exp in successful_exps
            if self._matches_scope(
                exp,
                task_type=task_type,
                model_family=model_family,
                dataset=dataset,
                implementation=implementation,
                runner=runner,
            )
        ]
        
        # 过滤出有该指标的实验
        valid_exps = []
        for exp in successful_exps:
            metric_value = self._metric_value(exp, metric)
            if metric_value is not None:
                valid_exps.append(exp)
        
        if not valid_exps:
            return []
        
        # 排序
        valid_exps.sort(
            key=lambda x: self._metric_value(x, metric),
            reverse=not minimize
        )
        
        return valid_exps[:top_n]

    @staticmethod
    def _matches_scope(
        experiment: Dict[str, Any],
        task_type: Optional[str] = None,
        model_family: Optional[str] = None,
        dataset: Optional[str] = None,
        implementation: Optional[str] = None,
        runner: Optional[str] = None,
    ) -> bool:
        """Match optional model-agnostic experiment scope fields."""
        task = experiment.get("task") or {}
        model = experiment.get("model") or {}
        execution = experiment.get("execution") or {}
        expected = (
            (task_type, task.get("type")),
            (model_family, model.get("family")),
            (dataset, task.get("dataset")),
            (implementation, model.get("implementation")),
            (runner, execution.get("runner")),
        )
        return all(not wanted or str(actual) == str(wanted) for wanted, actual in expected)
    
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
            for split_metrics in (exp.get("metrics") or {}).values():
                all_metrics.update(split_metrics.keys())
        
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
                "metrics": exp.get('metrics', {})
            }
        
        # 比较每个指标
        for metric in sorted(all_metrics):
            values = []
            for exp in experiments:
                value = self._metric_value(exp, metric)
                if value is not None:
                    values.append((exp['experiment_id'], value))
            
            if values:
                # 判断指标类型
                if all(isinstance(v[1], (int, float)) for v in values):
                    mode = self._metric_mode(experiments, metric)
                    sorted_values = sorted(values, key=lambda x: x[1], reverse=mode == "max")
                    comparison["metrics_comparison"][metric] = {
                        "type": "numeric",
                        "mode": mode,
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

    @staticmethod
    def _metric_value(record: Dict[str, Any], metric: str) -> Optional[Any]:
        for split in ("best", "test", "validation", "train", "summary"):
            values = (record.get("metrics") or {}).get(split, {})
            value = values.get(metric) if isinstance(values, dict) else None
            if value is not None:
                return value
            if split == "best" and isinstance(values, dict):
                if values.get("primary_metric") == metric and values.get("primary_value") is not None:
                    return values["primary_value"]
        return None

    @staticmethod
    def _metric_mode(records: List[Dict[str, Any]], metric: str) -> str:
        for record in records:
            task = record.get("task") or {}
            if task.get("primary_metric") == metric and task.get("metric_mode") in {"min", "max"}:
                return task["metric_mode"]
        normalized = metric.lower()
        maximize = ("accuracy", "precision", "recall", "f1", "auc", "map", "reward")
        return "max" if any(token in normalized for token in maximize) else "min"
    
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
        record, record_dir = self._load_record(experiment_id)
        if record is None:
            return False
        
        # 删除目录
        shutil.rmtree(record_dir / experiment_id)
        
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
    
    def _load_history(self, experiment_type: Optional[str] = None) -> List[Dict]:
        """加载历史记录"""
        history_file = self._get_history_file(experiment_type)
        if not history_file.exists():
            return []
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except (json.JSONDecodeError, Exception):
            # 文件损坏或格式错误，返回空列表
            return []
    
    def _update_history(self, experiment: Dict):
        """更新历史记录"""
        history = self._load_history(experiment.get("experiment_type"))
        history_file = self._get_history_file(experiment.get("experiment_type"))
        
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
        ensure_dir(history_file.parent)
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def _build_record(self,
                       experiment_type: str,
                       experiment_id: str,
                       config_path: str,
                       data_folder: str,
                       output_folder: Optional[str],
                       description: str,
                       config_backup_path: str,
                       extra_fields: Optional[Dict[str, Any]] = None,
                       stage: Optional[str] = None,
                       actor: Optional[Dict[str, Any]] = None,
                       task: Optional[Dict[str, Any]] = None,
                       model: Optional[Dict[str, Any]] = None,
                       execution: Optional[Dict[str, Any]] = None) -> ExperimentRecordDict:
        """根据实验类型生成对应结构的记录。"""
        created_at = datetime.now().isoformat()
        base_kwargs = {
            "experiment_id": experiment_id,
            "timestamp": created_at,
            "updated_at": created_at,
            "started_at": None,
            "completed_at": None,
            "description": description,
            "status": "created",
            "duration_seconds": 0,
            "config_path": config_path,
            "error": None,
            "stage": stage or {
                "data_processing": "data_preparation",
                "orchestration": "orchestration",
            }.get(experiment_type, "optimization"),
            "actor": actor or {
                "type": {
                    "data_processing": "data_processing_agent",
                    "orchestration": "coordinator",
                }.get(experiment_type, "hpo_agent")
            },
            "task": task or {
                "type": "generic",
                "dataset": data_folder,
                "primary_metric": None,
                "metric_mode": None,
            },
            "model": model or {
                "family": "unknown",
                "implementation": "unknown",
                "config_path": config_path,
            },
            "execution": execution or {
                "runner": "unknown",
                "output_folder": output_folder,
                "config_backup_path": config_backup_path,
            },
        }

        if experiment_type == "data_processing":
            record = DataProcessingExperimentRecord(**base_kwargs).to_dict()
        elif experiment_type == "orchestration":
            record = OrchestrationExperimentRecord(**base_kwargs).to_dict()
            record["linked_experiments"] = {}
            record["agent_messages"] = []
        else:
            record = HPOExperimentRecord(**base_kwargs).to_dict()

        if extra_fields:
            for key, value in extra_fields.items():
                if isinstance(value, dict) and isinstance(record.get(key), dict):
                    merged = dict(record.get(key) or {})
                    merged.update(value)
                    record[key] = merged
                else:
                    record[key] = value

        return record
    
    def _remove_from_history(self, experiment_id: str):
        """从历史记录中移除实验"""
        if self._custom_experiments_dir:
            history = [
                exp for exp in self._load_history()
                if exp["experiment_id"] != experiment_id
            ]
            ensure_dir(self.history_file.parent)
            with open(self.history_file, "w", encoding="utf-8") as stream:
                json.dump(history, stream, indent=2, ensure_ascii=False)
            return

        removed = False
        for experiment_type in ("hpo", "data_processing", "orchestration"):
            history = self._load_history(experiment_type)
            filtered = [exp for exp in history if exp['experiment_id'] != experiment_id]
            if len(filtered) != len(history):
                removed = True
            history_file = self._get_history_file(experiment_type)
            ensure_dir(history_file.parent)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(filtered, f, indent=2, ensure_ascii=False)

        if not removed:
            history = self._load_history()
            history = [exp for exp in history if exp['experiment_id'] != experiment_id]
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

    def _get_history_file(self, experiment_type: Optional[str] = None) -> Path:
        if experiment_type is None or self._custom_experiments_dir:
            return self.history_file
        return get_experiment_type_dir(experiment_type) / "experiments_history.json"

    def _load_record(self, experiment_id: str, experiment_type: Optional[str] = None) -> tuple[Optional[Dict], Path]:
        if experiment_type:
            record_dir = self._record_dir(experiment_type)
            record_path = record_dir / experiment_id / "experiment_record.json"
            if record_path.exists():
                with open(record_path, 'r', encoding='utf-8') as f:
                    return json.load(f), record_dir
            return None, record_dir

        candidate_dirs = [self.experiments_dir]
        for candidate_type in ("hpo", "data_processing", "orchestration"):
            record_dir = get_experiment_type_dir(candidate_type)
            if record_dir not in candidate_dirs:
                candidate_dirs.append(record_dir)

        for record_dir in candidate_dirs:
            record_path = record_dir / experiment_id / "experiment_record.json"
            if record_path.exists():
                with open(record_path, 'r', encoding='utf-8') as f:
                    return json.load(f), record_dir

        record_dir = self.experiments_dir
        record_path = record_dir / experiment_id / "experiment_record.json"
        if record_path.exists():
            with open(record_path, 'r', encoding='utf-8') as f:
                return json.load(f), record_dir
        return None, record_dir

    def _record_dir(self, experiment_type: str) -> Path:
        return self.experiments_dir if self._custom_experiments_dir else get_experiment_type_dir(experiment_type)


# 便捷函数
def create_experiment(config_path: str,
                    data_folder: str,
                    output_folder: Optional[str] = None,
                    description: str = "") -> str:
    """快速创建实验的便捷函数"""
    tracker = ExperimentTracker()
    return tracker.create_experiment(config_path, data_folder, output_folder, description)


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
