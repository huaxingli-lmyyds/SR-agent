"""
训练管理工具集合
提供模型训练、训练监控、训练控制等功能
使用 LangChain 工具接口，基于 utils 模块实现
"""

from langchain_core.tools import tool
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
import re

# 导入 utils 模块
from agent.utils import (
    get_experiment_log_path,
    get_experiment_dir,
    resolve_config_path,
    resolve_data_path,
    resolve_optional_project_path,
    ExperimentTracker,
)
from agent.utils.runner import run_training
from agent.core.adapters import RUNNER_ADAPTERS
from agent.core.contracts import OperationResult
from agent.core.experiment_service import ExperimentService

# 全局路径
def _find_model_paths(output_folder: Optional[str], exp_dir: Path) -> List[str]:
    candidates: List[Path] = []
    ckpt_scores: Dict[Path, float] = {}
    ckpt_dirs: List[Path] = []

    save_dirs: List[Path] = []
    if output_folder:
        save_dirs.append(Path(output_folder) / "save")
    if exp_dir.exists():
        save_dirs.append(exp_dir / "output" / "save")
        save_dirs.append(exp_dir / "results" / "save")

    for save_dir in save_dirs:
        if not save_dir.exists():
            continue
        candidates.extend(save_dir.glob("*.ckpt"))
        candidates.extend(save_dir.glob("*.pt"))
        for ckpt_dir in save_dir.glob("CKPT+*"):
            ckpt_dirs.append(ckpt_dir)
            meta_path = ckpt_dir / "CKPT.yaml"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as meta_file:
                    for line in meta_file:
                        if line.strip().startswith("ErrorRate:"):
                            _, value = line.split(":", 1)
                            ckpt_scores[ckpt_dir] = float(value.strip())
                            break
            except (OSError, ValueError):
                continue

    if output_folder:
        out_dir = Path(output_folder)
        if out_dir.exists():
            candidates.extend(out_dir.glob("*.ckpt"))
            candidates.extend(out_dir.glob("*.pt"))

    if exp_dir.exists():
        candidates.extend(exp_dir.glob("*.ckpt"))
        candidates.extend(exp_dir.glob("*.pt"))

    if ckpt_scores:
        best_ckpt_dir = min(ckpt_scores, key=ckpt_scores.get)
        return [str(best_ckpt_dir)]

    if ckpt_dirs:
        newest_ckpt_dir = max(ckpt_dirs, key=lambda p: p.stat().st_mtime)
        return [str(newest_ckpt_dir)]

    # Deduplicate
    uniq = {p.resolve(): p for p in candidates}
    if not uniq:
        return []

    best_path = min(uniq.values(), key=lambda p: p.stat().st_mtime)
    return [str(best_path)]


def _find_output_folder(
    exp_dir: Path,
    output_folder_path: Optional[Path],
    config_output_folder: Optional[str],
) -> Optional[Path]:
    candidates: List[Path] = []

    if output_folder_path is not None:
        candidates.append(output_folder_path)

    if config_output_folder:
        resolved = resolve_optional_project_path(str(config_output_folder))
        if resolved is not None:
            candidates.append(resolved)

    candidates.append(exp_dir / "output")
    candidates.append(exp_dir / "results")

    for candidate in candidates:
        if (candidate / "train_log.txt").exists() or (candidate / "save").exists():
            return candidate

    return output_folder_path


def _parse_training_log(train_log_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    epoch_data: List[Dict[str, Any]] = []
    final_metrics: Dict[str, Any] = {}

    if not train_log_path.exists():
        return epoch_data, final_metrics

    pattern = re.compile(
        r"epoch:\s*(\d+),\s*lr:\s*([\d.e+-]+)\s*-\s*"
        r"train loss:\s*([\d.e+-]+)\s*-\s*"
        r"valid loss:\s*([\d.e+-]+),\s*valid ErrorRate:\s*([\d.e+-]+)"
    )

    with open(train_log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            epoch_data.append(
                {
                    "epoch": int(match.group(1)),
                    "lr": float(match.group(2)),
                    "train_loss": float(match.group(3)),
                    "valid_loss": float(match.group(4)),
                    "valid_error_rate": float(match.group(5)),
                }
            )

    if epoch_data:
        final_epoch = epoch_data[-1]
        best_epoch = min(epoch_data, key=lambda item: item["valid_error_rate"])
        final_metrics = {
            "final_epoch": final_epoch["epoch"],
            "final_lr": final_epoch["lr"],
            "final_train_loss": final_epoch["train_loss"],
            "final_valid_loss": final_epoch["valid_loss"],
            "final_valid_error_rate": final_epoch["valid_error_rate"],
            "total_epochs": len(epoch_data),
            "best_epoch": best_epoch["epoch"],
            "best_valid_loss": best_epoch["valid_loss"],
            "best_error_rate": best_epoch["valid_error_rate"],
        }

    return epoch_data, final_metrics


@tool
def TrainModel(config_path: Optional[str] = None,
               data_folder: Optional[str] = None,
               description: Optional[str] = None,
               experiment_id: Optional[str] = None) -> str:
    """
    运行 ECAPA-TDNN 模型的训练脚本。
    训练完成后会自动保存实验记录，包括配置、训练日志和结果。
    
    参数:
        config_path: 配置文件路径，如果为 None 则使用默认配置
        data_folder: 数据文件夹路径，如果为 None 则使用配置文件中的设置
        description: 实验描述（可选）
        experiment_id: 实验 ID（可选，存在时继续训练，不存在时报错）
    
    Returns:
        str: 训练输出或错误信息
    """
    try:
        # 参数校验
        config_path_str = str(resolve_config_path(config_path))

        # experiment_tracker 创建记录或复用已有记录
        start_time = datetime.now()
        tracker = ExperimentTracker()
        record = None
        if experiment_id:
            record = tracker.get_experiment(experiment_id)
            if record is None:
                return OperationResult(
                    status="failed",
                    stage="training",
                    error=f"experiment_id not found: {experiment_id}",
                    experiment_id=experiment_id,
                ).to_json()

            execution_info = record.get("execution", {})
            backup_path = execution_info.get("config_backup_path")
            record_config_path = record.get("config_path")
            if not config_path:
                if backup_path and Path(backup_path).exists():
                    config_path_str = backup_path
                elif record_config_path and Path(record_config_path).exists():
                    config_path_str = record_config_path

        if not Path(config_path_str).exists():
            return OperationResult(
                status="failed",
                stage="training",
                error=f"config_path not found: {config_path_str}",
                experiment_id=experiment_id,
            ).to_json()

        # 使用 ConfigParser 读取配置（支持 YAML 引用解析）
        from agent.utils import ConfigParser
        parser = ConfigParser(config_path_str)
        config_data = parser.load_config(resolve_references=True)

        # 确定数据文件夹
        if data_folder:
            df = str(resolve_data_path(data_folder))
        else:
            df = str(resolve_data_path(config_data.get("data_folder")))

        output_folder = config_data.get("output_folder")
        if record:
            execution_info = record.get("execution", {})
            if execution_info.get("output_folder"):
                output_folder = execution_info.get("output_folder")
            if (record.get("task") or {}).get("dataset"):
                df = str(resolve_data_path(record["task"]["dataset"]))
        else:
            experiment_id = tracker.create_hpo_experiment(
                config_path=config_path_str,
                data_folder=df,
                output_folder=str(output_folder) if output_folder else None,
                description=description or "",
            )

        exp_dir = get_experiment_dir(experiment_id, "hpo", create=True)

        output_folder_path = resolve_optional_project_path(output_folder)
        if output_folder_path is None or exp_dir not in output_folder_path.parents:
            output_folder_path = exp_dir / "output"
            output_folder = str(output_folder_path)

        # runner.run_training()
        overrides = {
            "data_folder": df,
            "output_folder": str(output_folder_path),
        }
        train_result = run_training(config_path_str, overrides)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        status = "success"
        error_msg = None
        if isinstance(train_result, dict) and train_result.get("status") == "failed":
            status = "failed"
            error_msg = train_result.get("error")

        valid_error_rate = train_result.get("valid_error_rate") if isinstance(train_result, dict) else None
        output_folder_path = _find_output_folder(
            exp_dir,
            output_folder_path,
            config_data.get("output_folder"),
        )
        output_folder = str(output_folder_path) if output_folder_path else None

        model_paths = _find_model_paths(
            str(output_folder_path) if output_folder_path else None,
            exp_dir,
        )
        train_log_path = (
            output_folder_path / "train_log.txt" if output_folder_path else None
        )

        epoch_data, final_metrics = (
            _parse_training_log(train_log_path) if train_log_path else ([], {})
        )

        training_metrics = {"valid_error_rate": valid_error_rate}
        if final_metrics:
            training_metrics.update(final_metrics)

        # experiment_tracker 更新记录
        operation_result = RUNNER_ADAPTERS["speechbrain"].normalize_training_result({
            "status": status,
            "error": error_msg,
            "output_folder": str(output_folder_path) if output_folder_path else None,
            "metrics": training_metrics,
            "model_paths": model_paths,
            "train_log_path": str(train_log_path) if train_log_path else None,
            "epoch_data": epoch_data,
            "final_metrics": final_metrics,
        })
        operation_result.task = (record or {}).get("task") or {
            "type": "speaker_verification",
            "dataset": df,
            "primary_metric": "eer",
            "metric_mode": "min",
        }
        operation_result.model = (record or {}).get("model") or {
            "family": "ecapa_tdnn",
            "implementation": "speechbrain",
            "config_path": config_path_str,
        }
        operation_result.execution.update({
            "runner": "speechbrain",
            "output_folder": output_folder,
        })
        operation_result.parameters = dict(overrides)
        ExperimentService(tracker).record_result(
            experiment_id,
            operation_result,
            duration_seconds=duration,
            actor={"type": "hpo_agent", "name": "model_optimizer"},
        )
        return operation_result.to_json()
    except Exception as e:
        return OperationResult(
            status="failed",
            stage="training",
            error=str(e),
            experiment_id=experiment_id,
        ).to_json()


@tool
def EvaluateModel(experiment_id: Optional[str] = None) -> str:
    """
    评估指定实验的模型性能。
    
    参数:
        experiment_id: 实验 ID，如果为 None 则评估最近的实验
    
    Returns:
        str: 评估结果
    """
    try:
        tracker = ExperimentTracker()
        exp_dir = tracker.experiments_dir
        
        if experiment_id:
            target_dir = get_experiment_dir(experiment_id, "hpo")
            if not target_dir.exists():
                return f"❌ 实验不存在: {experiment_id}"
        else:
            # 获取最近的实验
            exps = sorted(exp_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if not exps:
                return "📋 暂无实验记录"
            target_dir = exps[0]
            experiment_id = target_dir.name
        
        tracker = ExperimentTracker()
        record = tracker.get_experiment(experiment_id)
        if not record:
            return f"⚠️  实验记录文件不存在: {target_dir / 'experiment_record.json'}"

        status = record.get('status', 'unknown')
        if status != "success":
            return f"⚠️  实验未成功完成，无法评估 (状态: {status})"

        model_paths = [
            item.get("path") for item in record.get("artifacts") or []
            if item.get("type") == "checkpoint"
        ]
        output_folder = (record.get("execution") or {}).get("output_folder")
        if not model_paths:
            model_paths = _find_model_paths(output_folder, target_dir)

        model_path = model_paths[0] if model_paths else None
        data_folder = (record.get("task") or {}).get("dataset")

        if not model_path:
            return "⚠️  未找到模型参数文件，无法评估"

        from agent.tools.evaluation_tools import RunEvaluation
        RunEvaluation.invoke({
            "model_path": model_path,
            "data_folder": data_folder,
            "experiment_id": experiment_id
        })

        updated = tracker.get_experiment(experiment_id) or {}
        evaluation = {"status": updated.get("status"), "results": (updated.get("metrics") or {}).get("test")}
        if not evaluation:
            return "⚠️  评估未写入实验记录"

        eval_results = evaluation.get("results", {})
        record_path = target_dir / "experiment_record.json"

        summary = f"\n📊 模型评估 - 实验 ID: {experiment_id}\n"
        summary += "=" * 80 + "\n\n"
        summary += f"评估状态: {evaluation.get('status', 'unknown')}\n"

        if eval_results:
            summary += "性能指标:\n"
            for key, value in eval_results.items():
                summary += f"  {key}: {value}\n"

        summary += f"\n📁 文件位置:\n"
        summary += f"  - 实验记录: {record_path}\n"

        return summary
    
    except Exception as e:
        return f"❌ 评估模型失败: {str(e)}"


@tool
def AnalyzeResults(experiment_id: Optional[str] = None) -> str:
    """
    分析实验结果，包括训练日志中的关键指标和趋势。
    
    参数:
        experiment_id: 实验 ID，如果为 None 则分析最近的实验
    
    Returns:
        str: 分析结果
    """
    try:
        from agent.utils import extract_log_metrics
        
        tracker = ExperimentTracker()
        exp_dir = tracker.experiments_dir
        
        if experiment_id:
            target_dir = get_experiment_dir(experiment_id, "hpo")
            if not target_dir.exists():
                return f"❌ 实验不存在: {experiment_id}"
        else:
            # 获取最近的实验
            exps = sorted(exp_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if not exps:
                return "📋 暂无实验记录"
            target_dir = exps[0]
            experiment_id = target_dir.name
        
        # 查找日志文件
        log_path = get_experiment_log_path(experiment_id, "hpo")
        if not log_path.exists():
            log_files = list(target_dir.glob("*.log")) + list(target_dir.glob("*.txt"))
            if log_files:
                log_path = log_files[0]
            else:
                return f"⚠️  未找到日志文件"
        
        # 提取指标
        metrics = extract_log_metrics(str(log_path))
        
        summary = f"\n📊 实验分析 - 实验 ID: {experiment_id}\n"
        summary += "=" * 80 + "\n\n"
        
        if isinstance(metrics, dict) and "error" in metrics:
            return f"⚠️  {metrics['error']}\n{summary}"
        
        summary += "提取的指标:\n"
        for key, value in metrics.items():
            summary += f"  {key}: {value}\n"
        
        # 读取实验记录
        record_path = target_dir / "experiment_record.json"
        if record_path.exists():
            with open(record_path, 'r', encoding='utf-8') as f:
                record = json.load(f)
            
            if record.get("metrics"):
                summary += "\n保存的结果:\n"
                for split, values in record["metrics"].items():
                    summary += f"  {split}: {values}\n"
        
        return summary
    
    except Exception as e:
        return f"❌ 分析结果失败: {str(e)}"


# 导出所有工具
__all__ = [
    'TrainModel',
    'EvaluateModel',
    'AnalyzeResults',
]
