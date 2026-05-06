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
    get_config_file,
    get_experiments_dir,
    ensure_dir,
    get_experiment_log_path,
    yaml_to_dict,
    get_project_root,
    ExperimentTracker,
)
from agent.utils.runner import run_training

# 全局路径
CONFIG_PATH = str(get_config_file("train_ecapa_tdnn.yaml"))


def _find_model_paths(output_folder: Optional[str], exp_dir: Path) -> List[str]:
    candidates: List[Path] = []

    if output_folder:
        out_dir = Path(output_folder)
        if out_dir.exists():
            candidates.extend(out_dir.glob("*.ckpt"))
            candidates.extend(out_dir.glob("*.pt"))

    if exp_dir.exists():
        candidates.extend(exp_dir.glob("*.ckpt"))
        candidates.extend(exp_dir.glob("*.pt"))

    # Deduplicate and sort by mtime desc
    uniq = {p.resolve(): p for p in candidates}
    sorted_paths = sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in sorted_paths]


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return get_project_root() / path


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
               description: Optional[str] = None) -> str:
    """
    运行 ECAPA-TDNN 模型的训练脚本。
    训练完成后会自动保存实验记录，包括配置、训练日志和结果。
    
    参数:
        config_path: 配置文件路径，如果为 None 则使用默认配置
        data_folder: 数据文件夹路径，如果为 None 则使用配置文件中的设置
        description: 实验描述（可选）
    
    Returns:
        str: 训练输出或错误信息
    """
    try:
        # 参数校验
        config_path_str = config_path if config_path else CONFIG_PATH
        if not Path(config_path_str).exists():
            return f"config_path not found: {config_path_str}"

        # 使用 ConfigParser 读取配置（支持 YAML 引用解析）
        from agent.utils import ConfigParser
        parser = ConfigParser(config_path_str)
        config_data = parser.load_config(resolve_references=True)

        # 确定数据文件夹
        if data_folder:
            df = data_folder
        elif config_data.get("data_folder"):
            df = str(config_data.get("data_folder"))
        else:
            df = "../datasets/voxceleb1"

        if not df or df == "!PLACEHOLDER":
            df = "../datasets/voxceleb1"

        # experiment_tracker 创建记录
        start_time = datetime.now()
        tracker = ExperimentTracker()
        experiment_id = tracker.create_experiment(
            config=yaml_to_dict(config_data),
            config_path=config_path_str,
            data_folder=df,
            description=description or "",
        )

        exp_dir = ensure_dir(get_experiments_dir() / experiment_id)

        # runner.run_training()
        overrides = [f"data_folder: {df}"]
        train_result = run_training(config_path_str, overrides)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        status = "success"
        error_msg = None
        if isinstance(train_result, dict) and train_result.get("status") == "failed":
            status = "failed"
            error_msg = train_result.get("error")

        eer = train_result.get("eer") if isinstance(train_result, dict) else None
        output_folder = config_data.get("output_folder")
        output_folder_path = _resolve_path(str(output_folder)) if output_folder else None
        model_paths = _find_model_paths(
            str(output_folder_path) if output_folder_path else None,
            exp_dir,
        )
        train_log_path = _resolve_path(str(config_data.get("train_log")))
        if train_log_path is None and output_folder_path is not None:
            train_log_path = output_folder_path / "train_log.txt"

        epoch_data, final_metrics = (
            _parse_training_log(train_log_path) if train_log_path else ([], {})
        )

        training_metrics = {"eer": eer}
        if final_metrics:
            training_metrics.update(final_metrics)

        # experiment_tracker 更新记录
        tracker.update_experiment(
            experiment_id=experiment_id,
            status=status,
            duration=duration,
            error=error_msg,
            results={"eer": eer},
            training={
                "output_folder": str(output_folder_path) if output_folder_path else None,
                "metrics": training_metrics,
                "model_paths": model_paths,
                "train_log_path": str(train_log_path) if train_log_path else None,
                "epoch_data": epoch_data,
                "final_metrics": final_metrics,
            },
        )

        log_tail = ""
        if train_log_path and Path(train_log_path).exists():
            with open(train_log_path, "r", encoding="utf-8", errors="ignore") as fin:
                lines = fin.readlines()
                log_tail = "".join(lines[-20:])
        else:
            log_tail = "(train_log not found)"

        result_lines = [
            f"Experiment ID: {experiment_id}",
            f"EER: {eer if eer is not None else 'N/A'}",
            f"Experiment dir: {exp_dir}",
            "Training log (last 20 lines):",
            log_tail.rstrip() or "(empty)",
        ]
        if status == "failed" and error_msg:
            result_lines.append(f"Error: {error_msg}")

        return "\n".join(result_lines)
    except Exception as e:
        return f"run training failed: {str(e)}"


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
        exp_dir = get_experiments_dir()
        
        if experiment_id:
            target_dir = exp_dir / experiment_id
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

        training_info = record.get("training", {})
        model_paths = training_info.get("model_paths") or []
        output_folder = training_info.get("output_folder")
        if not model_paths:
            model_paths = _find_model_paths(output_folder, target_dir)

        model_path = model_paths[0] if model_paths else None
        data_folder = training_info.get("data_folder")

        if not model_path:
            return "⚠️  未找到模型参数文件，无法评估"

        from agent.tools.evaluation_tools import RunEvaluation
        RunEvaluation.invoke({
            "model_path": model_path,
            "data_folder": data_folder,
            "experiment_id": experiment_id
        })

        updated = tracker.get_experiment(experiment_id) or {}
        evaluation = updated.get("evaluation")
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
        
        exp_dir = get_experiments_dir()
        
        if experiment_id:
            target_dir = exp_dir / experiment_id
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
        log_path = get_experiment_log_path(experiment_id)
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
            
            if record.get('results'):
                summary += "\n保存的结果:\n"
                for key, value in record['results'].items():
                    if key != 'final_output':
                        summary += f"  {key}: {value}\n"
        
        return summary
    
    except Exception as e:
        return f"❌ 分析结果失败: {str(e)}"


# 导出所有工具
__all__ = [
    'TrainModel',
    'EvaluateModel',
    'AnalyzeResults',
]