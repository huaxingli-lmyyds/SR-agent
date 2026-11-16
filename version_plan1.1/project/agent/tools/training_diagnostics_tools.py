"""
训练诊断工具集合
基于训练曲线判断收敛、过拟合、欠拟合等状态
"""

from langchain_core.tools import tool
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json
import re

from agent.utils import ExperimentTracker, get_experiment_dir


def _get_record(experiment_id: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    tracker = ExperimentTracker()
    if experiment_id is None:
        recent = tracker.list_experiments(limit=1)
        if not recent:
            return None, "no experiment record"
        experiment_id = recent[0]["experiment_id"]

    record = tracker.get_experiment(experiment_id)
    if not record:
        return None, f"experiment not found: {experiment_id}"
    return record, None


def _parse_train_log(train_log_path: Path) -> List[Dict[str, Any]]:
    if not train_log_path.exists():
        return []

    pattern = re.compile(
        r"epoch:\s*(\d+),\s*lr:\s*([\d.e+-]+)\s*-\s*"
        r"train loss:\s*([\d.e+-]+)\s*-\s*"
        r"valid loss:\s*([\d.e+-]+),\s*valid ErrorRate:\s*([\d.e+-]+)"
    )

    epoch_data: List[Dict[str, Any]] = []
    with open(train_log_path, "r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
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
    return epoch_data


def _collect_epoch_data(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    extensions = record.get("extensions") or {}
    epoch_data = ((record.get("execution") or {}).get("training_history") or [])
    if not epoch_data:
        for extension in extensions.values():
            if isinstance(extension, dict) and extension.get("epoch_data"):
                epoch_data = extension["epoch_data"]
                break
    if epoch_data:
        return epoch_data

    train_log_path = next((
        item.get("path") for item in record.get("artifacts") or []
        if item.get("type") == "log" and item.get("name") == "training_log"
    ), None)
    if train_log_path:
        return _parse_train_log(Path(train_log_path))

    exp_dir = get_experiment_dir(
        record["experiment_id"],
        record.get("experiment_type") or "hpo",
    )
    fallback_log = exp_dir / "train_log.txt"
    if fallback_log.exists():
        return _parse_train_log(fallback_log)

    return []


def _slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return (values[-1] - values[0]) / float(len(values) - 1)


def _trend_label(value: float, eps: float) -> str:
    if value > eps:
        return "up"
    if value < -eps:
        return "down"
    return "flat"


def _diagnostic_series(
    record: Dict[str, Any],
    epoch_data: List[Dict[str, Any]],
    last_n: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    required = ("train_loss", "valid_loss")
    if any(any(key not in item for key in required) for item in epoch_data):
        return None, "training history must contain train_loss and valid_loss"
    task = record.get("task") or {}
    primary_metric = task.get("primary_metric") or "valid_error_rate"
    metric_mode = task.get("metric_mode") or "min"
    metric_key = primary_metric
    if not all(metric_key in item for item in epoch_data):
        metric_key = "valid_error_rate"
    if not all(metric_key in item for item in epoch_data):
        return None, f"training history does not contain primary metric: {primary_metric}"
    window = epoch_data[-last_n:]
    return {
        "window": window,
        "primary_metric": primary_metric,
        "metric_key": metric_key,
        "metric_mode": metric_mode,
        "train_losses": [item["train_loss"] for item in window],
        "valid_losses": [item["valid_loss"] for item in window],
        "metric_values": [item[metric_key] for item in window],
    }, None


@tool
def AnalyzeTrainingCurves(experiment_id: Optional[str] = None, last_n: int = 5) -> str:
    """
    基于训练曲线数据给出趋势摘要。

    参数:
        experiment_id: 实验 ID，None 则取最近实验
        last_n: 分析最近 N 个 epoch

    Returns:
        str: 趋势摘要
    """
    record, err = _get_record(experiment_id)
    if err:
        return json.dumps({"status": "failed", "error": err}, ensure_ascii=False)

    epoch_data = _collect_epoch_data(record)
    if len(epoch_data) < 2:
        return json.dumps({"status": "failed", "error": "insufficient training history"}, ensure_ascii=False)

    series, err = _diagnostic_series(record, epoch_data, max(2, last_n))
    if err:
        return json.dumps({"status": "failed", "error": err}, ensure_ascii=False)
    window = series["window"]
    train_losses = series["train_losses"]
    valid_losses = series["valid_losses"]
    error_rates = series["metric_values"]
    gaps = [v - t for v, t in zip(valid_losses, train_losses)]

    eps = 1e-4
    train_slope = _slope(train_losses)
    valid_slope = _slope(valid_losses)
    error_slope = _slope(error_rates)
    gap_slope = _slope(gaps)

    return json.dumps({
        "status": "success",
        "experiment_id": record["experiment_id"],
        "primary_metric": series["primary_metric"],
        "window_size": len(window),
        "trends": {
            "train_loss": _trend_label(train_slope, eps),
            "valid_loss": _trend_label(valid_slope, eps),
            series["metric_key"]: _trend_label(error_slope, eps),
            "generalization_gap": _trend_label(gap_slope, eps),
        },
        "slopes": {
            "train_loss": train_slope,
            "valid_loss": valid_slope,
            series["metric_key"]: error_slope,
            "generalization_gap": gap_slope,
        },
    }, ensure_ascii=False)


@tool
def DiagnoseFitStatus(experiment_id: Optional[str] = None, last_n: int = 5) -> str:
    """
    根据训练/验证曲线给出过拟合或欠拟合的启发式判断。

    参数:
        experiment_id: 实验 ID，None 则取最近实验
        last_n: 分析最近 N 个 epoch

    Returns:
        str: 诊断结果
    """
    record, err = _get_record(experiment_id)
    if err:
        return json.dumps({"status": "failed", "error": err}, ensure_ascii=False)

    epoch_data = _collect_epoch_data(record)
    if len(epoch_data) < 3:
        return json.dumps({"status": "failed", "error": "insufficient training history"}, ensure_ascii=False)

    series, err = _diagnostic_series(record, epoch_data, max(3, last_n))
    if err:
        return json.dumps({"status": "failed", "error": err}, ensure_ascii=False)
    train_losses = series["train_losses"]
    valid_losses = series["valid_losses"]
    error_rates = series["metric_values"]
    gaps = [v - t for v, t in zip(valid_losses, train_losses)]

    eps = 1e-4
    train_slope = _slope(train_losses)
    valid_slope = _slope(valid_losses)
    error_slope = _slope(error_rates)
    gap_slope = _slope(gaps)
    gap_latest = gaps[-1]

    train_improve = (train_losses[0] - train_losses[-1]) / max(abs(train_losses[0]), 1e-8)
    valid_improve = (valid_losses[0] - valid_losses[-1]) / max(abs(valid_losses[0]), 1e-8)

    status = "stable"
    reasons: List[str] = []

    if train_slope < -eps and valid_slope > eps:
        status = "overfitting"
        reasons.append("train_loss 持续下降但 valid_loss 上升")
    if gap_slope > eps and gap_latest > 0.1:
        status = "overfitting"
        reasons.append("验证损失与训练损失差距扩大")

    if train_improve < 0.01 and valid_improve < 0.01:
        status = "underfitting"
        reasons.append("训练和验证损失几乎不下降")
    if abs(train_slope) < eps and abs(valid_slope) < eps:
        status = "underfitting"
        reasons.append("训练和验证曲线平坦")

    if status == "stable":
        metric_improving = error_slope < -eps if series["metric_mode"] == "min" else error_slope > eps
        if valid_slope < -eps or metric_improving:
            reasons.append("验证指标仍在改善")
        else:
            reasons.append("曲线变化较小，可能接近收敛")

    recommendations = []
    if status == "overfitting":
        recommendations.append("increase_regularization_or_data_augmentation")
    elif status == "underfitting":
        recommendations.append("increase_capacity_or_training_budget")
    else:
        recommendations.append("continue_tuning_or_evaluate")

    return json.dumps({
        "status": "success",
        "experiment_id": record["experiment_id"],
        "fit_status": status,
        "primary_metric": series["primary_metric"],
        "metric_mode": series["metric_mode"],
        "reasons": reasons,
        "recommendations": recommendations,
    }, ensure_ascii=False)


__all__ = [
    "AnalyzeTrainingCurves",
    "DiagnoseFitStatus",
]
