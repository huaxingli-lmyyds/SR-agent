"""
训练诊断工具集合
基于训练曲线判断收敛、过拟合、欠拟合等状态
"""

from langchain_core.tools import tool
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import re

from agent.utils import ExperimentTracker, get_experiment_dir


def _get_record(experiment_id: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    tracker = ExperimentTracker()
    if experiment_id is None:
        recent = tracker.list_experiments(limit=1)
        if not recent:
            return None, "📋 暂无实验记录"
        experiment_id = recent[0]["experiment_id"]

    record = tracker.get_experiment(experiment_id)
    if not record:
        return None, f"❌ 实验不存在: {experiment_id}"
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
    speechbrain = (record.get("extensions") or {}).get("speechbrain") or {}
    epoch_data = speechbrain.get("epoch_data") or []
    if epoch_data:
        return epoch_data

    train_log_path = next((
        item.get("path") for item in record.get("artifacts") or []
        if item.get("type") == "log" and item.get("name") == "training_log"
    ), None)
    if train_log_path:
        return _parse_train_log(Path(train_log_path))

    exp_dir = get_experiment_dir(record["experiment_id"], "hpo")
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
        return err

    epoch_data = _collect_epoch_data(record)
    if len(epoch_data) < 2:
        return "⚠️  训练曲线数据不足，无法分析"

    window = epoch_data[-max(2, last_n):]
    train_losses = [item["train_loss"] for item in window]
    valid_losses = [item["valid_loss"] for item in window]
    error_rates = [item["valid_error_rate"] for item in window]
    gaps = [v - t for v, t in zip(valid_losses, train_losses)]

    eps = 1e-4
    train_slope = _slope(train_losses)
    valid_slope = _slope(valid_losses)
    error_slope = _slope(error_rates)
    gap_slope = _slope(gaps)

    summary = f"\n📈 训练曲线趋势 - 实验 ID: {record['experiment_id']}\n"
    summary += "=" * 80 + "\n\n"
    summary += f"最近 {len(window)} 个 epoch:\n"
    summary += f"  train_loss: {train_losses[0]:.6f} -> {train_losses[-1]:.6f} ({_trend_label(train_slope, eps)})\n"
    summary += f"  valid_loss: {valid_losses[0]:.6f} -> {valid_losses[-1]:.6f} ({_trend_label(valid_slope, eps)})\n"
    summary += f"  valid_error_rate: {error_rates[0]:.6f} -> {error_rates[-1]:.6f} ({_trend_label(error_slope, eps)})\n"
    summary += f"  gap(valid-train): {gaps[0]:.6f} -> {gaps[-1]:.6f} ({_trend_label(gap_slope, eps)})\n"

    return summary


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
        return err

    epoch_data = _collect_epoch_data(record)
    if len(epoch_data) < 3:
        return "⚠️  训练曲线数据不足，无法诊断"

    window = epoch_data[-max(3, last_n):]
    train_losses = [item["train_loss"] for item in window]
    valid_losses = [item["valid_loss"] for item in window]
    error_rates = [item["valid_error_rate"] for item in window]
    gaps = [v - t for v, t in zip(valid_losses, train_losses)]

    eps = 1e-4
    train_slope = _slope(train_losses)
    valid_slope = _slope(valid_losses)
    error_slope = _slope(error_rates)
    gap_slope = _slope(gaps)
    gap_latest = gaps[-1]

    train_improve = (train_losses[0] - train_losses[-1]) / max(train_losses[0], 1e-8)
    valid_improve = (valid_losses[0] - valid_losses[-1]) / max(valid_losses[0], 1e-8)

    status = "stable"
    reasons: List[str] = []

    if train_slope < -eps and valid_slope > eps:
        status = "overfitting"
        reasons.append("train_loss 持续下降但 valid_loss 上升")
    if gap_slope > eps and gap_latest > 0.1:
        status = "overfitting"
        reasons.append("验证损失与训练损失差距扩大")

    if train_improve < 0.01 and valid_improve < 0.01 and error_rates[-1] > 0.05:
        status = "underfitting"
        reasons.append("训练/验证损失几乎不下降且 valid_error_rate 偏高")
    if abs(train_slope) < eps and abs(valid_slope) < eps and error_rates[-1] > 0.05:
        status = "underfitting"
        reasons.append("训练/验证曲线平坦且效果未达标")

    if status == "stable":
        if valid_slope < -eps or error_slope < -eps:
            reasons.append("验证指标仍在改善")
        else:
            reasons.append("曲线变化较小，可能接近收敛")

    summary = f"\n🧪 拟合状态诊断 - 实验 ID: {record['experiment_id']}\n"
    summary += "=" * 80 + "\n\n"
    summary += f"状态判断: {status}\n"
    for reason in reasons:
        summary += f"- {reason}\n"

    summary += "\n建议:\n"
    if status == "overfitting":
        summary += "- 考虑增大正则化、减少轮数或增大数据增强\n"
    elif status == "underfitting":
        summary += "- 考虑提高模型容量或训练轮数，或调整学习率\n"
    else:
        summary += "- 可继续微调参数或执行评估\n"

    return summary


__all__ = [
    "AnalyzeTrainingCurves",
    "DiagnoseFitStatus",
]
