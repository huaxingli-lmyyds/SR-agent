"""
Reward scoring tools for model selection.
"""

from langchain_core.tools import tool
from typing import Optional, Dict, Any
import json

from agent.utils import ExperimentTracker
from agent.utils.reward import compute_reward


def _normalize_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = record.get("evaluation") or {}
    eval_results = evaluation.get("results") or {}
    results = record.get("results") or {}

    metrics = {
        "eer": eval_results.get("eer", results.get("eer")),
        "min_dcf": eval_results.get("min_dcf", results.get("min_dcf")),
    }

    return metrics


@tool
def ScoreExperiment(experiment_id: Optional[str] = None,
                    weights_json: Optional[str] = None) -> str:
    """
    根据实验记录计算综合奖励分数。

    参数:
        experiment_id: 实验 ID，None 则使用最近实验
        weights_json: 权重 JSON 字符串（可选），示例: {"min_dcf": 0.4}

    Returns:
        str: 奖励评分摘要
    """
    tracker = ExperimentTracker()
    if experiment_id is None:
        recent = tracker.list_experiments(limit=1)
        if not recent:
            return "📋 暂无实验记录"
        experiment_id = recent[0]["experiment_id"]

    record = tracker.get_experiment(experiment_id)
    if not record:
        return f"❌ 实验不存在: {experiment_id}"

    weights: Optional[Dict[str, float]] = None
    if weights_json:
        try:
            parsed = json.loads(weights_json)
            if isinstance(parsed, dict):
                weights = parsed
        except json.JSONDecodeError as exc:
            return f"weights_json JSON decode failed: {exc}"

    metrics = _normalize_metrics(record)
    reward, breakdown = compute_reward(metrics, weights)
    if reward is None:
        return "⚠️  缺少 EER，无法计算奖励分数"

    summary = f"\n🎯 奖励评分 - 实验 ID: {experiment_id}\n"
    summary += "=" * 80 + "\n\n"
    summary += f"综合得分: {reward:.6f}\n"
    summary += "分项贡献:\n"
    for key, value in breakdown.items():
        summary += f"  - {key}: {value:.6f}\n"

    summary += "\n使用指标:\n"
    for key, value in metrics.items():
        summary += f"  - {key}: {value if value is not None else 'N/A'}\n"

    return summary


__all__ = ["ScoreExperiment"]
