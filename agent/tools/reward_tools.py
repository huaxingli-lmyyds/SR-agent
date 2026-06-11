"""
Reward scoring tools for model selection.
"""

from langchain_core.tools import tool
from typing import Optional, Dict, Any
import json

from agent.utils import ExperimentTracker
from agent.utils.reward import compute_objective_reward


def _normalize_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    metrics = record.get("metrics") or {}
    return {
        "eer": (metrics.get("test") or {}).get("eer", (metrics.get("validation") or {}).get("eer")),
        "min_dcf": (metrics.get("test") or {}).get("min_dcf"),
    }


def _recorded_metric(record: Dict[str, Any], metric: str) -> Any:
    for split in ("test", "validation", "train", "summary"):
        value = ((record.get("metrics") or {}).get(split) or {}).get(metric)
        if value is not None:
            return value
    return None


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
            return json.dumps({"status": "failed", "error": "no experiment record"}, ensure_ascii=False)
        experiment_id = recent[0]["experiment_id"]

    record = tracker.get_experiment(experiment_id)
    if not record:
        return json.dumps({"status": "failed", "error": "experiment not found", "experiment_id": experiment_id}, ensure_ascii=False)

    weights: Optional[Dict[str, float]] = None
    if weights_json:
        try:
            parsed = json.loads(weights_json)
            if isinstance(parsed, dict):
                weights = parsed
        except json.JSONDecodeError as exc:
            return json.dumps({"status": "failed", "error": f"weights_json JSON decode failed: {exc}"}, ensure_ascii=False)

    task = record.get("task") or {}
    primary_metric = task.get("primary_metric") or "eer"
    metric_mode = task.get("metric_mode") or "min"
    metrics = _normalize_metrics(record)
    if primary_metric not in metrics:
        metrics[primary_metric] = _recorded_metric(record, primary_metric)
    reward, breakdown = compute_objective_reward(
        metrics,
        primary_metric=primary_metric,
        mode=metric_mode,
        weights=weights,
    )
    if reward is None:
        return json.dumps({
            "status": "failed",
            "error": f"missing primary metric: {primary_metric}",
            "experiment_id": experiment_id,
        }, ensure_ascii=False)

    return json.dumps({
        "status": "success",
        "experiment_id": experiment_id,
        "primary_metric": primary_metric,
        "metric_mode": metric_mode,
        "reward": reward,
        "breakdown": breakdown,
        "metrics": metrics,
    }, ensure_ascii=False, default=str)


__all__ = ["ScoreExperiment"]
