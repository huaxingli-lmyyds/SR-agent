"""
实验历史工具集合
提供实验记录查询与比较能力，供智能体调用
"""

from langchain_core.tools import tool
from typing import Optional, List
from datetime import datetime

from agent.utils import ExperimentTracker


@tool
def CompareExperiments(experiment_ids: Optional[List[str]] = None,
                       metric: str = "eer") -> str:
    """
    比较多个实验的结果（训练/评估统一比较）。

    参数:
        experiment_ids: 实验 ID 列表，如果为 None 则比较最近的 5 个实验
        metric: 比较的指标，默认 'eer'，可选 'min_dcf', 'accuracy', 'error_rate', 'loss'

    Returns:
        str: 比较结果
    """
    try:
        tracker = ExperimentTracker()
        if not experiment_ids:
            recent = tracker.list_experiments(limit=5)
            if not recent:
                return "📋 暂无实验记录"
            experiment_ids = [exp["experiment_id"] for exp in recent]

        records = []
        for exp_id in experiment_ids:
            record = tracker.get_experiment(exp_id)
            if record:
                records.append(record)

        if not records:
            return "❌ 没有找到有效的实验记录"

        summary = f"\n📊 实验比较 - 基于指标: {metric}\n"
        summary += "=" * 80 + "\n\n"

        valid_records = [
            r for r in records
            if r.get('results') and r['results'].get(metric) is not None
        ]
        if not valid_records:
            return f"⚠️  没有实验包含指标 '{metric}'"

        reverse = metric in ['accuracy', 'precision', 'recall', 'f1']
        sorted_records = sorted(
            valid_records,
            key=lambda x: x['results'][metric],
            reverse=reverse,
        )

        for i, record in enumerate(sorted_records, 1):
            exp_id = record['experiment_id']
            value = record['results'][metric]
            status = record.get('status', 'unknown')
            duration = record.get('duration_seconds', 0)

            summary += f"{i}. 实验 {exp_id}\n"
            summary += f"   {metric}: {value:.6f}\n"
            summary += f"   状态: {status}\n"
            summary += f"   时长: {duration:.2f}s\n\n"

        best_exp = sorted_records[0]
        summary += f"✅ 最佳实验: {best_exp['experiment_id']} ({metric}: {best_exp['results'][metric]:.6f})\n"

        return summary

    except Exception as e:
        return f"❌ 比较实验失败: {str(e)}"


@tool
def GetExperimentResults(experiment_id: Optional[str] = None) -> str:
    """
    获取指定实验的训练与评估结果。

    参数:
        experiment_id: 实验 ID，如果为 None 则获取最近实验结果

    Returns:
        str: 训练与评估结果
    """
    try:
        tracker = ExperimentTracker()
        if experiment_id is None:
            recent = tracker.list_experiments(limit=1)
            if not recent:
                return "📋 暂无实验记录"
            experiment_id = recent[0]["experiment_id"]

        record = tracker.get_experiment(experiment_id)
        if not record:
            return f"❌ 实验不存在: {experiment_id}"

        training = record.get("training", {})
        evaluation = record.get("evaluation")

        summary = f"\n📊 实验结果 - 实验 ID: {experiment_id}\n"
        summary += "=" * 80 + "\n\n"
        summary += f"状态: {record.get('status', 'unknown')}\n"

        summary += "\n训练信息:\n"
        summary += f"  配置文件: {training.get('config_path', 'N/A')}\n"
        summary += f"  数据文件夹: {training.get('data_folder', 'N/A')}\n"
        summary += f"  日志路径: {training.get('train_log_path', 'N/A')}\n"
        summary += f"  输出目录: {training.get('output_folder', 'N/A')}\n"
        summary += f"  模型路径: {training.get('model_paths', []) or 'N/A'}\n"

        train_metrics = (training.get('metrics') or {})
        if train_metrics:
            summary += "  训练指标:\n"
            for key, value in train_metrics.items():
                summary += f"    {key}: {value}\n"

        if evaluation:
            summary += "\n评估信息:\n"
            summary += f"  评估状态: {evaluation.get('status', 'unknown')}\n"
            summary += f"  评估日志: {evaluation.get('evaluation_log_path', 'N/A')}\n"
            summary += f"  模型路径: {evaluation.get('model_path', 'N/A')}\n"
            eval_metrics = evaluation.get('results') or {}
            if eval_metrics:
                summary += "  评估指标:\n"
                for key, value in eval_metrics.items():
                    summary += f"    {key}: {value}\n"
        else:
            summary += "\n评估信息:\n  尚无评估结果\n"

        return summary

    except Exception as e:
        return f"❌ 获取实验结果失败: {str(e)}"


@tool
def ListExperiments(n: int = 10) -> str:
    """
    列出最近的实验结果（包含训练与评估概览）。

    参数:
        n: 显示最近的 n 个实验，默认 10

    Returns:
        str: 实验列表
    """
    try:
        tracker = ExperimentTracker()
        recent_exps = tracker.list_experiments(limit=n)
        if not recent_exps:
            return "📋 暂无实验记录"

        def _sort_key(exp):
            ts = exp.get("timestamp")
            if not ts:
                return datetime.min
            try:
                return datetime.fromisoformat(ts)
            except Exception:
                return ts

        recent_exps = sorted(recent_exps, key=_sort_key)

        summary = f"\n📋 最近的 {len(recent_exps)} 个实验:\n"
        summary += "=" * 80 + "\n\n"

        for i, exp in enumerate(recent_exps, 1):
            exp_id = exp.get("experiment_id")
            timestamp = exp.get("timestamp")
            time_str = "N/A"
            if timestamp:
                try:
                    time_str = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    time_str = timestamp

            evaluation = exp.get("evaluation")
            status = exp.get("status", "unknown")
            train_eer = "N/A"
            eval_eer = "N/A"
            min_dcf = "N/A"

            training = exp.get("training") or {}
            if training.get("metrics"):
                train_eer = training["metrics"].get("eer", "N/A")

            if evaluation and evaluation.get("results"):
                eval_eer = evaluation["results"].get("eer", "N/A")
                min_dcf = evaluation["results"].get("min_dcf", "N/A")

            summary += f"{i}. {exp_id}\n"
            summary += f"   时间: {time_str}\n"
            summary += f"   状态: {status}\n"
            if train_eer != "N/A":
                summary += f"   训练 EER: {train_eer:.4f}%\n"
            if eval_eer != "N/A":
                summary += f"   评估 EER: {eval_eer:.4f}%\n"
            if min_dcf != "N/A":
                summary += f"   minDCF: {min_dcf:.4f}\n"
            summary += "\n"

        return summary

    except Exception as e:
        return f"❌ 列出评估失败: {str(e)}"


__all__ = [
    'CompareExperiments',
    'GetExperimentResults',
    'ListExperiments',
]
