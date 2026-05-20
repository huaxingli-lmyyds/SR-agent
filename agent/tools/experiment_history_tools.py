"""
实验历史工具集合
提供实验记录查询与比较能力，供智能体调用
"""

from langchain_core.tools import tool
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from agent.utils import ExperimentTracker
from agent.utils.path_tool import (
    get_hpo_experiments_dir,
    get_data_processing_experiments_dir,
    get_manage_experiments_dir,
)


def _get_recent_record(
    tracker: ExperimentTracker,
    predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Optional[Dict[str, Any]]:
    recent = tracker.list_experiments(limit=50)
    if predicate is None:
        return recent[0] if recent else None

    for record in recent:
        if predicate(record):
            return record
    return None


def _format_time(value: Optional[str]) -> str:
    if not value:
        return "N/A"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return value


def _compare_numeric_records(records: List[Dict[str, Any]], metric_values: Dict[str, float], metric: str, title: str, better_is_higher: bool = False) -> str:
    if not metric_values:
        return f"⚠️  没有实验包含指标 '{metric}'"

    sorted_items = sorted(metric_values.items(), key=lambda item: item[1], reverse=better_is_higher)

    summary = f"\n📊 {title} - 基于指标: {metric}\n"
    summary += "=" * 80 + "\n\n"
    for i, (exp_id, value) in enumerate(sorted_items, 1):
        record = next((item for item in records if item.get("experiment_id") == exp_id), None)
        if not record:
            continue
        summary += f"{i}. 实验 {exp_id}\n"
        summary += f"   {metric}: {value:.6f}\n"
        summary += f"   状态: {record.get('status', 'unknown')}\n"
        summary += f"   时长: {record.get('duration_seconds', 0):.2f}s\n\n"

    best_exp_id, best_value = sorted_items[0]
    summary += f"✅ 最佳实验: {best_exp_id} ({metric}: {best_value:.6f})\n"
    return summary


@tool
def CompareHPOExperiments(experiment_ids: Optional[List[str]] = None,
                           metric: str = "eer") -> str:
    """
    比较超参数智能体的实验结果（训练/评估统一比较）。

    参数:
        experiment_ids: 实验 ID 列表，如果为 None 则比较最近的 5 个实验
        metric: 比较的指标，默认 'eer'，可选 'min_dcf', 'accuracy', 'error_rate', 'loss'

    Returns:
        str: 比较结果
    """
    try:
        tracker = ExperimentTracker(get_hpo_experiments_dir())
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

        valid_records = [
            r for r in records
            if r.get('results') and r['results'].get(metric) is not None
        ]
        if not valid_records:
            return f"⚠️  没有实验包含指标 '{metric}'"

        reverse = metric in ['accuracy', 'precision', 'recall', 'f1']
        metric_values = {
            record['experiment_id']: record['results'][metric]
            for record in valid_records
            if isinstance(record['results'][metric], (int, float))
        }
        return _compare_numeric_records(valid_records, metric_values, metric, "HPO 实验比较", better_is_higher=reverse)

    except Exception as e:
        return f"❌ 比较实验失败: {str(e)}"


@tool
def GetHPOExperimentResults(experiment_id: Optional[str] = None) -> str:
    """
    获取指定实验的训练与评估结果。

    参数:
        experiment_id: 实验 ID，如果为 None 则获取最近实验结果

    Returns:
        str: 训练与评估结果
    """
    try:
        tracker = ExperimentTracker(get_hpo_experiments_dir())
        if experiment_id is None:
            recent = _get_recent_record(tracker)
            if not recent:
                return "📋 暂无实验记录"
            experiment_id = recent["experiment_id"]

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
        summary += f"  模型路径: {', '.join(training.get('model_paths', [])) or 'N/A'}\n"

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
def ListHPOExperiments(n: int = 10) -> str:
    """
    列出最近的实验结果（包含训练与评估概览）。

    参数:
        n: 显示最近的 n 个实验，默认 10

    Returns:
        str: 实验列表
    """
    try:
        tracker = ExperimentTracker(get_hpo_experiments_dir())
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


@tool
def CompareDataProcessingExperiments(experiment_ids: Optional[List[str]] = None,
                                     metric: str = "train") -> str:
    """
    比较数据处理智能体的实验结果。

    参数:
        experiment_ids: 实验 ID 列表，如果为 None 则比较最近的 5 个实验
        metric: 比较的指标，默认 'train'，可选 'train', 'dev', 'test', 'enrol'

    Returns:
        str: 比较结果
    """
    try:
        tracker = ExperimentTracker(get_data_processing_experiments_dir())
        if not experiment_ids:
            recent = tracker.list_experiments(limit=5)
            if not recent:
                return "📋 暂无数据处理实验记录"
            experiment_ids = [exp["experiment_id"] for exp in recent]

        records = [tracker.get_experiment(exp_id) for exp_id in experiment_ids]
        records = [record for record in records if record]
        if not records:
            return "❌ 没有找到有效的数据处理实验记录"

        metric_values: Dict[str, float] = {}
        for record in records:
            data_processing = record.get("data_processing") or {}
            stats = data_processing.get("stats") or {}
            value = stats.get(metric)
            if isinstance(value, (int, float)):
                metric_values[record["experiment_id"]] = float(value)

        return _compare_numeric_records(records, metric_values, metric, "数据处理实验比较")

    except Exception as e:
        return f"❌ 比较数据处理实验失败: {str(e)}"


@tool
def GetDataProcessingExperimentResults(experiment_id: Optional[str] = None) -> str:
    """
    获取指定数据处理实验的记录摘要。

    参数:
        experiment_id: 实验 ID，如果为 None 则获取最近实验结果

    Returns:
        str: 数据处理结果摘要
    """
    try:
        tracker = ExperimentTracker(get_data_processing_experiments_dir())
        if experiment_id is None:
            recent = _get_recent_record(tracker)
            if not recent:
                return "📋 暂无数据处理实验记录"
            experiment_id = recent["experiment_id"]

        record = tracker.get_experiment(experiment_id)
        if not record:
            return f"❌ 实验不存在: {experiment_id}"

        data_processing = record.get("data_processing") or {}
        stats = data_processing.get("stats") or {}
        summary = data_processing.get("summary") or {}

        text = f"\n🧹 数据处理实验结果 - 实验 ID: {experiment_id}\n"
        text += "=" * 80 + "\n\n"
        text += f"状态: {record.get('status', 'unknown')}\n"
        text += f"时间: {_format_time(record.get('timestamp'))}\n"
        text += f"数据目录: {data_processing.get('data_folder', 'N/A')}\n"
        text += f"保存目录: {data_processing.get('save_folder', data_processing.get('output_folder', 'N/A'))}\n"

        if stats:
            text += "\nCSV 统计:\n"
            for key, value in stats.items():
                text += f"  {key}: {value}\n"

        if summary:
            text += "\n摘要:\n"
            for key, value in summary.items():
                if key != "best_config":
                    text += f"  {key}: {value}\n"

        return text

    except Exception as e:
        return f"❌ 获取数据处理实验结果失败: {str(e)}"


@tool
def ListDataProcessingExperiments(n: int = 10) -> str:
    """
    列出最近的数据处理实验结果。

    参数:
        n: 显示最近的 n 个实验，默认 10

    Returns:
        str: 实验列表
    """
    try:
        tracker = ExperimentTracker(get_data_processing_experiments_dir())
        recent_exps = tracker.list_experiments(limit=n)
        if not recent_exps:
            return "📋 暂无数据处理实验记录"

        summary = f"\n📋 最近的 {len(recent_exps)} 个数据处理实验:\n"
        summary += "=" * 80 + "\n\n"

        for i, exp in enumerate(recent_exps, 1):
            dp = exp.get("data_processing") or {}
            stats = dp.get("stats") or {}
            summary += f"{i}. {exp.get('experiment_id')}\n"
            summary += f"   时间: {_format_time(exp.get('timestamp'))}\n"
            summary += f"   状态: {exp.get('status', 'unknown')}\n"
            if dp.get("data_folder"):
                summary += f"   数据目录: {dp.get('data_folder')}\n"
            if stats:
                summary += f"   train/dev/test/enrol: {stats.get('train', 'N/A')}/{stats.get('dev', 'N/A')}/{stats.get('test', 'N/A')}/{stats.get('enrol', 'N/A')}\n"
            summary += "\n"

        return summary

    except Exception as e:
        return f"❌ 列出数据处理实验失败: {str(e)}"


@tool
def CompareOrchestrationExperiments(experiment_ids: Optional[List[str]] = None,
                                    metric: str = "rounds") -> str:
    """
    比较统筹智能体的实验结果。

    参数:
        experiment_ids: 实验 ID 列表，如果为 None 则比较最近的 5 个实验
        metric: 比较的指标，默认 'rounds'，可选 'rounds', 'messages'

    Returns:
        str: 比较结果
    """
    try:
        tracker = ExperimentTracker(get_manage_experiments_dir())
        if not experiment_ids:
            recent = tracker.list_experiments(limit=5)
            if not recent:
                return "📋 暂无统筹实验记录"
            experiment_ids = [exp["experiment_id"] for exp in recent]

        records = [tracker.get_experiment(exp_id) for exp_id in experiment_ids]
        records = [record for record in records if record]
        if not records:
            return "❌ 没有找到有效的统筹实验记录"

        metric_values: Dict[str, float] = {}
        for record in records:
            orchestration = record.get("orchestration") or {}
            if metric == "messages":
                value = len(record.get("a2a_messages") or [])
            else:
                value = orchestration.get(metric)
            if isinstance(value, (int, float)):
                metric_values[record["experiment_id"]] = float(value)

        return _compare_numeric_records(records, metric_values, metric, "统筹实验比较")

    except Exception as e:
        return f"❌ 比较统筹实验失败: {str(e)}"


@tool
def GetOrchestrationExperimentResults(experiment_id: Optional[str] = None) -> str:
    """
    获取指定统筹实验的记录摘要。

    参数:
        experiment_id: 实验 ID，如果为 None 则获取最近实验结果

    Returns:
        str: 统筹结果摘要
    """
    try:
        tracker = ExperimentTracker(get_manage_experiments_dir())
        if experiment_id is None:
            recent = _get_recent_record(tracker)
            if not recent:
                return "📋 暂无统筹实验记录"
            experiment_id = recent["experiment_id"]

        record = tracker.get_experiment(experiment_id)
        if not record:
            return f"❌ 实验不存在: {experiment_id}"

        orchestration = record.get("orchestration") or {}
        linked = record.get("linked_experiments") or {}

        text = f"\n🧩 统筹实验结果 - 实验 ID: {experiment_id}\n"
        text += "=" * 80 + "\n\n"
        text += f"状态: {record.get('status', 'unknown')}\n"
        text += f"时间: {_format_time(record.get('timestamp'))}\n"
        text += f"轮数: {orchestration.get('rounds', 0)}\n"
        text += f"数据处理实验: {linked.get('data_processing', 'N/A')}\n"
        text += f"HPO 实验: {linked.get('hpo', 'N/A')}\n"

        messages = record.get("a2a_messages") or []
        if messages:
            text += f"\nA2A 消息数: {len(messages)}\n"

        history = record.get("data_processing_summary_history") or []
        if history:
            text += f"数据处理摘要次数: {len(history)}\n"

        feedback = record.get("hpo_feedback_history") or []
        if feedback:
            text += f"HPO 反馈次数: {len(feedback)}\n"

        return text

    except Exception as e:
        return f"❌ 获取统筹实验结果失败: {str(e)}"


@tool
def ListOrchestrationExperiments(n: int = 10) -> str:
    """
    列出最近的统筹实验结果。

    参数:
        n: 显示最近的 n 个实验，默认 10

    Returns:
        str: 实验列表
    """
    try:
        tracker = ExperimentTracker(get_manage_experiments_dir())
        recent_exps = tracker.list_experiments(limit=n)
        if not recent_exps:
            return "📋 暂无统筹实验记录"

        summary = f"\n📋 最近的 {len(recent_exps)} 个统筹实验:\n"
        summary += "=" * 80 + "\n\n"

        for i, exp in enumerate(recent_exps, 1):
            orchestration = exp.get("orchestration") or {}
            linked = exp.get("linked_experiments") or {}
            summary += f"{i}. {exp.get('experiment_id')}\n"
            summary += f"   时间: {_format_time(exp.get('timestamp'))}\n"
            summary += f"   状态: {exp.get('status', 'unknown')}\n"
            summary += f"   轮数: {orchestration.get('rounds', 0)}\n"
            if linked:
                summary += f"   关联实验: {linked}\n"
            summary += "\n"

        return summary

    except Exception as e:
        return f"❌ 列出统筹实验失败: {str(e)}"


__all__ = [
    'CompareHPOExperiments',
    'GetHPOExperimentResults',
    'ListHPOExperiments',
    'CompareDataProcessingExperiments',
    'GetDataProcessingExperimentResults',
    'ListDataProcessingExperiments',
    'CompareOrchestrationExperiments',
    'GetOrchestrationExperimentResults',
    'ListOrchestrationExperiments',
    'CompareExperiments',
    'GetExperimentResults',
    'ListExperiments',
]


# 兼容旧名称，默认映射到 HPO 记录
CompareExperiments = CompareHPOExperiments
GetExperimentResults = GetHPOExperimentResults
ListExperiments = ListHPOExperiments
