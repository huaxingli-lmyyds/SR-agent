"""
评估工具集合
提供模型评估、指标计算、结果分析等功能
使用 LangChain 工具接口，基于 utils 模块实现
"""

from langchain_core.tools import tool
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import re

# 导入 utils 模块
from agent.utils import (
    get_config_file,
    get_experiments_dir,
    ensure_dir,
    get_project_root,
    ExperimentTracker,
    extract_scores_data,
    compute_metrics_from_scores
)

# 全局路径
VERIFICATION_CONFIG = str(get_config_file("verification_ecapa.yaml"))


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return get_project_root() / path


def _parse_evaluation_log(log_path: Path) -> dict:
    metrics = {"eer": None, "min_dcf": None}
    if not log_path.exists():
        return metrics

    content = log_path.read_text(encoding="utf-8", errors="ignore")

    eer_match = re.search(r"EER\(%\)\s*=\s*([\d.e+-]+)", content, re.IGNORECASE)
    if eer_match:
        try:
            metrics["eer"] = float(eer_match.group(1))
        except ValueError:
            pass

    min_dcf_match = re.search(r"minDCF\s*=\s*([\d.e+-]+)", content, re.IGNORECASE)
    if min_dcf_match:
        try:
            metrics["min_dcf"] = float(min_dcf_match.group(1))
        except ValueError:
            pass

    return metrics


@tool
def RunEvaluation(model_path: Optional[str] = None,
                  verification_config: Optional[str] = None,
                  data_folder: Optional[str] = None,
                  experiment_id: Optional[str] = None) -> str:
    """
    运行 ECAPA-TDNN 模型的评估流程，计算 EER 和 minDCF 等指标。
    
    参数:
        model_path: 模型路径，如果为 None 则使用配置文件中的预训练模型
        verification_config: 评估配置文件路径，如果为 None 则使用默认配置
        data_folder: 数据文件夹路径，如果为 None 则使用配置文件中的设置
        experiment_id: 实验 ID，若未提供则默认使用最近实验
    
    Returns:
        str: 评估结果或错误信息
    """
    try:
        ver_config = verification_config if verification_config else VERIFICATION_CONFIG

        tracker = ExperimentTracker()
        if experiment_id is None:
            recent = tracker.list_experiments(limit=1)
            if not recent:
                return "📋 暂无实验记录"
            experiment_id = recent[0]["experiment_id"]

        record = tracker.get_experiment(experiment_id)
        if not record:
            return f"❌ 实验不存在: {experiment_id}"

        exp_dir = ensure_dir(get_experiments_dir() / experiment_id)
        eval_output_folder = exp_dir / "evaluation"

        training_info = record.get("training", {})
        if data_folder is None:
            data_folder = training_info.get("data_folder")

        if not data_folder or data_folder == "!PLACEHOLDER":
            data_folder = "../datasets/voxceleb1"

        # 记录开始时间
        start_time = datetime.now()

        # 运行评估（非控制台调用）
        from agent.utils import runner
        eval_result = runner.run_evaluation(
            config_path=ver_config,
            model_path=model_path,
            data_folder=data_folder,
            overrides=[f"output_folder: {eval_output_folder}"],
        )

        if eval_result.get("status") == "failed":
            return f"❌ 运行评估失败: {eval_result.get('error')}"

        eer = eval_result.get("eer")
        min_dcf = eval_result.get("min_dcf")
        output_folder = eval_result.get("output_folder")
        scores_path = eval_result.get("scores_path")

        output_folder_path = eval_output_folder
        eval_log_path = output_folder_path / "log.txt" if output_folder_path else None
        log_metrics = _parse_evaluation_log(eval_log_path) if eval_log_path else {}
        if log_metrics.get("eer") is not None:
            eer = log_metrics["eer"]
        if log_metrics.get("min_dcf") is not None:
            min_dcf = log_metrics["min_dcf"]
        
        # 记录结束时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 尝试从 scores.txt 中读取结果
        scores_data = None
        scores_file = Path(scores_path) if scores_path else None
        if scores_file and scores_file.exists():
            scores_data = extract_scores_data(str(scores_file))

        results_payload = {
            "eer": eer,
            "min_dcf": min_dcf,
        }

        if scores_data and not scores_data.get("error"):
            metrics = compute_metrics_from_scores(scores_data['positive_scores'],
                                                 scores_data['negative_scores'])
            results_payload.update(metrics)

        tracker.update_experiment(
            experiment_id=experiment_id,
            evaluation={
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "status": "success",
                "evaluation_log_path": str(eval_log_path) if eval_log_path else None,
                "output_folder": str(output_folder_path) if output_folder_path else output_folder,
                "model_path": model_path,
                "results": results_payload
            },
            results=results_payload
        )

        summary = f"""✅ 评估完成！
实验 ID: {experiment_id}
评估时长: {duration:.2f} 秒

📊 性能指标:
  - EER: {eer if eer is not None else 'N/A'}%
  - minDCF: {min_dcf if min_dcf is not None else 'N/A'}"""

        if scores_data and not scores_data.get("error"):
            summary += f"""
  - 准确率: {results_payload.get('accuracy', 'N/A')}
  - 精确率: {results_payload.get('precision', 'N/A')}
  - 召回率: {results_payload.get('recall', 'N/A')}
  - F1分数: {results_payload.get('f1', 'N/A')}"""

        summary += f"""

📁 文件位置:
    - 评估日志: {eval_log_path if eval_log_path else 'N/A'}
    - 分数文件: {scores_file if scores_file and scores_file.exists() else 'N/A'}"""

        if eer is not None:
            summary += f"\n\n💡 EER 越低越好，最佳目标是 < 5%"

        return summary
    except Exception as e:
        import traceback
        return f"❌ 运行评估失败: {str(e)}\n{traceback.format_exc()}"


# 导出所有工具
__all__ = [
    'RunEvaluation',
]