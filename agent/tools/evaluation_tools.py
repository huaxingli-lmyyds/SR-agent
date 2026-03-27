"""
评估工具集合
提供模型评估、指标计算、结果分析等功能
使用 LangChain 工具接口，基于 utils 模块实现
"""

from langchain_core.tools import tool
from typing import Optional, Dict, List
from pathlib import Path
import subprocess
import re
from datetime import datetime
import json

# 导入 utils 模块
from agent.utils import (
    get_config_file,
    get_experiments_dir,
    ensure_dir,
    get_experiment_log_path,
    ExperimentLogger,
    extract_scores_data,
    compute_metrics_from_scores,
    get_results_dir,
    get_train_script
)

# 全局路径
VERIFICATION_CONFIG = str(get_config_file("verification_ecapa.yaml"))
VERIFICATION_SCRIPT = str(get_train_script("voxceleb/verification_cosine.py"))


@tool
def RunEvaluation(model_path: Optional[str] = None,
                  verification_config: Optional[str] = None,
                  data_folder: Optional[str] = None) -> str:
    """
    运行 ECAPA-TDNN 模型的评估脚本，计算 EER 和 minDCF 等指标。
    
    参数:
        model_path: 模型路径，如果为 None 则使用配置文件中的预训练模型
        verification_config: 评估配置文件路径，如果为 None 则使用默认配置
        data_folder: 数据文件夹路径，如果为 None 则使用配置文件中的设置
    
    Returns:
        str: 评估结果或错误信息
    """
    try:
        # 确定配置文件路径
        ver_config = verification_config if verification_config else VERIFICATION_CONFIG
        
        
        # 使用 ConfigParser 读取配置（支持 YAML 引用解析）
        from agent.utils import ConfigParser
        parser = ConfigParser(ver_config)
        config_data = parser.load_config(resolve_references=True)
        
        
        # 确定数据文件夹
        if data_folder:
            df = data_folder
        elif config_data.get('data_folder'):
            df = str(config_data.get('data_folder'))
        else:
            df = "../datasets/voxceleb1"
        
        # 确定输出文件夹
        output_folder = config_data.get('output_folder', 'results/verification')
        
        # 记录开始时间
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        eval_id = f"eval_{timestamp}"
        
        # 准备评估目录
        eval_dir = ensure_dir(get_results_dir() / eval_id)
        
        print(f"\n{'='*80}")
        print(f"🎯 开始评估 - 评估 ID: {eval_id}")
        print(f"{'='*80}")
        print(f"配置文件: {ver_config}")
        print(f"数据文件夹: {df}")
        print(f"模型路径: {model_path if model_path else '使用配置中的预训练模型'}")
        print(f"输出文件夹: {output_folder}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # 构建命令
        cmd = ["python", VERIFICATION_SCRIPT, ver_config]
        if data_folder:
            cmd.append(f"--data_folder={df}")
        
        # 运行评估
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent.parent)
        )
        
        # 记录结束时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 保存评估日志
        logger = ExperimentLogger(eval_id, "evaluation")
        logger.log_training(
            config_file=ver_config,
            data_folder=df,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            stdout=result.stdout,
            stderr=result.stderr
        )
        
        # 解析输出结果
        eer = None
        min_dcf = None
        
        # 从输出中提取 EER
        eer_match = re.search(r'EER\(%\)=([\d.]+)', result.stdout)
        if eer_match:
            eer = float(eer_match.group(1))
        
        # 从输出中提取 minDCF
        min_dcf_match = re.search(r'minDCF=([\d.]+)', result.stdout)
        if min_dcf_match:
            min_dcf = float(min_dcf_match.group(1))
        
        # 尝试从 scores.txt 中读取结果
        scores_file = Path(output_folder) / "scores.txt"
        scores_data = None
        if scores_file.exists():
            scores_data = extract_scores_data(str(scores_file))
        
        if result.returncode == 0:
            # 评估成功
            eval_record = {
                "eval_id": eval_id,
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "status": "success",
                "config_file": ver_config,
                "data_folder": df,
                "model_path": model_path,
                "output_folder": output_folder,
                "log_file": str(logger.log_path),
                "results": {
                    "eer": eer,
                    "min_dcf": min_dcf,
                }
            }
            
            # 如果有 scores 文件，计算更多指标
            if scores_data and not scores_data.get("error"):
                metrics = compute_metrics_from_scores(scores_data['positive_scores'], 
                                                     scores_data['negative_scores'])
                eval_record["results"].update(metrics)
            
            # 保存评估记录
            record_path = eval_dir / "evaluation_record.json"
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(eval_record, f, indent=2, ensure_ascii=False)
            
            summary = f"""✅ 评估完成！
评估 ID: {eval_id}
评估时长: {duration:.2f} 秒

📊 性能指标:
  - EER: {eer if eer else 'N/A'}%
  - minDCF: {min_dcf if min_dcf else 'N/A'}"""
            
            if scores_data and not scores_data.get("error"):
                summary += f"""
  - 准确率: {eval_record['results'].get('accuracy', 'N/A')}
  - 精确率: {eval_record['results'].get('precision', 'N/A')}
  - 召回率: {eval_record['results'].get('recall', 'N/A')}
  - F1分数: {eval_record['results'].get('f1', 'N/A')}"""
            
            summary += f"""

📁 文件位置:
  - 评估目录: {eval_dir}
  - 评估日志: {logger.log_path}
  - 评估记录: {record_path}
  - 分数文件: {scores_file if scores_file.exists() else 'N/A'}"""
            
            if eer is not None:
                summary += f"\n\n💡 EER 越低越好，最佳目标是 < 5%"
            
            return summary
        else:
            # 评估失败
            eval_record = {
                "eval_id": eval_id,
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "status": "failed",
                "config_file": ver_config,
                "data_folder": df,
                "model_path": model_path,
                "log_file": str(logger.log_path),
                "error": result.stderr[-1000:]
            }
            
            # 保存失败记录
            record_path = eval_dir / "evaluation_record.json"
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(eval_record, f, indent=2, ensure_ascii=False)
            
            return f"""❌ 评估失败 (评估 ID: {eval_id})
评估时长: {duration:.2f} 秒

错误信息:
{result.stderr[-500:]}

📁 文件位置:
  - 评估目录: {eval_dir}
  - 评估日志: {logger.log_path}
"""
    except Exception as e:
        import traceback
        return f"❌ 运行评估失败: {str(e)}\n{traceback.format_exc()}"


@tool
def GetEvaluationResults(eval_id: Optional[str] = None) -> str:
    """
    获取指定评估的结果。
    
    参数:
        eval_id: 评估 ID，如果为 None 则获取最近的评估结果
    
    Returns:
        str: 评估结果
    """
    try:
        results_dir = get_results_dir()
        
        if eval_id:
            target_dir = results_dir / eval_id
            if not target_dir.exists():
                return f"❌ 评估不存在: {eval_id}"
        else:
            # 获取最近的评估
            evals = sorted(results_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if not evals:
                return "📋 暂无评估记录"
            target_dir = evals[0]
            eval_id = target_dir.name
        
        # 读取评估记录
        record_path = target_dir / "evaluation_record.json"
        if not record_path.exists():
            return f"⚠️  评估记录文件不存在: {record_path}"
        
        with open(record_path, 'r', encoding='utf-8') as f:
            record = json.load(f)
        
        status = record.get('status', 'unknown')
        summary = f"\n📊 评估结果 - ID: {eval_id}\n"
        summary += "=" * 80 + "\n\n"
        summary += f"状态: {status}\n"
        
        if status != "success":
            summary += "⚠️  评估未成功完成\n"
            return summary
        
        summary += "✅ 评估成功完成\n\n"
        
        if record.get('results'):
            results = record['results']
            summary += "性能指标:\n"
            if results.get('eer'):
                summary += f"  EER: {results['eer']:.4f}%\n"
            if results.get('min_dcf'):
                summary += f"  minDCF: {results['min_dcf']:.4f}\n"
            if results.get('accuracy'):
                summary += f"  准确率: {results['accuracy']:.4f}\n"
            if results.get('precision'):
                summary += f"  精确率: {results['precision']:.4f}\n"
            if results.get('recall'):
                summary += f"  召回率: {results['recall']:.4f}\n"
            if results.get('f1'):
                summary += f"  F1分数: {results['f1']:.4f}\n"
        
        # 评估时长
        duration = record.get('duration_seconds', 0)
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        summary += f"\n评估时长: {hours}h {minutes}m {seconds}s\n"
        
        # 配置信息
        summary += f"\n配置信息:\n"
        summary += f"  配置文件: {record.get('config_file', 'N/A')}\n"
        summary += f"  数据文件夹: {record.get('data_folder', 'N/A')}\n"
        summary += f"  模型路径: {record.get('model_path', 'N/A')}\n"
        
        return summary
    
    except Exception as e:
        return f"❌ 获取评估结果失败: {str(e)}"


@tool
def CompareEvaluations(eval_ids: Optional[List[str]] = None,
                       metric: str = "eer") -> str:
    """
    比较多个评估的结果。
    
    参数:
        eval_ids: 评估 ID 列表，如果为 None 则比较最近的 5 个评估
        metric: 比较的指标，默认 'eer'，可选 'min_dcf', 'accuracy', 'precision', 'recall', 'f1'
    
    Returns:
        str: 比较结果
    """
    try:
        results_dir = get_results_dir()
        
        if not eval_ids:
            # 获取最近的 5 个评估
            evals = sorted(results_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
            if not evals:
                return "📋 暂无评估记录"
            eval_ids = [eval_dir.name for eval_dir in evals]
        
        # 读取所有评估记录
        records = []
        for eval_id in eval_ids:
            record_path = results_dir / eval_id / "evaluation_record.json"
            if record_path.exists():
                with open(record_path, 'r', encoding='utf-8') as f:
                    records.append(json.load(f))
        
        if not records:
            return "❌ 没有找到有效的评估记录"
        
        summary = f"\n📊 评估比较 - 基于指标: {metric}\n"
        summary += "=" * 80 + "\n\n"
        
        # 按指标排序
        valid_records = [r for r in records if r.get('results') and r['results'].get(metric) is not None]
        
        if not valid_records:
            return f"⚠️  没有评估包含指标 '{metric}'"
        
        # 根据指标类型决定排序方向（越小越好：eer, min_dcf；越大越好：accuracy, precision, recall, f1）
        reverse = metric in ['accuracy', 'precision', 'recall', 'f1']
        sorted_records = sorted(valid_records,
                              key=lambda x: x['results'][metric],
                              reverse=reverse)
        
        for i, record in enumerate(sorted_records, 1):
            eval_id = record['eval_id']
            value = record['results'][metric]
            status = record.get('status', 'unknown')
            duration = record.get('duration_seconds', 0)
            
            summary += f"{i}. 评估 {eval_id}\n"
            summary += f"   {metric}: {value:.6f}\n"
            summary += f"   状态: {status}\n"
            summary += f"   时长: {duration:.2f}s\n\n"
        
        # 找出最佳评估
        best_eval = sorted_records[0]
        summary += f"✅ 最佳评估: {best_eval['eval_id']} ({metric}: {best_eval['results'][metric]:.6f})\n"
        
        return summary
    
    except Exception as e:
        return f"❌ 比较评估失败: {str(e)}"


@tool
def ListEvaluations(n: int = 10) -> str:
    """
    列出最近的评估记录。
    
    参数:
        n: 显示最近的 n 个评估，默认 10
    
    Returns:
        str: 评估列表
    """
    try:
        results_dir = get_results_dir()
        
        if not results_dir.exists():
            return "📋 暂无评估记录"
        
        # 获取所有评估目录
        eval_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        
        if not eval_dirs:
            return "📋 暂无评估记录"
        
        # 按修改时间排序（最新的在前）
        eval_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # 只显示最近的 n 个
        recent_evals = eval_dirs[:n]
        
        summary = f"\n📋 最近的 {len(recent_evals)} 个评估:\n"
        summary += "=" * 80 + "\n\n"
        
        for i, eval_dir in enumerate(recent_evals, 1):
            eval_id = eval_dir.name
            mtime = eval_dir.stat().st_mtime
            time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # 尝试读取评估记录
            record_path = eval_dir / "evaluation_record.json"
            status = "unknown"
            eer = "N/A"
            min_dcf = "N/A"
            
            if record_path.exists():
                try:
                    with open(record_path, 'r', encoding='utf-8') as f:
                        record = json.load(f)
                    status = record.get('status', 'unknown')
                    if record.get('results'):
                        eer = record['results'].get('eer', 'N/A')
                        min_dcf = record['results'].get('min_dcf', 'N/A')
                except Exception:
                    pass
            
            summary += f"{i}. {eval_id}\n"
            summary += f"   时间: {time_str}\n"
            summary += f"   状态: {status}\n"
            if eer != "N/A":
                summary += f"   EER: {eer:.4f}%\n"
            if min_dcf != "N/A":
                summary += f"   minDCF: {min_dcf:.4f}\n"
            summary += f"   路径: {eval_dir}\n\n"
        
        return summary
    
    except Exception as e:
        return f"❌ 列出评估失败: {str(e)}"


# 导出所有工具
__all__ = [
    'RunEvaluation',
    'GetEvaluationResults',
    'CompareEvaluations',
    'ListEvaluations',
]