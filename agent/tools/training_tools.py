"""
训练管理工具集合
提供模型训练、训练监控、训练控制等功能
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
    create_experiment,
    list_experiments,
    find_best_experiment,
    ExperimentLogger
)

# 全局路径
CONFIG_PATH = str(get_config_file("train_ecapa_tdnn.yaml"))
TRAIN_SCRIPT = str(Path(__file__).parent.parent.parent / "recipes" / "voxceleb" / "train_speaker_embeddings.py")


@tool
def TrainModel(config_path: Optional[str] = None, 
               data_folder: Optional[str] = None) -> str:
    """
    运行 ECAPA-TDNN 模型的训练脚本。
    训练完成后会自动保存实验记录，包括配置、训练日志和结果。
    
    参数:
        config_path: 配置文件路径，如果为 None 则使用默认配置
        data_folder: 数据文件夹路径，如果为 None 则使用配置文件中的设置
    
    Returns:
        str: 训练输出或错误信息
    """
    try:
        import shutil
        
        # 确定配置文件路径
        config_path_str = config_path if config_path else CONFIG_PATH
        
        # 使用 ConfigParser 读取配置（支持 YAML 引用解析）
        from agent.utils import ConfigParser
        parser = ConfigParser(config_path_str)
        config_data = parser.load_config(resolve_references=True)
        
        # 确定数据文件夹
        if data_folder:
            df = data_folder
        elif config_data.get('data_folder'):
            df = str(config_data.get('data_folder'))
        else:
            df = "../datasets/voxceleb1"
        
        # 记录开始时间
        start_time = datetime.now()
        experiment_id = start_time.strftime("%Y%m%d_%H%M%S")
        
        # 创建实验记录
        exp_record = create_experiment(
            experiment_id=experiment_id,
            config_file=config_path_str,
            data_folder=df,
            timestamp=start_time
        )
        
        # 准备实验目录
        exp_dir = ensure_dir(get_experiments_dir() / experiment_id)
        
        # 备份配置
        config_backup = exp_dir / "config.yaml"
        shutil.copy2(config_path_str, config_backup)
        
        print(f"\n{'='*80}")
        print(f"🚀 开始训练 - 实验 ID: {experiment_id}")
        print(f"{'='*80}")
        print(f"配置文件: {config_path_str}")
        print(f"数据文件夹: {df}")
        print(f"实验目录: {exp_dir}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # 运行训练
        result = subprocess.run(
            ["python", TRAIN_SCRIPT, config_path_str, f"--data_folder={df}"],
            capture_output=True, text=True, 
            cwd=str(Path(__file__).parent.parent.parent)
        )
        
        # 记录结束时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 使用实验记录器保存日志
        logger = ExperimentLogger(experiment_id)
        logger.log_training(
            config_file=config_path_str,
            data_folder=df,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            stdout=result.stdout,
            stderr=result.stderr
        )
        
        if result.returncode == 0:
            # 训练成功，提取训练结果
            training_output = result.stdout
            
            # 尝试从输出中提取性能指标
            error_rate = None
            eer = None
            accuracy = None
            loss = None
            
            # 查找常见的性能指标
            error_match = re.search(r'error rate[:\s]+([\d.]+)', training_output.lower())
            if error_match:
                error_rate = float(error_match.group(1))
            
            eer_match = re.search(r'EER[:\s]+([\d.]+)', training_output)
            if eer_match:
                eer = float(eer_match.group(1))
            
            acc_match = re.search(r'accuracy[:\s]+([\d.]+)', training_output.lower())
            if acc_match:
                accuracy = float(acc_match.group(1))
            
            loss_match = re.search(r'loss[:\s]+([\d.]+)', training_output.lower())
            if loss_match:
                loss = float(loss_match.group(1))
            
            # 更新实验记录
            exp_record.update({
                'duration_seconds': duration,
                'status': 'success',
                'config_backup': str(config_backup),
                'log_file': str(logger.log_path),
                'results': {
                    'error_rate': error_rate,
                    'eer': eer,
                    'accuracy': accuracy,
                    'loss': loss,
                    'final_output': training_output[-1000:]
                }
            })
            
            # 保存实验记录
            record_path = exp_dir / "experiment_record.json"
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(exp_record, f, indent=2, ensure_ascii=False)
            
            return f"""✅ 训练完成！
实验 ID: {experiment_id}
训练时长: {duration:.2f} 秒

📊 性能指标:
  - 错误率: {error_rate if error_rate else 'N/A'}
  - 等错误率 (EER): {eer if eer else 'N/A'}
  - 准确率: {accuracy if accuracy else 'N/A'}
  - 损失: {loss if loss else 'N/A'}

📁 文件位置:
  - 实验目录: {exp_dir}
  - 配置备份: {config_backup}
  - 训练日志: {logger.log_path}
  - 实验记录: {record_path}

最后输出:
{training_output[-200:]}"""
        else:
            # 训练失败
            exp_record.update({
                'duration_seconds': duration,
                'status': 'failed',
                'config_backup': str(config_backup),
                'log_file': str(logger.log_path),
                'error': result.stderr[-1000:]
            })
            
            # 保存失败记录
            record_path = exp_dir / "experiment_record.json"
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(exp_record, f, indent=2, ensure_ascii=False)
            
            return f"""❌ 训练失败 (实验 ID: {experiment_id})
训练时长: {duration:.2f} 秒

错误信息:
{result.stderr[-500:]}

📁 文件位置:
  - 实验目录: {exp_dir}
  - 训练日志: {logger.log_path}
"""
    except Exception as e:
        import traceback
        return f"❌ 运行训练失败: {str(e)}\n{traceback.format_exc()}"


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
        
        # 读取实验记录
        record_path = target_dir / "experiment_record.json"
        if not record_path.exists():
            return f"⚠️  实验记录文件不存在: {record_path}"
        
        with open(record_path, 'r', encoding='utf-8') as f:
            record = json.load(f)
        
        status = record.get('status', 'unknown')
        summary = f"\n📊 模型评估 - 实验 ID: {experiment_id}\n"
        summary += "=" * 80 + "\n\n"
        summary += f"状态: {status}\n"
        
        if status != "success":
            summary += "⚠️  实验未成功完成，无法评估\n"
            return summary
        
        summary += "✅ 训练成功完成\n\n"
        
        if record.get('results'):
            results = record['results']
            summary += "性能指标:\n"
            if results.get('error_rate'):
                summary += f"  错误率: {results['error_rate']:.4f}\n"
            if results.get('eer'):
                summary += f"  EER: {results['eer']:.4f}\n"
            if results.get('accuracy'):
                summary += f"  准确率: {results['accuracy']:.4f}\n"
            if results.get('loss'):
                summary += f"  损失: {results['loss']:.4f}\n"
        
        # 训练时长
        duration = record.get('duration_seconds', 0)
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        summary += f"\n训练时长: {hours}h {minutes}m {seconds}s\n"
        
        # 尝试查找模型文件
        model_files = list(target_dir.glob("*.ckpt")) + list(target_dir.glob("*.pt"))
        if model_files:
            summary += f"\n📁 模型文件:\n"
            for model_file in model_files:
                size_mb = model_file.stat().st_size / (1024 * 1024)
                summary += f"  - {model_file.name} ({size_mb:.2f} MB)\n"
        
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


@tool
def CompareExperiments(experiment_ids: Optional[List[str]] = None, 
                     metric: str = "eer") -> str:
    """
    比较多个实验的结果。
    
    参数:
        experiment_ids: 实验 ID 列表，如果为 None 则比较最近的 5 个实验
        metric: 比较的指标，默认 'eer'，可选 'accuracy', 'error_rate', 'loss'
    
    Returns:
        str: 比较结果
    """
    try:
        from agent.utils import compare_experiments
        
        exp_dir = get_experiments_dir()
        
        if not experiment_ids:
            # 获取最近的 5 个实验
            exps = sorted(exp_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
            if not exps:
                return "📋 暂无实验记录"
            experiment_ids = [exp.name for exp in exps]
        
        # 读取所有实验记录
        records = []
        for exp_id in experiment_ids:
            record_path = exp_dir / exp_id / "experiment_record.json"
            if record_path.exists():
                with open(record_path, 'r', encoding='utf-8') as f:
                    records.append(json.load(f))
        
        if not records:
            return "❌ 没有找到有效的实验记录"
        
        summary = f"\n📊 实验比较 - 基于指标: {metric}\n"
        summary += "=" * 80 + "\n\n"
        
        # 按指标排序
        valid_records = [r for r in records if r.get('results') and r['results'].get(metric) is not None]
        
        if not valid_records:
            return f"⚠️  没有实验包含指标 '{metric}'"
        
        # 根据指标类型决定排序方向（越小越好：error_rate, eer, loss；越大越好：accuracy）
        reverse = metric == 'accuracy'
        sorted_records = sorted(valid_records, 
                              key=lambda x: x['results'][metric], 
                              reverse=reverse)
        
        for i, record in enumerate(sorted_records, 1):
            exp_id = record['experiment_id']
            value = record['results'][metric]
            status = record.get('status', 'unknown')
            duration = record.get('duration_seconds', 0)
            
            summary += f"{i}. 实验 {exp_id}\n"
            summary += f"   {metric}: {value:.6f}\n"
            summary += f"   状态: {status}\n"
            summary += f"   时长: {duration:.2f}s\n\n"
        
        # 找出最佳实验
        best_exp = sorted_records[0]
        summary += f"✅ 最佳实验: {best_exp['experiment_id']} ({metric}: {best_exp['results'][metric]:.6f})\n"
        
        return summary
    
    except Exception as e:
        return f"❌ 比较实验失败: {str(e)}"


# 导出所有工具
__all__ = [
    'TrainModel',
    'EvaluateModel',
    'AnalyzeResults',
    'CompareExperiments',
]