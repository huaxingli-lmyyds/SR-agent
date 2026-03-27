from logging import config
from logging.config import dictConfig
from math import e, exp
import os
import re
from tabnanny import verbose
import dotenv
from regex import P
from torch import cuda
import yaml
import subprocess
import json
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from typing import Union, Dict, List, Optional,Any
from datetime import datetime

# load environment variables from .env file
dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())
os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("ZHIU_API_BASE_URL")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from langchain_openai import ChatOpenAI

# the path to the ECAPA-TDNN configuration file
CONFIG_PATH = "../configs/train_ecapa_tdnn.yaml"
TRAIN_SCRIPT = "../recipes/voxceleb/train_speaker_embeddings.py"
EVAL_SCRIPT = "../recipes/voxceleb/speaker_verification_cosine.py"
SYSTEM_PROMPT_PATH = "../prompts/hpo_agent_prompt.txt"

# 实验记录目录和文件
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
EXPERIMENTS_FILE = EXPERIMENTS_DIR / "experiments_history.json"
EXPERIMENTS_CONFIGS_DIR = EXPERIMENTS_DIR / "configs"

# 确保实验目录存在
EXPERIMENTS_DIR.mkdir(exist_ok=True)
EXPERIMENTS_CONFIGS_DIR.mkdir(exist_ok=True)

# create the LLM
llm = ChatOpenAI(model="GLM-4.7", temperature=0.2, max_tokens=2000)


@tool
def read_config(config_path=CONFIG_PATH) -> str:
    """
    读取当前 ECAPA-TDNN 配置文件的内容，包括完整的模型结构信息。
    使用 ruamel.yaml 解析，保留 YAML 标签但不实例化对象。

    Returns:
      配置文件的完整 YAML 内容（包括模型结构）或错误信息
    """
    try:
        # 使用 ruamel.yaml 加载，保留标签但不实例化
        from ruamel.yaml import YAML
        
        yaml_parser = YAML()
        yaml_parser.preserve_quotes = True
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml_parser.load(f)
        
        # 转换为普通字典以便于显示
        config_dict = yaml_to_dict(config)
        
        # 返回关键配置和模型结构
        summary = "当前配置 (包括模型结构):\n"
        summary += "=" * 80 + "\n\n"
        
        # 基础训练参数
        basic_params = ['lr', 'batch_size', 'number_of_epochs', 'step_size', 'seed']
        summary += "📊 基础训练参数:\n"
        for param in basic_params:
            if param in config_dict:
                summary += f"  {param}: {config_dict[param]}\n"
        summary += "\n"
        
        # 模型结构参数
        model_sections = ['embedding_model', 'classifier', 'compute_cost', 'opt_class']
        summary += "🏗️  模型结构参数:\n"
        
        for section in model_sections:
            if section in config_dict:
                value = config_dict[section]
                if isinstance(value, dict):
                    summary += f"\n  {section}:\n"
                    # 格式化显示嵌套结构
                    formatted = format_nested_dict(value, indent=4)
                    summary += formatted
                else:
                    summary += f"  {section}: {value}\n"
        
        return summary
    except Exception as e:
        return f"❌ 读取配置失败: {str(e)}"


# define the tools
@tool
def modify_config(config_json: str, persist: bool = True) -> str:
    """
    使用 JSON 形式的配置更新 ECAPA-TDNN YAML 文件。

    参数:
      config_json: JSON 字符串或 dict，表示要更新的字段
        例如: '{"lr": 0.0005,"classifier": {"input_size": 200}}'
      persist: 是否写回文件（True）或仅预览（False），默认 True。

    Returns:
      操作结果描述
    """
    try:
        if isinstance(config_json, str):
            updates = json.loads(config_json)
        elif isinstance(config_json, dict):
            updates = config_json
        else:
            return "config_json 必须是 JSON 字符串或 dict 类型"

        try:
            from ruamel.yaml import YAML
        except ImportError:
            return "需要安装 ruamel.yaml: pip install ruamel.yaml"

        yaml_parser = YAML()
        yaml_parser.preserve_quotes = True

        def _deep_update(orig, upd):
            for k, v in upd.items():
                if k in orig and isinstance(orig[k], dict) and isinstance(v, dict):
                    _deep_update(orig[k], v)
                else:
                    orig[k] = v

        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml_parser.load(f)

        if config is None:
            config = {}

        _deep_update(config, updates)

        if persist:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml_parser.dump(config, f)

        return f"配置已更新: {updates} (persist={persist})"

    except Exception as e:
        return f"修改配置失败: {str(e)}"

def create_experiment_dirs(experiment_id: str) -> dict:
    """
    为实验创建目录结构
    
    Args:
        experiment_id: 实验ID（格式：YYYYMMDD_HHMMSS）
    
    Returns:
        包含所有目录路径的字典
    """
    # 主实验目录
    exp_dir = EXPERIMENTS_DIR / f"exp_{experiment_id}"
    
    # 子目录
    results_dir = exp_dir / "results"
    save_dir = results_dir / "save"
    eval_dir = exp_dir / "evaluation"
    
    # 创建所有目录
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    save_dir.mkdir(exist_ok=True)
    eval_dir.mkdir(exist_ok=True)
    
    return {
        'experiment_dir': exp_dir,
        'results_dir': results_dir,
        'save_dir': save_dir,
        'eval_dir': eval_dir
    }


@tool
def run_training(config_path: str = CONFIG_PATH, experiment_id: str = None) -> str:
    """
    运行 ECAPA-TDNN 模型的训练脚本。
    训练完成后会自动保存实验记录，包括配置、训练日志和结果。
    每次训练会在 experiments/exp_{experiment_id}/ 下创建独立的文件夹结构，
    训练结果也保存到该实验目录下，避免不同训练之间相互干扰。
    
    Args:
        config_path: 配置文件路径，默认为全局 CONFIG_PATH
        experiment_id: 实验ID，如果为None则自动按照时间生成（格式：YYYYMMDD_HHMMSS）
    
    Returns:
      训练输出或错误信息
    """
    try:
        from datetime import datetime
        import shutil
        import sys
        
        # 生成或使用提供的实验ID
        start_time = datetime.now()
        if experiment_id is None:
            experiment_id = start_time.strftime("%Y%m%d_%H%M%S")
        
        # 创建实验目录结构
        exp_dirs = create_experiment_dirs(experiment_id)
        print(f"📁 创建实验目录: {exp_dirs['experiment_dir']}")
        
        # 读取当前配置（只读取基本信息，不修改配置文件）
        current_config = load_config(config_path)
        
        # 检查 data_folder 是否设置
        data_folder = current_config.get('data_folder')
        if data_folder == '!PLACEHOLDER' or not data_folder:
            # 设置默认数据目录
            data_folder = "../datasets/voxceleb1"
            print(f"⚠️  data_folder 未设置，使用默认路径: {data_folder}")
        
        # 关键修改：设置实验特定的输出目录
        # 使用实验目录下的results子目录作为训练输出目录
        # 这样每次训练都有独立的输出目录，不会相互干扰
        relative_output_folder = f"experiments/exp_{experiment_id}/results"
        absolute_output_folder = str(exp_dirs['results_dir'])
        
        print(f"✅ 设置训练输出目录: {absolute_output_folder}")
        
        # 保存原始配置文件的备份到实验目录
        config_backup_path = exp_dirs['experiment_dir'] / "config.yaml"
        try:
            shutil.copy2(config_path, config_backup_path)
            print(f"✅ 原始配置已备份到: {config_backup_path}")
        except Exception as e:
            print(f"⚠️ 备份配置文件失败: {e}")
        
        # 使用当前Python解释器（虚拟环境的Python）
        python_executable = sys.executable
        print(f"使用Python解释器: {python_executable}")
        print(f"开始训练实验: {experiment_id}")
        
        # 运行训练脚本，通过命令行参数指定输出目录和数据目录
        # 不修改配置文件，保留原始配置中的所有实例化信息（如 !new: 标签）
        result = subprocess.run(
            [python_executable, TRAIN_SCRIPT, config_path, 
             f"--data_folder={data_folder}",
             f"--output_folder={relative_output_folder}"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # 记录结束时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
        # if 1 == 1:  # 无论训练成功与否都继续执行后续代码，记录实验结果
            # 使用实验特定的输出路径（通过命令行参数指定的）
            output_folder = relative_output_folder
            
            # 解析训练日志
            # 日志通常在output_folder/train_log.txt
            train_log_path = exp_dirs['results_dir'] / "train_log.txt"
            epoch_data, final_metrics = parse_training_log(train_log_path)
            
            # 查找最新的checkpoint文件夹
            checkpoint_path = None
            save_dir = Path(__file__).parent / output_folder / "save"
            if save_dir.exists():
                # checkpoint是文件夹，格式: CKPT+2026-03-25+02-36-45+00
                ckpt_dirs = list(save_dir.glob("CKPT+*"))
                if ckpt_dirs:
                    # 找到最新的checkpoint文件夹（按修改时间排序）
                    checkpoint_path = sorted(ckpt_dirs, key=lambda x: x.stat().st_mtime)[-1]
                    print(f"✅ 找到最新checkpoint文件夹: {checkpoint_path}")
            
            # # 如果训练日志存在，复制到实验目录（已在训练前备份配置，这里不再重复）
            # if train_log_path.exists():
            #     train_log_backup = exp_dirs['results_dir'] / "train_log.txt"
            #     shutil.copy2(train_log_path, train_log_backup)
            #     print(f"✅ 训练日志已复制到: {train_log_backup}")
            
            # 如果checkpoint存在，记录路径

            checkpoint_info = {}
            if checkpoint_path:
                checkpoint_info['checkpoint_path'] = str(checkpoint_path)
                
                
                
            # 创建并保存实验记录
            experiment_record = create_experiment_record(
                experiment_id=experiment_id,
                start_time=start_time,
                duration=duration,
                status="success",
                config=current_config,
                training_log_path=str(train_log_path),
                epoch_data=epoch_data,
                final_metrics=final_metrics,
                config_backup_path=str(config_backup_path),
                experiment_dir=str(exp_dirs['experiment_dir']),
                output_folder=str(output_folder),
                checkpoint_info=checkpoint_info
            )
            save_experiment_record(experiment_record)
            
            # 保存实验记录到单独的文件
            exp_record_path = exp_dirs['experiment_dir'] / "experiment_record.json"
            with open(exp_record_path, 'w', encoding='utf-8') as f:
                json.dump(experiment_record, f, indent=2, ensure_ascii=False)
            print(f"✅ 实验记录已保存到: {exp_record_path}")
            
            # 构建返回信息
            result_summary = f"✅ 训练完成！\n"
            result_summary += f"实验ID: {experiment_id}\n"
            result_summary += f"训练时长: {duration:.2f} 秒\n"
            result_summary += f"实验目录: {exp_dirs['experiment_dir']}\n"
            result_summary += f"训练输出目录: {output_folder}\n"
            
            if final_metrics:
                result_summary += f"\n性能指标:\n"
                result_summary += f"  - 最终Epoch: {final_metrics.get('final_epoch', 'N/A')}\n"
                result_summary += f"  - 最终学习率: {final_metrics.get('final_lr', 'N/A'):.2e}\n"
                result_summary += f"  - 最终训练损失: {final_metrics.get('final_train_loss', 'N/A'):.4f}\n"
                result_summary += f"  - 最终验证损失: {final_metrics.get('final_valid_loss', 'N/A'):.4f}\n"
                result_summary += f"  - 最终验证错误率: {final_metrics.get('final_valid_error_rate', 'N/A'):.4f}\n"
                result_summary += f"  - 最佳Epoch: {final_metrics.get('best_epoch', 'N/A')}\n"
                result_summary += f"  - 最佳验证错误率: {final_metrics.get('best_error_rate', 'N/A'):.4f}\n"
            else:
                result_summary += "\n未能解析训练日志\n"
            
            if checkpoint_path:
                result_summary += f"\n模型Checkpoints:\n"
                result_summary += f"  - 最新checkpoint: {checkpoint_path.name}\n"
                result_summary += f"  - 完整路径: {checkpoint_path}\n"
            
            result_summary += f"\n配置和结果已保存到 experiments/exp_{experiment_id}/\n"
            
            return result_summary
        else:
            # 训练失败，记录失败信息
            # 处理错误信息（可能是字节串或字符串）
            error_msg = "Unknown error"
            if result.stderr:
                if isinstance(result.stderr, bytes):
                    try:
                        error_msg = result.stderr.decode('utf-8', errors='ignore')[-1000:]
                    except:
                        error_msg = str(result.stderr)[-1000:]
                else:
                    error_msg = str(result.stderr)[-1000:]
            
            experiment_record = create_experiment_record(
                start_time=start_time,
                duration=duration,
                status="failed",
                config=current_config,
                experiment_dir=str(exp_dirs['experiment_dir']),
                error=error_msg
            )
            save_experiment_record(experiment_record)
            
            # 返回给用户的错误信息
            user_error_msg = "Unknown error"
            if result.stderr:
                if isinstance(result.stderr, bytes):
                    try:
                        user_error_msg = result.stderr.decode('utf-8', errors='ignore')[-500:]
                    except:
                        user_error_msg = str(result.stderr)[-500:]
                else:
                    user_error_msg = str(result.stderr)[-500:]
            
            return f"❌ 训练失败 (实验ID: {experiment_id})\n错误信息: {user_error_msg}"
    except Exception as e:
            return f"❌ 运行训练失败: {str(e)}"

@tool
def run_evaluation(eval_config_path: str = "../configs/verification_ecapa.yaml", 
                 experiment_id: str = None,
                 checkpoint_path: str = None) -> str:
    """
    运行 ECAPA-TDNN 模型的评估脚本，使用训练后的模型。
    评估结果会保存到对应的实验记录中。
    从日志文件中读取EER和minDCF参数。

    Args:
        eval_config_path: 评估配置文件路径，默认为 "../configs/verification_ecapa.yaml"
        experiment_id: 实验ID，如果为None则使用最新的成功实验
        checkpoint_path: checkpoint文件路径，如果为None则从实验记录中读取

    Returns:
      评估结果或错误信息
    """
    try:
        from datetime import datetime
        import re
        import sys
        import shutil
        
        # 确定目标实验
        target_experiment = None
        
        if EXPERIMENTS_FILE.exists():
            with open(EXPERIMENTS_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # 根据experiment_id查找实验
            if experiment_id:
                target_experiment = next((exp for exp in history if exp['experiment_id'] == experiment_id), None)
            else:
                # 使用最新的成功实验
                successful_experiments = [exp for exp in history if exp['status'] == 'success']
                if successful_experiments:
                    target_experiment = successful_experiments[-1]
        
        if not target_experiment or target_experiment['status'] != 'success':
            return "❌ 未找到可评估的实验记录。请先运行训练。"
        
        # 从实验记录中获取checkpoint路径
        if checkpoint_path is None:
            if target_experiment.get('checkpoint_info') and target_experiment['checkpoint_info'].get('checkpoint_path'):
                checkpoint_path = target_experiment['checkpoint_info']['checkpoint_path']
                print(f"✅ 从实验记录中读取checkpoint: {checkpoint_path}")
            else:
                return "❌ 实验记录中未找到checkpoint路径。请检查训练是否成功。"
        
        # 检查checkpoint文件是否存在
        if not Path(checkpoint_path).exists():
            return f"❌ Checkpoint文件不存在: {checkpoint_path}"
        
        # 获取实验目录信息
        exp_id = target_experiment['experiment_id']
        exp_dir = Path(target_experiment.get('experiment_dir', f'./experiments/exp_{exp_id}'))
        eval_dir = exp_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置评估的独立输出目录
        relative_eval_output_folder = f"experiments/exp_{exp_id}/evaluation/results"
        absolute_eval_output_folder = eval_dir / "results"
        
        print(f"✅ 设置评估输出目录: {absolute_eval_output_folder}")
        
        # 读取评估配置以获取data_folder（不修改配置文件）
        eval_config = load_config(eval_config_path)
        
        # 设置data_folder
        data_folder = eval_config.get('data_folder')
        if data_folder == '!PLACEHOLDER' or not data_folder:
            data_folder = "../datasets/voxceleb1"
            print(f"⚠️  data_folder 未设置，使用默认路径: {data_folder}")
        
        # 备份原始评估配置文件到实验目录
        eval_config_backup = eval_dir / "verification_config.yaml"
        try:
            shutil.copy2(eval_config_path, eval_config_backup)
            print(f"✅ 评估配置已备份到: {eval_config_backup}")
        except Exception as e:
            print(f"⚠️ 备份评估配置失败: {e}")
        
        start_time = datetime.now()
        
        # 使用当前Python解释器（虚拟环境的Python）
        python_executable = sys.executable
        print(f"使用Python解释器: {python_executable}")
        print(f"开始评估实验: {exp_id}")
        print(f"使用checkpoint: {checkpoint_path}")
        
        # 运行评估脚本，通过命令行参数指定输出目录、pretrain_path和数据目录
        # pretrain_path指向训练保存的checkpoint文件夹
        # 不修改配置文件，保留原始配置中的所有实例化信息
        result = subprocess.run(
            [python_executable, EVAL_SCRIPT, eval_config_path,
             f"--data_folder={data_folder}",
             f"--pretrain_path={checkpoint_path}",
             f"--output_folder={relative_eval_output_folder}"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 从日志文件中读取评估结果
        eer = None
        min_dcf = None
        eval_log_path = None
        
        try:
            # 从配置中获取seed来确定日志路径
            seed = eval_config.get('seed', '1234')
            if isinstance(seed, dict):
                seed = '1234'
            elif not isinstance(seed, str):
                seed = str(seed)
            else:
                seed = str(seed)
            
            # 从我们指定的评估输出目录中读取日志
            # 路径: experiments/exp_{exp_id}/evaluation/results/speaker_verification_ecapa/{seed}/log.txt
            log_path = absolute_eval_output_folder / "speaker_verification_ecapa" / seed / "log.txt"
            
            print(f"🔍 查找评估日志: {log_path}")
            
            if log_path.exists():
                eval_log_path = log_path
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                
                # 从日志中提取EER和minDCF
                eer_match = re.search(r'EER\(%\)=([\d.]+)', log_content[-1000:])
                if eer_match:
                    eer = float(eer_match.group(1))
                
                dcf_match = re.search(r'minDCF=([\d.]+)', log_content[-1000:])
                if dcf_match:
                    min_dcf = float(dcf_match.group(1))
                
                print(f"✅ 从日志文件中读取到: EER={eer}, minDCF={min_dcf}")
                
                # # 复制评估日志到实验目录
                # eval_log_backup = eval_dir / "evaluation_log.txt"
                # shutil.copy2(log_path, eval_log_backup)
                # print(f"✅ 评估日志已复制到: {eval_log_backup}")
            else:
                print(f"⚠️ 日志文件不存在: {log_path}")
                # 尝试查找可能的日志文件
                if absolute_eval_output_folder.exists():
                    print(f"📁 评估输出目录内容:")
                    for item in absolute_eval_output_folder.rglob("*.txt"):
                        print(f"   - {item}")
        except Exception as e:
            print(f"⚠️ 从日志文件读取评估结果失败: {e}")
            import traceback
            traceback.print_exc()
        
        if result.returncode == 0:
            # 更新实验记录（添加评估结果）
            try:
                if EXPERIMENTS_FILE.exists():
                    with open(EXPERIMENTS_FILE, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                    
                    # 更新对应的实验记录
                    for exp in history:
                        if exp['experiment_id'] == exp_id:
                            exp['evaluation_results'] = {
                                'timestamp': end_time.isoformat(),
                                'duration_seconds': duration,
                                'eer': eer,
                                'min_dcf': min_dcf,
                                'evaluation_log_path': str(eval_log_path) if eval_log_path else None,
                                'checkpoint_used': checkpoint_path,
                                'evaluation_dir': str(eval_dir)
                            }
                            break
                    
                    # 保存更新后的历史
                    with open(EXPERIMENTS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(history, f, indent=2, ensure_ascii=False)
                    
                    # 同时更新实验目录中的实验记录文件
                    exp_record_path = exp_dir / "experiment_record.json"
                    if exp_record_path.exists():
                        with open(exp_record_path, 'r', encoding='utf-8') as f:
                            exp_record = json.load(f)
                        
                        exp_record['evaluation_results'] = {
                            'timestamp': end_time.isoformat(),
                            'duration_seconds': duration,
                            'eer': eer,
                            'min_dcf': min_dcf,
                            'evaluation_log_path': str(eval_log_path) if eval_log_path else None,
                            'checkpoint_used': checkpoint_path,
                            'evaluation_dir': str(eval_dir)
                        }
                        
                        with open(exp_record_path, 'w', encoding='utf-8') as f:
                            json.dump(exp_record, f, indent=2, ensure_ascii=False)
                        print(f"✅ 实验记录已更新: {exp_record_path}")
            except Exception as e:
                print(f"⚠️ 保存评估结果到实验记录失败: {e}")
            
            result_summary = f"✅ 评估完成！\n"
            result_summary += f"实验ID: {exp_id}\n"
            result_summary += f"评估时长: {duration:.2f} 秒\n"
            result_summary += f"使用的Checkpoint: {checkpoint_path}\n"
            result_summary += f"评估结果目录: {eval_dir}\n"
            
            if eer is not None:
                result_summary += f"\n性能指标:\n"
                result_summary += f"  - EER (等错误率): {eer:.4f}%\n"
            if min_dcf is not None:
                result_summary += f"  - minDCF: {min_dcf:.4f}\n"
            
            result_summary += f"\n评估结果已保存到实验记录中\n"
            if eval_log_path:
                result_summary += f"评估日志: {eval_log_path}\n"
            
            return result_summary
        else:
            return f"❌ 评估失败\n错误信息: {result.stderr[-500:] if result.stderr else 'Unknown error'}"
    except Exception as e:
        import traceback
        return f"❌ 运行评估失败: {str(e)}\n{traceback.format_exc()}"



@tool
def get_training_logs() -> str:
    """
    获取训练日志以监控训练过程和性能。

    Returns:
      训练日志内容或错误信息
    """
    try:
        config = load_yaml_config_hyperpyyaml(CONFIG_PATH)
        log_path = config.get('train_log')
        
        
        if not os.path.exists(log_path):
            return f"训练日志文件不存在: {log_path}"
        
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 返回最后1000个字符
        return f"训练日志 (最近内容):\n{content[-1000:]}"
    except Exception as e:
        return f"读取训练日志失败: {str(e)}"

@tool
def backup_config() -> str:
    """
    备份当前配置文件，防止实验失败导致配置丢失。

    Returns:
      备份操作结果
    """
    try:
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{CONFIG_PATH}.backup_{timestamp}"
        
        shutil.copy2(CONFIG_PATH, backup_path)
        return f"✅ 配置已备份到: {backup_path}"
    except Exception as e:
        return f"❌ 备份配置失败: {str(e)}"

@tool
def view_experiment_history(n: int = 10) -> str:
    """
    查看实验历史记录。

    参数:
      n: 显示最近 n 次实验记录，默认 10

    Returns:
      实验历史记录摘要
    """
    try:
        if not EXPERIMENTS_FILE.exists():
            return "📋 暂无实验记录。请先运行训练以创建实验记录。"
        
        with open(EXPERIMENTS_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if not history:
            return "📋 实验历史为空。"
        
        # 只显示最近的 n 次实验
        recent_experiments = history[-n:]
        
        summary = f"\n📊 最近 {len(recent_experiments)} 次实验记录:\n"
        summary += "=" * 80 + "\n\n"
        
        for exp in reversed(recent_experiments):
            status_icon = "✅" if exp['status'] == 'success' else "❌"
            summary += f"{status_icon} 实验 ID: {exp['experiment_id']}\n"
            summary += f"   时间: {exp['timestamp']}\n"
            summary += f"   状态: {exp['status']}\n"
            summary += f"   训练时长: {exp['duration_seconds']:.2f} 秒\n"
            summary += f"   配置: lr={exp['config'].get('lr')}, batch_size={exp['config'].get('batch_size')}, epochs={exp['config'].get('number_of_epochs')}\n"
            
            if exp['status'] == 'success':
                # 显示训练指标
                if exp.get('final_metrics'):
                    metrics = exp['final_metrics']
                    summary += f"   训练指标:\n"
                    if 'best_error_rate' in metrics:
                        summary += f"     - 最佳验证错误率: {metrics['best_error_rate']:.4f} (Epoch {metrics['best_epoch']})\n"
                    if 'final_valid_error_rate' in metrics:
                        summary += f"     - 最终验证错误率: {metrics['final_valid_error_rate']:.4f} (Epoch {metrics['final_epoch']})\n"
                    if 'final_train_loss' in metrics:
                        summary += f"     - 最终训练损失: {metrics['final_train_loss']:.4f}\n"
                    if 'final_valid_loss' in metrics:
                        summary += f"     - 最终验证损失: {metrics['final_valid_loss']:.4f}\n"
                
                # 显示评估指标
                if exp.get('evaluation_results'):
                    eval_results = exp['evaluation_results']
                    summary += f"   评估指标:\n"
                    if eval_results.get('eer') is not None:
                        summary += f"     - EER: {eval_results['eer']:.4f}\n"
                    if eval_results.get('min_dcf') is not None:
                        summary += f"     - minDCF: {eval_results['min_dcf']:.4f}\n"
                    if eval_results.get('accuracy') is not None:
                        summary += f"     - 准确率: {eval_results['accuracy']:.4f}\n"
            
            summary += "\n"
        
        return summary
    except Exception as e:
        return f"❌ 读取实验历史失败: {str(e)}"

@tool
def analyze_training_trends(experiment_id: str = None) -> str:
    """
    分析训练趋势，识别过拟合、欠拟合等问题。
    如果不提供 experiment_id，则分析最新的实验。

    参数:
      experiment_id: 实验ID，如果为None则分析最新实验

    Returns:
      训练趋势分析报告
    """
    try:
        if not EXPERIMENTS_FILE.exists():
            return "📋 暂无实验记录。"
        
        with open(EXPERIMENTS_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # 获取目标实验
        target_exp = None
        if experiment_id:
            target_exp = next((exp for exp in history if exp['experiment_id'] == experiment_id), None)
        else:
            # 获取最新的成功实验
            successful_experiments = [exp for exp in history if exp['status'] == 'success']
            if successful_experiments:
                target_exp = successful_experiments[-1]
        
        if not target_exp or target_exp['status'] != 'success':
            return "📋 未找到指定的成功实验记录。"
        
        epoch_data = target_exp.get('epoch_data', [])
        if not epoch_data or len(epoch_data) < 5:
            return "📋 实验数据不足，无法进行趋势分析。"
        
        # 分析趋势
        train_losses = [ep['train_loss'] for ep in epoch_data]
        valid_losses = [ep['valid_loss'] for ep in epoch_data]
        valid_error_rates = [ep['valid_error_rate'] for ep in epoch_data]
        
        summary = f"\n📈 训练趋势分析 (实验ID: {target_exp['experiment_id']})\n"
        summary += "=" * 80 + "\n\n"
        
        # 1. 过拟合检测
        # 检查验证损失是否在上升
        valid_loss_trend = valid_losses[-5:] if len(valid_losses) >= 5 else valid_losses
        is_valid_loss_increasing = all(valid_loss_trend[i] >= valid_loss_trend[i-1] for i in range(1, len(valid_loss_trend)))
        
        if is_valid_loss_increasing:
            summary += "⚠️  可能存在过拟合：\n"
            summary += f"   - 验证损失在最近 {len(valid_loss_trend)} 个epoch中持续上升\n"
            summary += f"   - 建议尝试：增加正则化、减少模型复杂度、增加数据增强\n\n"
        
        # 2. 欠拟合检测
        avg_train_loss = sum(train_losses[-10:]) / min(10, len(train_losses))
        avg_valid_loss = sum(valid_losses[-10:]) / min(10, len(valid_losses))
        
        if avg_train_loss > 0.5 and avg_valid_loss > 0.5:
            summary += "⚠️  可能存在欠拟合：\n"
            summary += f"   - 训练和验证损失都较高 (train={avg_train_loss:.4f}, valid={avg_valid_loss:.4f})\n"
            summary += f"   - 建议尝试：增加模型容量、增加训练epoch、调整学习率\n\n"
        
        # 3. 训练稳定性分析
        loss_variance = sum((x - avg_train_loss)**2 for x in train_losses[-10:]) / min(10, len(train_losses))
        
        if loss_variance > 0.01:
            summary += "⚠️  训练不稳定：\n"
            summary += f"   - 训练损失方差较大 ({loss_variance:.4f})\n"
            summary += f"   - 建议尝试：降低学习率、使用学习率调度、增加batch size\n\n"
        else:
            summary += "✅ 训练较为稳定\n\n"
        
        # 4. 学习率建议
        final_lr = target_exp['final_metrics'].get('final_lr', 0)
        best_epoch = target_exp['final_metrics'].get('best_epoch', 0)
        total_epochs = len(epoch_data)
        
        summary += f"📊 关键指标：\n"
        summary += f"   - 最佳epoch: {best_epoch} / {total_epochs}\n"
        summary += f"   - 最终学习率: {final_lr:.2e}\n"
        summary += f"   - 最佳验证错误率: {min(valid_error_rates):.4f}\n"
        
        # 根据最佳epoch位置提供建议
        if best_epoch < total_epochs * 0.3:
            summary += f"\n💡 建议：最佳epoch出现较早，考虑：\n"
            summary += f"   - 增加正则化防止过拟合\n"
            summary += f"   - 使用学习率衰减\n"
        elif best_epoch > total_epochs * 0.8:
            summary += f"\n💡 建议：最佳epoch出现较晚，考虑：\n"
            summary += f"   - 增加训练epoch\n"
            summary += f"   - 提高学习率加速训练\n"
        
        return summary
    except Exception as e:
        return f"❌ 分析训练趋势失败: {str(e)}"


@tool
def get_experiment_details(experiment_id: str) -> str:
    """
    获取特定实验的详细信息。

    参数:
      experiment_id: 实验ID

    Returns:
      实验详细信息
    """
    try:
        if not EXPERIMENTS_FILE.exists():
            return "📋 暂无实验记录。"
        
        with open(EXPERIMENTS_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        exp = next((e for e in history if e['experiment_id'] == experiment_id), None)
        
        if not exp:
            return f"📋 未找到实验ID为 {experiment_id} 的记录。"
        
        summary = f"\n📋 实验详细信息 (ID: {exp['experiment_id']})\n"
        summary += "=" * 80 + "\n\n"
        summary += f"时间: {exp['timestamp']}\n"
        summary += f"状态: {exp['status']}\n"
        summary += f"训练时长: {exp['duration_seconds']:.2f} 秒\n\n"
        
        summary += "配置:\n"
        for key, value in exp['config'].items():
            summary += f"  {key}: {value}\n"
        
        if exp['status'] == 'success':
            if exp.get('final_metrics'):
                summary += "\n训练指标:\n"
                metrics = exp['final_metrics']
                for key, value in metrics.items():
                    summary += f"  {key}: {value}\n"
            
            if exp.get('evaluation_results'):
                summary += "\n评估指标:\n"
                eval_results = exp['evaluation_results']
                for key, value in eval_results.items():
                    if key != 'output':
                        summary += f"  {key}: {value}\n"
        
        summary += f"\n配置备份: {exp.get('config_backup_path', 'N/A')}\n"
        summary += f"训练日志: {exp.get('training_log_path', 'N/A')}\n"
        
        return summary
    except Exception as e:
        return f"❌ 获取实验详情失败: {str(e)}"


@tool
def compare_experiments(experiment_ids: str) -> str:
    """
    比较多个实验的性能。

    参数:
      experiment_ids: 实验ID列表，用逗号分隔，例如 "20260326_001,20260326_002"

    Returns:
      实验比较结果
    """
    try:
        if not EXPERIMENTS_FILE.exists():
            return "📋 暂无实验记录。"
        
        ids = [id.strip() for id in experiment_ids.split(',')]
        
        with open(EXPERIMENTS_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        experiments = [exp for exp in history if exp['experiment_id'] in ids]
        
        if len(experiments) < 2:
            return f"📋 找到 {len(experiments)} 个实验，至少需要2个进行比较。"
        
        summary = f"\n📊 实验比较 ({len(experiments)} 个实验)\n"
        summary += "=" * 80 + "\n\n"
        
        # 比较表格
        summary += f"{'实验ID':<20} {'学习率':<12} {'Batch':<8} {'Epochs':<8} {'最佳错误率':<12} {'EER':<10}\n"
        summary += "-" * 80 + "\n"
        
        for exp in experiments:
            exp_id = exp['experiment_id']
            lr = exp['config'].get('lr', 'N/A')
            batch_size = exp['config'].get('batch_size', 'N/A')
            epochs = exp['config'].get('number_of_epochs', 'N/A')
            
            best_error_rate = 'N/A'
            if exp.get('final_metrics') and 'best_error_rate' in exp['final_metrics']:
                best_error_rate = f"{exp['final_metrics']['best_error_rate']:.4f}"
            
            eer = 'N/A'
            if exp.get('evaluation_results') and 'eer' in exp['evaluation_results']:
                eer = f"{exp['evaluation_results']['eer']:.4f}"
            
            summary += f"{exp_id:<20} {lr:<12} {batch_size:<8} {epochs:<8} {best_error_rate:<12} {eer:<10}\n"
        
        # 找出最佳配置
        successful_exps = [exp for exp in experiments if exp['status'] == 'success']
        if successful_exps:
            best_exp = min(successful_exps, 
                          key=lambda x: x['final_metrics'].get('best_error_rate', float('inf'))
                          if x.get('final_metrics') else float('inf'))
            
            summary += "\n🏆 最佳配置（按最佳验证错误率）：\n"
            summary += f"  实验ID: {best_exp['experiment_id']}\n"
            summary += f"  学习率: {best_exp['config'].get('lr')}\n"
            summary += f"  批次大小: {best_exp['config'].get('batch_size')}\n"
            summary += f"  最佳错误率: {best_exp['final_metrics'].get('best_error_rate', 'N/A')}\n"
        
        return summary
    except Exception as e:
        return f"❌ 比较实验失败: {str(e)}"


@tool
def get_best_experiment(metric: str = "best_error_rate") -> str:
    """
    找出最佳实验（根据指定指标）。

    参数:
      metric: 优化指标，可选:
        - 'best_error_rate' (最佳验证错误率，越小越好，默认)
        - 'final_valid_error_rate' (最终验证错误率，越小越好)
        - 'eer' (等错误率，越小越好)
        - 'accuracy' (准确率，越大越好)

    Returns:
      最佳实验的详细信息
    """
    try:
        if not EXPERIMENTS_FILE.exists():
            return "📋 暂无实验记录。请先运行训练以创建实验记录。"
        
        with open(EXPERIMENTS_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # 筛选成功的实验
        successful_experiments = [exp for exp in history if exp['status'] == 'success']
        
        if not successful_experiments:
            return "📋 暂无成功的实验记录。"
        
        # 根据指标找到最佳实验
        best_exp = None
        metric_name = ""
        metric_value = None
        
        if metric == "best_error_rate":
            # 最佳验证错误率（越小越好）
            best_exp = min(successful_experiments, 
                         key=lambda x: x['final_metrics'].get('best_error_rate', float('inf')) 
                         if x.get('final_metrics') and 'best_error_rate' in x['final_metrics'] else float('inf'))
            metric_name = "最佳验证错误率"
            if best_exp and best_exp.get('final_metrics'):
                metric_value = best_exp['final_metrics'].get('best_error_rate')
        
        elif metric == "final_valid_error_rate":
            # 最终验证错误率（越小越好）
            best_exp = min(successful_experiments, 
                         key=lambda x: x['final_metrics'].get('final_valid_error_rate', float('inf'))
                         if x.get('final_metrics') and 'final_valid_error_rate' in x['final_metrics'] else float('inf'))
            metric_name = "最终验证错误率"
            if best_exp and best_exp.get('final_metrics'):
                metric_value = best_exp['final_metrics'].get('final_valid_error_rate')
        
        elif metric == "eer":
            # EER 越小越好（来自评估结果）
            best_exp = min(successful_experiments, 
                         key=lambda x: x['evaluation_results'].get('eer', float('inf'))
                         if x.get('evaluation_results') and 'eer' in x['evaluation_results'] else float('inf'))
            metric_name = "等错误率 (EER)"
            if best_exp and best_exp.get('evaluation_results'):
                metric_value = best_exp['evaluation_results'].get('eer')
        
        elif metric == "accuracy":
            # 准确率越大越好（来自评估结果）
            best_exp = max(successful_experiments, 
                         key=lambda x: x['evaluation_results'].get('accuracy', 0)
                         if x.get('evaluation_results') and 'accuracy' in x['evaluation_results'] else 0)
            metric_name = "准确率"
            if best_exp and best_exp.get('evaluation_results'):
                metric_value = best_exp['evaluation_results'].get('accuracy')
        
        else:
            return f"❌ 不支持的指标: {metric}。请使用 'best_error_rate', 'final_valid_error_rate', 'eer' 或 'accuracy'。"
        
        if best_exp is None or metric_value is None:
            return f"📋 没有找到包含 {metric} 指标的实验记录。"
        
        # 构建返回摘要
        summary = f"\n🏆 最佳实验 (按 {metric_name}):\n"
        summary += "=" * 80 + "\n\n"
        summary += f"实验 ID: {best_exp['experiment_id']}\n"
        summary += f"时间: {best_exp['timestamp']}\n"
        summary += f"训练时长: {best_exp['duration_seconds']:.2f} 秒\n"
        summary += f"\n配置:\n"
        for key, value in best_exp['config'].items():
            summary += f"  {key}: {value}\n"
        
        # 显示训练指标
        if best_exp.get('final_metrics'):
            summary += f"\n训练指标:\n"
            metrics = best_exp['final_metrics']
            if 'best_error_rate' in metrics:
                summary += f"  最佳验证错误率: {metrics['best_error_rate']:.4f} (Epoch {metrics.get('best_epoch', 'N/A')})\n"
            if 'final_valid_error_rate' in metrics:
                summary += f"  最终验证错误率: {metrics['final_valid_error_rate']:.4f} (Epoch {metrics.get('final_epoch', 'N/A')})\n"
            if 'final_train_loss' in metrics:
                summary += f"  最终训练损失: {metrics['final_train_loss']:.4f}\n"
            if 'final_valid_loss' in metrics:
                summary += f"  最终验证损失: {metrics['final_valid_loss']:.4f}\n"
        
        # 显示评估指标
        if best_exp.get('evaluation_results'):
            summary += f"\n评估指标:\n"
            eval_results = best_exp['evaluation_results']
            if eval_results.get('eer') is not None:
                summary += f"  EER: {eval_results['eer']:.4f}\n"
            if eval_results.get('min_dcf') is not None:
                summary += f"  minDCF: {eval_results['min_dcf']:.4f}\n"
            if eval_results.get('accuracy') is not None:
                summary += f"  准确率: {eval_results['accuracy']:.4f}\n"
        
        summary += f"\n配置备份: {best_exp.get('config_backup_path', 'N/A')}\n"
        summary += f"训练日志: {best_exp.get('training_log_path', 'N/A')}\n"
        
        return summary
    except Exception as e:
        return f"❌ 查找最佳实验失败: {str(e)}"

    
    
def load_yaml_config_hyperpyyaml(config_path):
    """加载包含 SpeechBrain 特殊标签的 YAML 配置文件，使用 hyperpyyaml
    
    注意：此函数不会实例化对象（如 !new: 标签），仅解析配置和引用
    使用 resolve_references 而不是 load_hyperpyyaml 来避免对象实例化
    """
    try:
        from hyperpyyaml import load_hyperpyyaml
        
        # 使用 hyperpyyaml 加载配置，能够处理 !ref, !new:, !name: 等特殊标签
        with open(config_path, 'r', encoding='utf-8') as f:
            # 提供默认的 overrides 来处理 !PLACEHOLDER 标签
            overrides = {
                'data_folder': './data'  # 为占位符提供默认值
            }
            
            # hyperpyyaml 的 load_hyperpyyaml 可以处理 SpeechBrain 的特殊标签
            # 返回 (config, overrides) 元组，我们只需要 config
            config = load_hyperpyyaml(f, overrides=overrides)
        
        return config
    except ImportError:
        print("⚠️ hyperpyyaml 未安装，使用备用解析方法")
        return load_config(config_path)
    except Exception as e:
        print(f"⚠️ hyperpyyaml 加载失败: {e}")
        print("尝试使用备用解析方法...")
        return load_config(config_path)


def yaml_to_dict(data, config=None):
    """
    将 ruamel.yaml 解析的数据转换为标准 Python 字典，
    保留所有结构信息但不实例化对象。
    同时处理所有 <key> 形式的引用，替换为实际值。
    
    参数:
      data: ruamel.yaml 解析的数据（可能是 CommentedMap 等类型）
      config: 完整的配置字典，用于解析引用
    
    Returns:
      标准的 Python 字典/列表
    """
    # 处理 TaggedScalar 类型
    if hasattr(data, 'value') and hasattr(data, 'tag'):
        # 提取标签和值
        tag = str(data.tag) if hasattr(data, 'tag') else ''
        value = data.value
        
        # 如果有config，尝试解析引用
        if config is not None:
            # 处理 <key> 形式的引用
            if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
                ref_key = value[1:-1].strip()
                if ref_key in config:
                    return parse_yaml_tags(config[ref_key], config)
                else:
                    # 引用不存在，返回空字符串
                    return ""
            # 处理包含 <key> 的字符串
            elif isinstance(value, str) and "<" in value and ">" in value:
                result = value
                import re
                matches = re.findall(r'<([^>]+)>', value)
                for match in matches:
                    if match in config:
                        resolved = parse_yaml_tags(config[match], config)
                        result = result.replace(f"<{match}>", str(resolved))
                    else:
                        # 引用不存在，移除这个占位符
                        result = result.replace(f"<{match}>", "")
                return result
            # 处理 !ref, !apply: 等特殊标签
            elif isinstance(value, str) and (value.startswith("!ref <") or value.startswith("!apply:") or value.startswith("!new:")):
                # 保留原始字符串（不带标签）
                return value
            else:
                return value
        else:
            # 没有config，只提取值
            return value
    
    elif isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = yaml_to_dict(value, config)
        return result
    
    elif isinstance(data, list):
        return [yaml_to_dict(item, config) for item in data]
    
    else:
        # 对于基本类型（字符串、数字、布尔值等），如果有config则解析引用
        if config is not None and isinstance(data, str):
            # 处理 <key> 形式的引用
            if data.startswith("<") and data.endswith(">"):
                ref_key = data[1:-1].strip()
                if ref_key in config:
                    return parse_yaml_tags(config[ref_key], config)
                else:
                    return ""
            # 处理包含 <key> 的字符串
            elif "<" in data and ">" in data:
                result = data
                import re
                matches = re.findall(r'<([^>]+)>', data)
                for match in matches:
                    if match in config:
                        resolved = parse_yaml_tags(config[match], config)
                        result = result.replace(f"<{match}>", str(resolved))
                    else:
                        result = result.replace(f"<{match}>", "")
                return result
            else:
                return data
        else:
            return data


def format_nested_dict(data, indent=0):
    """
    格式化显示嵌套字典，用于美观地展示配置信息。
    
    参数:
      data: 要格式化的字典数据
      indent: 缩进空格数
    
    Returns:
      格式化的字符串
    """
    result = []
    prefix = " " * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            result.append(f"{prefix}{key}:")
            # 递归格式化嵌套字典
            nested = format_nested_dict(value, indent + 2)
            result.append(nested)
        elif isinstance(value, list):
            # 格式化列表
            if value and isinstance(value[0], dict):
                # 列表中包含字典
                result.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        nested = format_nested_dict(item, indent + 2)
                        result.append(nested)
                    else:
                        result.append(f"{prefix}  - {item}")
            else:
                # 简单列表，在一行显示
                result.append(f"{prefix}{key}: {value}")
        else:
            # 简单键值对
            result.append(f"{prefix}{key}: {value}")
    
    return "\n".join(result)


def load_config(config_path) -> Dict:
        """
        加载配置文件并转换为标准字典
        解析所有 <key> 形式的引用，替换为实际值
        
        参数:
            config_path: 配置文件路径
        
        Returns:
            配置数据（标准字典）
        """
        try:
            from ruamel.yaml import YAML
            
            yaml_parser = YAML()
            yaml_parser.preserve_quotes = True
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml_parser.load(f)
            
            if config is None:
                config = {}
            
            # 先转换为标准字典
            config_dict = yaml_to_dict(config)
            
            # 再次传入config_dict来解析引用
            config_dict = yaml_to_dict(config_dict, config_dict)
            
            return config_dict
        
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}") from e

def resolve_yaml_references(config_path: Union[str, Path]) -> Dict:
    """
    解析配置中的所有 YAML 引用（!ref, !new:, !apply: 等）
    
    参数:
        config_path: 配置文件路径
    
    Returns:
        解析后的配置字典
    """
    from ruamel.yaml import YAML
    
    yaml_parser = YAML()
    yaml_parser.preserve_quotes = True
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml_parser.load(f)
    
    if config is None:
        return {}
    
    # 转换为标准字典
    config_dict = yaml_to_dict(config)
    
    resolved_config = {}
    
    for key, value in config_dict.items():
        # 解析每个值
        resolved_value = parse_yaml_tags(value, config_dict)
        resolved_config[key] = resolved_value
    
    return resolved_config


def parse_yaml_tags(value: Any, config: Dict) -> Any:
    """
    解析 YAML 中的特殊标签（!ref, !new:, !apply: 等）
    
    参数:
        value: 要解析的值
        config: 完整的配置字典
    
    Returns:
        解析后的值
    """
    # 处理 !ref <key> 引用（TaggedScalar 已经被转换）
    if isinstance(value, str) and value.startswith("!ref <") and value.endswith(">"):
        ref_key = value[6:-1].strip()  # 移除 "!ref <" 和 ">"
        
        # 递归解析嵌套引用
        if ref_key in config:
            resolved = parse_yaml_tags(config[ref_key], config)
            return resolved
        else:
            return value
    
    # 处理 <key> 形式的引用（简化版引用语法）
    elif isinstance(value, str) and value.startswith("<") and value.endswith(">"):
        ref_key = value[1:-1].strip()  # 移除 "<" 和 ">"
        
        # 递归解析嵌套引用
        if ref_key in config:
            resolved = parse_yaml_tags(config[ref_key], config)
            return resolved
        else:
            return value
    
    # 处理 !apply:function [args] 或 !new:class {params}
    elif isinstance(value, str) and (value.startswith("!apply:") or value.startswith("!new:")):
        # 保留原始字符串，不实例化，只保留参数结构
        return value
    
    # 处理 !PLACEHOLDER
    elif isinstance(value, str) and value == "!PLACEHOLDER":
        return "!PLACEHOLDER"
    
    # 处理包含 <key> 的字符串（如 "results/ecapa_augment/<seed>"）
    elif isinstance(value, str) and "<" in value and ">" in value:
        # 简单的字符串替换
        result = value
        import re
        # 查找所有 <key> 模式
        matches = re.findall(r'<([^>]+)>', value)
        for match in matches:
            if match in config:
                resolved = parse_yaml_tags(config[match], config)
                result = result.replace(f"<{match}>", str(resolved))
        return result
    
    # 处理 TaggedScalar 类型
    elif hasattr(value, 'value'):
        return parse_yaml_tags(value.value, config)
    
    # 处理字典
    elif isinstance(value, dict):
        resolved_dict = {}
        for key, val in value.items():
            resolved_dict[key] = parse_yaml_tags(val, config)
        return resolved_dict
    
    # 处理列表
    elif isinstance(value, list):
        return [parse_yaml_tags(item, config) for item in value]
    
    # 其他类型直接返回
    else:
        return value



def parse_training_log(train_log_path: Path) -> tuple:
    """
    解析训练日志文件，提取每个epoch的数据
    
    参数:
      train_log_path: 训练日志文件路径
      
    Returns:
      (epoch_data, final_metrics) 元组
    """
    epoch_data = []
    final_metrics = {}
    
    if not train_log_path.exists():
        return epoch_data, final_metrics
    
    try:
        with open(train_log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # 格式: epoch: 10, lr: 7.96e-05 - train loss: 2.74e-01 - valid loss: 2.47e-01, valid ErrorRate: 4.89e-03
        log_pattern = re.compile(r'epoch:\s*(\d+),\s*lr:\s*([\d.e+-]+)\s*-\s*train loss:\s*([\d.e+-]+)\s*-\s*valid loss:\s*([\d.e+-]+),\s*valid ErrorRate:\s*([\d.e+-]+)')
        
        for line in log_content.split('\n'):
            match = log_pattern.search(line)
            if match:
                epoch_data.append({
                    'epoch': int(match.group(1)),
                    'lr': float(match.group(2)),
                    'train_loss': float(match.group(3)),
                    'valid_loss': float(match.group(4)),
                    'valid_error_rate': float(match.group(5))
                })
        
        # 提取最终指标
        if epoch_data:
            final_epoch = epoch_data[-1]
            final_metrics = {
                'final_epoch': final_epoch['epoch'],
                'final_lr': final_epoch['lr'],
                'final_train_loss': final_epoch['train_loss'],
                'final_valid_loss': final_epoch['valid_loss'],
                'final_valid_error_rate': final_epoch['valid_error_rate'],
                'total_epochs': len(epoch_data)
            }
            
            # 找出最佳epoch
            best_epoch = min(epoch_data, key=lambda x: x['valid_error_rate'])
            final_metrics['best_epoch'] = best_epoch['epoch']
            final_metrics['best_valid_loss'] = best_epoch['valid_loss']
            final_metrics['best_error_rate'] = best_epoch['valid_error_rate']
    except Exception as e:
        print(f"⚠️ 解析训练日志失败: {e}")
    
    return epoch_data, final_metrics


def create_experiment_record(experiment_id: str, start_time: datetime, duration: float, status: str, 
                            config: dict, **kwargs) -> dict:
    """
    创建实验记录字典（统一入口）
    
    参数:
      start_time: 实验开始时间
      duration: 实验持续时间（秒）
      status: 实验状态 ('success' 或 'failed')
      config: 配置参数字典
      **kwargs: 其他要添加的字段（如 final_metrics, error 等）
    
    Returns:
      实验记录字典
    """
    if experiment_id is None:
        experiment_id = start_time.strftime("%Y%m%d_%H%M%S")
    
    record = {
        "experiment_id": experiment_id,
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "status": status,
        "config": {
            "lr": config.get("lr"),
            "batch_size": config.get("batch_size"),
            "number_of_epochs": config.get("number_of_epochs"),
            "step_size": config.get("step_size"),
            "seed": config.get("seed")
        }
    }
    
    # 添加额外字段
    record.update(kwargs)
    
    return record


def save_experiment_record(experiment_record) -> None:
    """
    保存实验记录到历史文件。

    参数:
      experiment_record: 包含实验信息的字典
    """
    try:
        history = []
        
        # 读取现有的实验历史
        if EXPERIMENTS_FILE.exists():
            try:
                with open(EXPERIMENTS_FILE, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # 只有文件不为空时才解析
                        history = json.loads(content)
                    else:
                        print(f"ℹ️  实验历史文件为空，创建新记录")
            except json.JSONDecodeError as e:
                print(f"⚠️  实验历史文件格式错误，将创建新记录: {e}")
                history = []
            except Exception as e:
                print(f"⚠️  读取实验历史文件失败: {e}")
                history = []
        
        # 添加新实验记录
        history.append(experiment_record)
        
        # 保存到文件
        with open(EXPERIMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 实验记录已保存: {experiment_record.get('experiment_id', 'Unknown')}")
    except Exception as e:
        print(f"⚠️ 保存实验记录失败: {str(e)}")
        import traceback
        traceback.print_exc()

# list of tools
tools = [
    modify_config, 
    run_training, 
    run_evaluation, 
    read_config, 
    get_training_logs, 
    backup_config,
    view_experiment_history,
    get_best_experiment,
    analyze_training_trends,
    get_experiment_details,
    compare_experiments
]

def load_system_prompt(prompt_path=SYSTEM_PROMPT_PATH) -> str:
    """加载系统提示词"""
    try:
        prompt_path1 = Path(prompt_path)
        if prompt_path1.exists():
            with open(prompt_path1, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return "你是一个声纹识别模型超参数优化专家。"
    except Exception as e:
        print(f"⚠️ 加载系统提示词失败: {e}")
        return "你是一个声纹识别模型超参数优化专家。"

def create_agent():
    """创建带有系统提示词的智能体"""
    system_prompt = load_system_prompt()
    
    # 使用 LangChain 1.0 最新 API
    try:
        from langchain.agents import create_agent
        
        # 直接创建并返回 agent（LangChain 1.0 新方式）
        agent = create_agent(
            model=llm,  # 使用 ZhipuAI 模型
            tools=tools,
            system_prompt=system_prompt,
        )
        
        return agent
        
    except Exception as e:
        print(f"无法使用标准 Agent API: {e}")
        


def main():
    # """主程序入口"""
    # print("=" * 80)
    # print("ECAPA-TDNN 超参数优化智能体系统")
    # print("=" * 80)
    # print()
    # agent = create_agent()
    
    
    
    
    # print("load_config...")
    # result = load_config(config_path=CONFIG_PATH)
    # print(type(result), result)
    # re = format_nested_dict(result)
    # print(type(re), re)
    # 测试成功，加载配置完全正确，并且能够解析引用
    
    
    
    agent = create_agent()
    
    
    for chunk in agent.stream({
        "messages": [{"role": "user", "content": "优化ECAPA-TDNN模型的超参数，目标是让ERR降到4%以下"}]
    }):
        print(chunk)
    
    
    
    # run_training_result = run_training.invoke({"experiment_id": "111"})
    #测试完成，能够调用train函数并且可以进行训练

    # result = analyze_training_trends.invoke(input={"experiment_id": "20260326_092044"})
    # print(result)



if __name__ == "__main__":
    main()
