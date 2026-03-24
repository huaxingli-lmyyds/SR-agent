import os
import re
from tabnanny import verbose
import dotenv
import yaml
import subprocess
import json
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

# load environment variables from .env file
dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())
os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("ZHIU_API_BASE_URL")


from langchain_openai import ChatOpenAI

# the path to the ECAPA-TDNN configuration file
CONFIG_PATH = "../configs/train_ecapa_tdnn.yaml"
TRAIN_SCRIPT = "../recipes/voxceleb/train_speaker_embeddings.py"
EVAL_SCRIPT = "../recipes/voxceleb/speaker_verification_cosine.py"
SYSTEM_PROMPT_PATH = "prompts/hpo_prompt.txt"

# 实验记录目录和文件
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
EXPERIMENTS_FILE = EXPERIMENTS_DIR / "experiments_history.json"
EXPERIMENTS_CONFIGS_DIR = EXPERIMENTS_DIR / "configs"

# 确保实验目录存在
EXPERIMENTS_DIR.mkdir(exist_ok=True)
EXPERIMENTS_CONFIGS_DIR.mkdir(exist_ok=True)

# create the LLM
llm = ChatOpenAI(model="GLM-4.7", temperature=0.2, max_tokens=2000)

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

@tool
def run_training() -> str:
    """
    运行 ECAPA-TDNN 模型的训练脚本。
    训练完成后会自动保存实验记录，包括配置、训练日志和结果。

    Returns:
      训练输出或错误信息
    """
    try:
        from datetime import datetime
        import shutil
        import re
        
        # 读取当前配置
        current_config = load_yaml_config(CONFIG_PATH)
        
        # 记录开始时间
        start_time = datetime.now()
        
        # 运行训练
        result = subprocess.run(
            ["python", TRAIN_SCRIPT, CONFIG_PATH],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # 记录结束时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            # 训练成功，提取训练结果
            training_output = result.stdout
            
            # 尝试从输出中提取性能指标
            error_rate = None
            eer = None
            accuracy = None
            
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
            
            # 创建实验记录
            experiment_id = start_time.strftime("%Y%m%d_%H%M%S")
            experiment_record = {
                "experiment_id": experiment_id,
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "status": "success",
                "config": {
                    "lr": current_config.get("lr"),
                    "batch_size": current_config.get("batch_size"),
                    "number_of_epochs": current_config.get("number_of_epochs"),
                    "step_size": current_config.get("step_size"),
                    "seed": current_config.get("seed")
                },
                "results": {
                    "error_rate": error_rate,
                    "eer": eer,
                    "accuracy": accuracy,
                    "final_output": training_output[-1000:]  # 保存最后1000字符
                },
                "training_log_path": str(current_config.get('train_log', ''))
            }
            
            # 保存配置副本
            config_backup_path = EXPERIMENTS_CONFIGS_DIR / f"config_{experiment_id}.yaml"
            shutil.copy2(CONFIG_PATH, config_backup_path)
            experiment_record["config_backup_path"] = str(config_backup_path)
            
            # 保存实验记录到历史文件
            save_experiment_record(experiment_record)
            
            return f"""✅ 训练完成！
实验ID: {experiment_id}
训练时长: {duration:.2f} 秒
性能指标:
  - 错误率: {error_rate if error_rate else 'N/A'}
  - 等错误率 (EER): {eer if eer else 'N/A'}
  - 准确率: {accuracy if accuracy else 'N/A'}
  
配置和结果已保存到 experiments/ 目录
最后输出: {training_output[-200:]}"""
        else:
            # 训练失败，记录失败信息
            experiment_id = start_time.strftime("%Y%m%d_%H%M%S")
            experiment_record = {
                "experiment_id": experiment_id,
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "status": "failed",
                "config": {
                    "lr": current_config.get("lr"),
                    "batch_size": current_config.get("batch_size"),
                    "number_of_epochs": current_config.get("number_of_epochs"),
                    "step_size": current_config.get("step_size"),
                    "seed": current_config.get("seed")
                },
                "error": result.stderr[-1000:]
            }
            save_experiment_record(experiment_record)
            
            return f"❌ 训练失败 (实验ID: {experiment_id})\n错误信息: {result.stderr}"
    except Exception as e:
        return f"❌ 运行训练失败: {str(e)}"

@tool
def run_evaluation() -> str:
    """
    运行 ECAPA-TDNN 模型的评估脚本，使用训练后的模型。

    Returns:
      评估结果或错误信息
    """
    try:
        # 需要一个评估配置文件，暂时使用默认的
        eval_config = "../recipes/voxceleb/hparams/verification_ecapa.yaml"
        result = subprocess.run(
            ["python", EVAL_SCRIPT, eval_config],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            return f"评估完成: {result.stdout[-500:]}"  # 最后500字符
        else:
            return f"评估失败: {result.stderr}"
    except Exception as e:
        return f"运行评估失败: {str(e)}"

@tool
def read_config() -> str:
    """
    读取当前 ECAPA-TDNN 配置文件的内容。

    Returns:
      配置文件的 YAML 内容或错误信息
    """
    try:
        config = load_yaml_config(CONFIG_PATH)
        
        # 只返回关键配置部分，避免信息过载
        key_params = [
            'lr', 'batch_size', 'number_of_epochs', 'step_size',
            'embedding_model', 'compute_cost', 'opt_class', 'seed'
        ]
        
        summary = "当前关键配置:\n"
        for key in key_params:
            if key in config:
                value = config[key]
                # 处理嵌套字典
                if isinstance(value, dict) and key == 'compute_cost':
                    if 'loss_fn' in value and isinstance(value['loss_fn'], dict):
                        loss_params = {k: v for k, v in value['loss_fn'].items() if not k.startswith('_')}
                        summary += f"  {key}: {loss_params}\n"
                    else:
                        summary += f"  {key}: {value}\n"
                else:
                    summary += f"  {key}: {value}\n"
        
        return summary
    except Exception as e:
        return f"读取配置失败: {str(e)}"

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
            summary += f"   配置: lr={exp['config'].get('lr')}, batch_size={exp['config'].get('batch_size')}\n"
            
            if exp['status'] == 'success' and exp.get('results'):
                results = exp['results']
                summary += f"   结果: "
                if results.get('error_rate'):
                    summary += f"错误率={results['error_rate']:.4f} "
                if results.get('eer'):
                    summary += f"EER={results['eer']:.4f} "
                if results.get('accuracy'):
                    summary += f"准确率={results['accuracy']:.4f} "
                summary += "\n"
            
            summary += "\n"
        
        return summary
    except Exception as e:
        return f"❌ 读取实验历史失败: {str(e)}"

@tool
def get_best_experiment(metric: str = "eer") -> str:
    """
    找出最佳实验（根据指定指标）。

    参数:
      metric: 优化指标，可选 'eer' (等错误率，越小越好) 或 'accuracy' (准确率，越大越好)，默认 'eer'

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
        if metric == "eer":
            # EER 越小越好
            best_exp = min(successful_experiments, 
                         key=lambda x: x['results'].get('eer', float('inf')) if x['results'].get('eer') is not None else float('inf'))
            metric_name = "等错误率 (EER)"
        elif metric == "accuracy":
            # 准确率越大越好
            best_exp = max(successful_experiments, 
                         key=lambda x: x['results'].get('accuracy', 0) if x['results'].get('accuracy') is not None else 0)
            metric_name = "准确率"
        else:
            return f"❌ 不支持的指标: {metric}。请使用 'eer' 或 'accuracy'。"
        
        if best_exp['results'].get(metric) is None:
            return f"📋 没有找到包含 {metric} 指标的实验记录。"
        
        summary = f"\n🏆 最佳实验 (按 {metric_name}):\n"
        summary += "=" * 80 + "\n\n"
        summary += f"实验 ID: {best_exp['experiment_id']}\n"
        summary += f"时间: {best_exp['timestamp']}\n"
        summary += f"训练时长: {best_exp['duration_seconds']:.2f} 秒\n"
        summary += f"\n配置:\n"
        for key, value in best_exp['config'].items():
            summary += f"  {key}: {value}\n"
        summary += f"\n性能指标:\n"
        summary += f"  {metric_name}: {best_exp['results'][metric]:.4f}\n"
        if best_exp['results'].get('error_rate'):
            summary += f"  错误率: {best_exp['results']['error_rate']:.4f}\n"
        if best_exp['results'].get('eer') and metric != 'eer':
            summary += f"  EER: {best_exp['results']['eer']:.4f}\n"
        if best_exp['results'].get('accuracy') and metric != 'accuracy':
            summary += f"  准确率: {best_exp['results']['accuracy']:.4f}\n"
        summary += f"\n配置文件: {best_exp.get('config_backup_path', 'N/A')}\n"
        
        return summary
    except Exception as e:
        return f"❌ 查找最佳实验失败: {str(e)}"

def load_yaml_config(config_path):
    """加载 YAML 配置文件"""
    try:
        return parse_yaml_simpler(config_path)
    except Exception as e:
        print(f"⚠️ 加载 YAML 配置失败: {e}")
        return {}
    
    
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
        return parse_yaml_simpler(config_path)
    except Exception as e:
        print(f"⚠️ hyperpyyaml 加载失败: {e}")
        print("尝试使用备用解析方法...")
        return parse_yaml_simpler(config_path)


def parse_yaml_simpler(config_path):
    """简单解析 YAML 文件，提取关键值"""
    import re
    
    config = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 简单的键值对解析
    for line in lines:
        line = line.strip()
        # 跳过注释和空行
        if not line or line.startswith('#'):
            continue
        
        # 跳过包含特殊标签的行
        if '!apply:' in line or '!ref' in line or '!new:' in line or '!name:' in line:
            continue
        
        # 简单的键: 值匹配
        if ':' in line and not line.startswith('-'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                
                # 尝试解析不同类型的值
                if value:
                    # 布尔值
                    if value.lower() in ['true', 'false']:
                        config[key] = value.lower() == 'true'
                    # 整数
                    elif value.isdigit():
                        config[key] = int(value)
                    # 浮点数
                    elif re.match(r'^\d+\.\d+$', value):
                        config[key] = float(value)
                    # 字符串
                    else:
                        config[key] = value
    
    return config

def save_experiment_record(experiment_record):
    """
    保存实验记录到历史文件。

    参数:
      experiment_record: 包含实验信息的字典
    """
    try:
        # 读取现有历史
        history = []
        if EXPERIMENTS_FILE.exists():
            with open(EXPERIMENTS_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # 添加新实验
        history.append(experiment_record)
        
        # 保存回文件
        with open(EXPERIMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 实验记录已保存: {experiment_record['experiment_id']}")
    except Exception as e:
        print(f"⚠️ 保存实验记录失败: {str(e)}")

# list of tools
tools = [
    modify_config, 
    run_training, 
    run_evaluation, 
    read_config, 
    get_training_logs, 
    backup_config,
    view_experiment_history,
    get_best_experiment
]

def load_system_prompt():
    """加载系统提示词"""
    try:
        prompt_path = Path(__file__).parent / SYSTEM_PROMPT_PATH
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
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
        print("尝试使用简化的实现...")
        
        # 简化版本：直接调用工具
        return SimpleAgent(system_prompt, tools, llm)


class SimpleAgent:
    """简化的智能体实现"""
    def __init__(self, system_prompt, tools, llm):
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm
        self.tool_descriptions = "\n".join([
            f"{tool.name}: {tool.description}" for tool in tools
        ])
    
    def invoke(self, inputs):
        """执行智能体任务"""
        user_input = inputs.get("input", "")
        
        # 构建提示词
        prompt = f"""{self.system_prompt}

可用工具:
{self.tool_descriptions}

用户请求: {user_input}

请选择合适的工具来完成任务。如果需要修改配置，请提供JSON格式的参数。
"""
        
        # 调用 LLM
        response = self.llm.invoke(prompt)
        
        # 解析响应并执行工具
        result = self._parse_and_execute(response.content, user_input)
        
        return {"output": result}
    
    def _parse_and_execute(self, response, original_input):
        """解析LLM响应并执行工具"""
        # 简单的关键词匹配
        if "读取配置" in original_input or "read_config" in original_input.lower():
            return self.tools["read_config"].invoke({})
        elif "备份" in original_input or "backup" in original_input.lower():
            return self.tools["backup_config"].invoke({})
        elif "实验历史" in original_input or "history" in original_input.lower():
            return self.tools["view_experiment_history"].invoke({})
        elif "最佳实验" in original_input or "best" in original_input.lower():
            # 检查是否指定了指标
            if "eer" in original_input.lower():
                return self.tools["get_best_experiment"].invoke({"metric": "eer"})
            elif "accuracy" in original_input.lower() or "准确率" in original_input:
                return self.tools["get_best_experiment"].invoke({"metric": "accuracy"})
            else:
                return self.tools["get_best_experiment"].invoke({})
        elif "学习率" in original_input or "lr" in original_input.lower():
            # 尝试提取学习率值
            import re
            lr_match = re.search(r'(\d+\.?\d*)', original_input)
            if lr_match:
                lr_value = float(lr_match.group(1))
                return self.tools["modify_config"].invoke({
                    "config_json": '{"lr": ' + str(lr_value) + '}',
                    "persist": True
                })
        elif "批次" in original_input or "batch" in original_input.lower():
            import re
            batch_match = re.search(r'(\d+)', original_input)
            if batch_match:
                batch_value = int(batch_match.group(1))
                return self.tools["modify_config"].invoke({
                    "config_json": '{"batch_size": ' + str(batch_value) + '}',
                    "persist": True
                })
        
        # 默认返回 LLM 响应
        return f"智能体响应: {response}\n\n提示: 尝试使用明确的指令，如'读取配置'、'查看实验历史'或'将学习率调整为0.001'"

def main():
    """主程序入口"""
    print("=" * 80)
    print("ECAPA-TDNN 超参数优化智能体系统")
    print("=" * 80)
    print()
    
    # 创建智能体
    print("正在初始化智能体...")
    agent_executor = create_agent()
    print("智能体初始化完成！")
    print()
    
    # 显示使用说明
    print("使用说明:")
    print("- 输入自然语言指令，例如:")
    print("  '将学习率调整为 0.01'")
    print("  '分析当前配置并给出优化建议'")
    print("  '训练模型并评估性能'")
    print("- 输入 'quit' 或 'exit' 退出程序")
    print("- 输入 'help' 查看更多示例")
    print()
    print("=" * 80)
    print()
    
    # 交互式循环
    while True:
        try:
            user_input = input("请输入指令: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("感谢使用，再见！")
                break
                
            if user_input.lower() == 'help':
                print("\n常用指令示例:")
                print("  基础操作:")
                print("  1. 读取当前配置")
                print("  2. 备份当前配置")
                print()
                print("  配置修改:")
                print("  3. 将学习率调整为 0.001")
                print("  4. 将批次大小调整为 64")
                print()
                print("  训练与评估:")
                print("  5. 训练模型")
                print("  6. 评估模型性能")
                print("  7. 查看训练日志")
                print()
                print("  实验管理:")
                print("  8. 查看实验历史")
                print("  9. 查看最佳实验 (按EER)")
                print("  10. 查看最佳实验 (按准确率)")
                print()
                print("  高级功能:")
                print("  11. 优化超参数以提升性能")
                print("  12. 分析当前配置并给出改进建议")
                print()
                continue
            
            print(f"\n正在处理: {user_input}")
            print("-" * 80)
            
            # 执行智能体（LangChain 1.0 新方式：使用 messages 格式）
            result = agent_executor.invoke({
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            })
            
            print("-" * 80)
            print(f"\n✅ 执行结果:")
            print(result.content)
            print()
            
        except KeyboardInterrupt:
            print("\n\n程序被中断")
            continue
        except Exception as e:
            print(f"\n发生错误: {e}")
            print()
            continue

if __name__ == "__main__":
    main()
