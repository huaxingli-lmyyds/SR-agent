import os
import dotenv
import yaml
from langchain_core.tools import tool

# create an openai llm
from langchain_core.messages import SystemMessage

# load environment variables from .env file
dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY")
os.environ["ZHIU_API_BASE_URL"] = os.getenv("ZHIU_API_BASE_URL")

from langchain_community.chat_models import ChatZhipuAI


# define the agent
# hpo_agent = ChatZhipuAI(model="glm-4.7", 
#                         temperature=0.2,
#                         max_tokens=20)

# the path to the ECAPA-TDNN configuration file and log file
CONFIG_PATH = "../configs/train_ecapa_tdnn.yaml"


#define the tools
@tool
def modify_config(config_json: str, persist: bool = True) -> str:
    """
    使用 JSON 形式的配置更新 ECAPA-TDNN YAML 文件。

    参数:
      config_json: JSON 字符串或 dict，表示要更新的字段
        例如: '{"lr": 0.02,"classifier": {"input_size": 1000}}'
      persist: 是否写回文件（True）或仅预览（False），默认 True。

    Returns:
      操作结果描述
    """
    try:
        if isinstance(config_json, str):
            import json
            updates = json.loads(config_json)
        elif isinstance(config_json, dict):
            updates = config_json
        else:
            return "❌ config_json 必须是 JSON 字符串或 dict 类型"

        try:
            from ruamel.yaml import YAML
        except ImportError:
            return "❌ 需要安装 ruamel.yaml: pip install ruamel.yaml"

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

        return f"✅ 配置已更新: {updates} (persist={persist})"

    except Exception as e:
        return f"❌ 修改配置失败: {str(e)}"


print(modify_config.invoke('{"lr": 0.02,"classifier": {"input_size": 1000}}', persist=True))