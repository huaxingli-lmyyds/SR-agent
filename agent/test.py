import os
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

print(llm.invoke("你好，世界！"))