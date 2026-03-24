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
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY")
os.environ["ZHIU_API_BASE_URL"] = os.getenv("ZHIU_API_BASE_URL")

from langchain_community.chat_models import ChatZhipuAI

llm = ChatZhipuAI(model="glm-4.7", temperature=0.2, max_tokens=2000)

llm.invoke("你好，世界！")