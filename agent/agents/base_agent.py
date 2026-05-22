"""
Shared base class for LangChain agents in this project.
"""

from __future__ import annotations

import os
from typing import List, Optional, Any

import dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


def _load_env() -> None:
    dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())
    os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = os.getenv("ZHIU_API_BASE_URL")


class BaseLangChainAgent:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_iterations: int,
        verbose: bool,
    ) -> None:
        _load_env()
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )
        self.tools: List[Any] = []
        self.system_prompt: str = ""
        self.agent = None

    def _build_agent(
        self,
        tools: List[Any],
        system_prompt: str,
        middleware: Optional[Any] = None,
    ) -> Any:
        return create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware,
        )

    def _invoke(self, objective: str) -> Any:
        messages = [{"role": "user", "content": objective}]
        return self.agent.invoke({"messages": messages})

    def _invoke_with_recursion_limit(self, objective: str, recursion_limit: int) -> Any:
        messages = [{"role": "user", "content": objective}]
        return self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": recursion_limit},
        )

    @staticmethod
    def _extract_message_content(message: Any) -> str:
        if message is None:
            return ""
        if isinstance(message, dict):
            content = message.get("content", message)
        else:
            content = getattr(message, "content", message)
        return str(content)
