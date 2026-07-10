"""Shared advisory-model and LangGraph agent boundaries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def _load_env() -> None:
    import dotenv

    env_path = dotenv.find_dotenv(usecwd=True)
    if not env_path:
        candidate = Path.cwd() / ".env"
        env_path = str(candidate) if candidate.exists() else ""
    dotenv.load_dotenv(dotenv_path=env_path or None)

    api_key = os.getenv("ZHIPUAI_API_KEY")
    api_base = os.getenv("ZHIPUAI_API_BASE_URL")
    if api_base and api_base.rstrip("/").endswith("/chat/completions"):
        raise ValueError(
            "ZHIPUAI_API_BASE_URL must be the OpenAI-compatible API root, "
            "for example https://llmapi.paratera.com/v1; do not include "
            "/chat/completions because the OpenAI client appends that path."
        )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ["OPENAI_BASE_URL"] = api_base


class AdvisoryAgentBase:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_iterations: int,
        verbose: bool,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._llm: Optional[Any] = None

    @property
    def llm(self) -> Any:
        """Create the advisory model only when a workflow explicitly requests it."""
        if self._llm is None:
            from langchain_openai import ChatOpenAI

            _load_env()
            api_base = os.getenv("ZHIPUAI_API_BASE_URL")
            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                base_url=api_base,
                timeout=30,
                max_retries=1,
            )
        return self._llm

    @staticmethod
    def _extract_message_content(message: Any) -> str:
        if message is None:
            return ""
        if isinstance(message, dict):
            content = message.get("content", message)
        else:
            content = getattr(message, "content", message)
        return str(content)


class LangGraphAgent(AdvisoryAgentBase):
    """Uniform request/result boundary for specialized LangGraph agents."""

    action: str = ""

    def execute_task(self, request: Any) -> Any:
        from agent.agents.communication import AgentTaskResult

        if request.action != self.action:
            return AgentTaskResult(
                status="failed",
                error=f"unsupported action: {request.action}; expected: {self.action}",
                request_id=request.request_id,
            )
        try:
            return self.run_workflow(request)
        except Exception as exc:
            return AgentTaskResult(
                status="failed",
                error=str(exc),
                experiment_ids=request.experiment_ids,
                request_id=request.request_id,
            )

    def run_workflow(self, request: Any) -> Any:
        raise NotImplementedError
