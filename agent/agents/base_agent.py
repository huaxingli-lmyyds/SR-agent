"""Shared advisory-model and LangGraph agent boundaries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional


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
                timeout=300,
                max_retries=2,
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

    def _invoke_with_readonly_tools(
        self,
        prompt: str,
        tools: List[Any],
        *,
        max_tool_rounds: int = 2,
    ) -> Any:
        """Run a bounded read-only tool loop, then require a final model answer."""
        if not tools or not hasattr(self.llm, "bind_tools"):
            return self.llm.invoke(prompt)
        try:
            from langchain_core.messages import HumanMessage, ToolMessage

            bound = self.llm.bind_tools(tools)
            messages: List[Any] = [HumanMessage(content=prompt)]
            by_name = {getattr(item, "name", ""): item for item in tools}
            for _ in range(max(int(max_tool_rounds), 0)):
                response = bound.invoke(messages)
                tool_calls = list(getattr(response, "tool_calls", None) or [])
                if not tool_calls:
                    return response
                messages.append(response)
                for call in tool_calls:
                    name = str(call.get("name") or "")
                    call_id = str(call.get("id") or name or "tool_call")
                    selected = by_name.get(name)
                    if selected is None:
                        result = f"read-only tool is not available: {name}"
                    else:
                        try:
                            result = str(selected.invoke(call.get("args") or {}))
                        except Exception as exc:
                            result = f"{type(exc).__name__}: {exc}"
                    messages.append(ToolMessage(content=result, tool_call_id=call_id))
            messages.append(HumanMessage(
                content=(
                    "Using the read-only evidence already collected, return the final raw JSON "
                    "object required by the original schema. Do not call another tool."
                )
            ))
            return self.llm.invoke(messages)
        except Exception:
            # Provider/tool-calling incompatibility must not disable the advisor.
            return self.llm.invoke(prompt)


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
