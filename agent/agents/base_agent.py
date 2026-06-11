"""
Shared base class for LangChain agents in this project.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Any

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

    @classmethod
    def _extract_execution_trace(cls, result: Any) -> List[Dict[str, Any]]:
        """Normalize legacy intermediate steps and message-based tool calls."""
        if not isinstance(result, dict):
            return []

        trace: List[Dict[str, Any]] = []
        for step in result.get("intermediate_steps") or []:
            normalized = cls._normalize_legacy_step(step)
            if normalized:
                trace.append(normalized)

        calls: Dict[str, Dict[str, Any]] = {}
        for message in result.get("messages") or []:
            tool_calls = cls._message_value(message, "tool_calls") or []
            if not tool_calls:
                additional = cls._message_value(message, "additional_kwargs") or {}
                if isinstance(additional, dict):
                    tool_calls = additional.get("tool_calls") or []

            for call in tool_calls:
                normalized = cls._normalize_tool_call(call)
                call_id = normalized.get("tool_call_id") or f"call_{len(calls)}"
                normalized["tool_call_id"] = call_id
                calls[call_id] = normalized
                trace.append(normalized)

            call_id = cls._message_value(message, "tool_call_id")
            if call_id:
                output = cls._extract_message_content(message)
                if call_id in calls:
                    calls[call_id]["output"] = output
                else:
                    normalized = {
                        "tool": cls._message_value(message, "name") or "unknown_tool",
                        "input": {},
                        "output": output,
                        "tool_call_id": call_id,
                    }
                    calls[call_id] = normalized
                    trace.append(normalized)

        return trace

    @staticmethod
    def _message_value(message: Any, key: str) -> Any:
        if isinstance(message, dict):
            return message.get(key)
        return getattr(message, key, None)

    @classmethod
    def _normalize_legacy_step(cls, step: Any) -> Optional[Dict[str, Any]]:
        if isinstance(step, dict) and "tool" in step:
            return {
                "tool": str(step.get("tool") or "unknown_tool"),
                "input": step.get("input") or {},
                "output": step.get("output", ""),
                "tool_call_id": step.get("tool_call_id"),
            }
        if not isinstance(step, (tuple, list)) or not step:
            return None
        action = step[0]
        return {
            "tool": str(getattr(action, "tool", "unknown_tool")),
            "input": getattr(action, "tool_input", {}) or {},
            "output": step[1] if len(step) > 1 else "",
            "tool_call_id": getattr(action, "tool_call_id", None),
        }

    @classmethod
    def _normalize_tool_call(cls, call: Any) -> Dict[str, Any]:
        name = cls._message_value(call, "name")
        arguments = cls._message_value(call, "args")
        call_id = cls._message_value(call, "id")
        function = cls._message_value(call, "function")
        if isinstance(function, dict):
            name = name or function.get("name")
            arguments = arguments if arguments is not None else function.get("arguments")
        if isinstance(arguments, str):
            try:
                import json
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                pass
        return {
            "tool": str(name or "unknown_tool"),
            "input": arguments or {},
            "output": "",
            "tool_call_id": call_id,
        }
