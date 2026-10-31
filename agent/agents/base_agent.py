"""Shared advisory-model and LangGraph agent boundaries."""

from __future__ import annotations

import os
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional

from .advisory_response import (
    decode_json_response, extract_message_content, response_diagnostics,
    correction_prompt, validate_advice_response,
    exception_diagnostics, retryable_request_error, unsupported_feature, trial_ids_from_evidence,
)


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
            import httpx
            mode = os.getenv("SR_AGENT_LLM_PROXY_MODE", "environment")
            if mode not in {"environment", "direct"}:
                raise ValueError("SR_AGENT_LLM_PROXY_MODE must be environment or direct")
            # Dedicated clients avoid sharing a cached transport across advisors.
            # Direct mode is explicit; never bypass an environment proxy on error.
            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                base_url=api_base,
                timeout=300,
                max_retries=0,
                http_client=httpx.Client(trust_env=mode == "environment"),
                http_async_client=httpx.AsyncClient(trust_env=mode == "environment"),
            )
        return self._llm

    @staticmethod
    def _extract_message_content(message: Any) -> str:
        return extract_message_content(message)

    @contextmanager
    def _advisory_session(self):
        """One deadline covers tool rounds, transport retries and format repair."""
        outer = getattr(self, "_advisory_deadline", None)
        if outer is None:
            self._advisory_deadline = time.monotonic() + 300.0
            self._advisor_transport_events = []
            self._advisor_evidence_trial_ids = set()
        try:
            yield
        finally:
            if outer is None:
                self._advisory_deadline = None

    def _advisory_remaining(self):
        resources = getattr(self, "_resource_budget", None)
        remaining = resources.check()["wall_seconds_remaining"] if resources else None
        local = max(0.0, getattr(self, "_advisory_deadline", None) - time.monotonic())
        return min(local, remaining) if remaining is not None else local

    def _json_model(self, model):
        if getattr(getattr(model, "bound", model), "root_client", None) is not None:
            if not getattr(self, "_json_mode_unsupported", False):
                return model.bind(response_format={"type": "json_object"})
        return model

    def _request_json_advice(self, prompt: str, *, validator=None) -> dict:
        with self._advisory_session():
            original_prompt = prompt
            self._advisor_format_events = []
            failures = []
            for attempt in range(2):
                response = None
                stage = "request"
                try:
                    response = self._invoke_advisory_model(self._json_model(self.llm), prompt)
                    stage = "json_decode"
                    value = decode_json_response(response)
                    stage = "schema_validation"
                    validate_advice_response(value, original_prompt)
                    if validator is not None:
                        validator(value)
                    self._advisor_format_events.append({"attempt": attempt + 1, "status": "success"})
                    return value
                except Exception as exc:
                    from agent.hpo.resources import ResourceBudgetExhausted
                    if isinstance(exc, ResourceBudgetExhausted):
                        raise
                    diagnostics = response_diagnostics(response, exc, stage=stage)
                    failures.append(diagnostics)
                    self._advisor_format_events.append({"attempt": attempt + 1, "status": "failed", **diagnostics})
                    if stage == "request" or attempt == 1 or self._advisory_remaining() <= 1:
                        return {"advice_error": diagnostics["error"],
                                "advice_diagnostics": {**diagnostics, "response_attempts": failures}}
                    prompt = correction_prompt(original_prompt, diagnostics)

    def _invoke_with_readonly_tools(
        self,
        prompt: str,
        tools: List[Any],
        *,
        max_tool_rounds: int = 2,
    ) -> Any:
        """Run a bounded read-only tool loop, then require a final model answer."""
        with self._advisory_session():
            return self._invoke_readonly_rounds(prompt, tools, max_tool_rounds=max_tool_rounds)

    def _invoke_readonly_rounds(self, prompt, tools, *, max_tool_rounds):
        if not tools or not hasattr(self.llm, "bind_tools"):
            return self._invoke_advisory_model(self._json_model(self.llm), prompt)
        try:
            from langchain_core.messages import HumanMessage, ToolMessage

            # The SDK auto-parser used for response_format requires strict tools.
            if getattr(self.llm, "root_client", None) is not None:
                bound = self.llm.bind_tools(tools, strict=True)
            else:
                bound = self.llm.bind_tools(tools)
            bound = self._json_model(bound)
            messages: List[Any] = [HumanMessage(content=prompt)]
            by_name = {getattr(item, "name", ""): item for item in tools}
            for _ in range(max(int(max_tool_rounds), 0)):
                response = self._invoke_advisory_model(bound, messages)
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
                            raw_result = selected.invoke(call.get("args") or {})
                            result = raw_result if isinstance(raw_result, str) else json.dumps(raw_result, default=str)
                            try:
                                self._advisor_evidence_trial_ids.update(trial_ids_from_evidence(json.loads(result)))
                            except ValueError:
                                pass
                        except Exception as exc:
                            result = f"{type(exc).__name__}: {exc}"
                    messages.append(ToolMessage(content=result, tool_call_id=call_id))
            messages.append(HumanMessage(
                content=(
                    "Using the read-only evidence already collected, return the final raw JSON "
                    "object required by the original schema and phase-specific response_contract. "
                    "Include every required key and preserve locked values. Do not wrap the object "
                    "in schema or data, narrate tool use, or call another tool. If evidence remains "
                    "insufficient, return the permitted no-change response."
                )
            ))
            return self._invoke_advisory_model(self._json_model(self.llm), messages)
        except Exception as exc:
            from agent.hpo.resources import ResourceBudgetExhausted
            if isinstance(exc, ResourceBudgetExhausted):
                raise
            if not unsupported_feature(exc, "tool"):
                raise
            return self._invoke_advisory_model(self._json_model(self.llm), prompt)

    def _invoke_advisory_model(self, model: Any, prompt: Any) -> Any:
        with self._advisory_session():
            for attempt in range(3):
                remaining = self._advisory_remaining()
                if remaining <= 0:
                    raise TimeoutError("advisor consultation deadline exceeded")
                current = model
                base = getattr(current, "bound", current)
                client = getattr(base, "root_client", None)
                if client is not None and hasattr(client, "with_options"):
                    client = client.with_options(max_retries=0)
                    copied = base.model_copy(update={"root_client": client, "client": client.chat.completions})
                    current = copied.bind(**getattr(current, "kwargs", {}))
                if hasattr(current, "bind"):
                    current = current.bind(timeout=min(120.0, remaining))
                started = time.monotonic()
                try:
                    response = current.invoke(prompt)
                except Exception as exc:
                    self._advisor_transport_events.append({
                        "attempt": attempt + 1, "status": "failed",
                        "duration_seconds": time.monotonic() - started,
                        **exception_diagnostics(exc),
                    })
                    remaining = self._advisory_remaining()
                    if attempt < 2 and unsupported_feature(exc, "response_format"):
                        self._json_mode_unsupported = True
                        kwargs = dict(getattr(model, "kwargs", {}))
                        kwargs.pop("response_format", None)
                        model = base.bind(**kwargs)
                        continue
                    delay = float(2 ** attempt)
                    if attempt == 2 or not retryable_request_error(exc) or remaining <= delay:
                        raise
                    time.sleep(delay)
                    continue
                self._advisor_transport_events.append({
                    "attempt": attempt + 1, "status": "success",
                    "duration_seconds": time.monotonic() - started,
                })
                if self._advisory_remaining() <= 0:
                    raise TimeoutError("advisor consultation deadline exceeded")
                return response


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
