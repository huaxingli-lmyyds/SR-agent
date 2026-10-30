import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from langchain_openai import ChatOpenAI
from openai import APIConnectionError

from agent.agents.base_agent import AdvisoryAgentBase
from agent.agents.hpo_agent import HPOAgent
from agent.agents.advisory_response import decode_json_response, exception_diagnostics
from agent.prompt import load_prompt_template


def response(content='{"ok":true}'):
    return {"id": "test", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


def model(handler):
    return ChatOpenAI(model="fake", api_key="fake", base_url="https://example.invalid/v1",
                      max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_real_sdk_recovers_transport_failure_and_keeps_json_mode(monkeypatch):
    monkeypatch.setattr("agent.agents.base_agent.time.sleep", lambda _: None)
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            raise httpx.ConnectError("proxy connection refused", request=request)
        return httpx.Response(200, json=response())
    agent = AdvisoryAgentBase("fake", 0, 1, False)
    agent._llm = model(handler)
    assert agent._request_json_advice("Return JSON") == {"ok": True}
    assert len(calls) == 2
    assert all(c["response_format"] == {"type": "json_object"} for c in calls)
    events = agent._advisor_transport_events
    assert events[0]["exception_chain"][-1]["type"] == "ConnectError"
    assert events[1]["status"] == "success"
    assert agent._request_json_advice("Return JSON") == {"ok": True}


def test_authentication_is_not_retried():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": "invalid key", "type": "authentication_error"}})
    agent = AdvisoryAgentBase("fake", 0, 1, False)
    agent._llm = model(handler)
    result = agent._request_json_advice("JSON")
    assert result["advice_diagnostics"]["http_status"] == 401
    assert len(calls) == 1


def test_json_mode_fallback_only_on_explicit_unsupported_feature():
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        return httpx.Response(200, json=response())
    agent = AdvisoryAgentBase("fake", 0, 1, False)
    agent._llm = model(handler)
    assert agent._request_json_advice("JSON") == {"ok": True}
    assert len(calls) == 2
    assert agent._request_json_advice("JSON") == {"ok": True}
    assert "response_format" not in calls[-1]


def test_network_failure_does_not_restart_whole_tool_loop(monkeypatch):
    monkeypatch.setattr("agent.agents.base_agent.time.sleep", lambda _: None)
    calls = []
    class Model:
        def bind_tools(self, tools):
            return self
        def invoke(self, prompt):
            calls.append(prompt)
            raise ConnectionError("unreachable")
    agent = AdvisoryAgentBase("fake", 0, 1, False)
    agent._llm = Model()
    with pytest.raises(ConnectionError):
        agent._invoke_with_readonly_tools("JSON", [SimpleNamespace(name="read")])
    assert len(calls) == 3


def test_format_retry_uses_original_evidence_and_strict_schema():
    agent = object.__new__(HPOAgent)
    payload = deepcopy(load_prompt_template("hpo_strategy_proposal")["examples"][0]["output"])
    calls = []
    def invoke(prompt, **kwargs):
        calls.append(json.loads(prompt))
        if len(calls) == 1:
            return SimpleNamespace(content='{!"action":"keep_strategy"}')
        return SimpleNamespace(content=json.dumps(payload))
    agent._invoke_strategy_model = invoke
    proposal = agent._request_strategy_proposal(json.dumps({"context": {"trial_id": "trial_a"}}), error_code="invalid")
    assert proposal.action == "keep_strategy"
    assert len(calls) == 2
    assert calls[0]["context"] == calls[1]["context"]
    assert calls[1]["response_correction"]["failure_stage"] == "json_decode"
    assert agent._advisor_format_events[0]["status"] == "failed"


def test_malformed_second_response_stays_invalid():
    agent = object.__new__(HPOAgent)
    agent._invoke_strategy_model = lambda *a, **k: SimpleNamespace(content='{!"action":"keep_strategy"}')
    result = agent._request_strategy_proposal('{}', error_code="invalid")
    assert result.action == "invalid_proposal"
    assert len(result.evidence["response_attempts"]) == 2


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_ambiguous_json_is_rejected(text):
    with pytest.raises(ValueError):
        decode_json_response(SimpleNamespace(content=text))


def test_nested_network_cause_is_recorded_without_credentials(monkeypatch):
    monkeypatch.setenv("ZHIPUAI_API_KEY", "sensitive-test-value")
    try:
        try:
            raise httpx.ConnectError("connection refused via https://user:password@proxy.invalid; sensitive-test-value")
        except httpx.ConnectError as cause:
            raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid")) from cause
    except APIConnectionError as exc:
        diagnostic = exception_diagnostics(exc)
    text = json.dumps(diagnostic)
    assert "ConnectError" in text
    assert "sensitive-test-value" not in text
    assert "user:password" not in text


def test_deadline_is_shared_across_retries(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("agent.agents.base_agent.time.monotonic", lambda: now[0])
    monkeypatch.setattr("agent.agents.base_agent.time.sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    calls = []
    class Model:
        def bind(self, **kwargs):
            calls.append(kwargs)
            return self
        def invoke(self, prompt):
            now[0] += 160
            raise ConnectionError("slow failure")
    agent = AdvisoryAgentBase("fake", 0, 1, False)
    with pytest.raises(ConnectionError):
        agent._invoke_advisory_model(Model(), "JSON")
    assert len(calls) == 2
    assert now[0] == 321


def test_valid_json_cannot_hide_corrupted_evidence_reference():
    agent = object.__new__(HPOAgent)
    payload = deepcopy(load_prompt_template("hpo_strategy_proposal")["examples"][0]["output"])
    payload["hypotheses"] = [{"id": "h1", "claim": "observed", "evidence_trial_ids": ["trial_!a"], "expected_signal": {}}]
    agent._invoke_strategy_model = lambda *a, **k: SimpleNamespace(content=json.dumps(payload))
    result = agent._request_strategy_proposal('{"context":{"trial_id":"trial_a"}}', error_code="invalid")
    assert result.action == "invalid_proposal"
    assert result.evidence["failure_stage"] == "schema_validation"
    assert "unobserved Trial IDs" in result.evidence["error"]


def test_overflowing_json_numbers_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        decode_json_response('{"x":1e9999}')


def test_real_sdk_tool_rounds_and_json_mode_work_together():
    from langchain_core.tools import tool
    tool_calls = []
    @tool
    def read_evidence() -> str:
        """Read synthetic evidence."""
        tool_calls.append(1)
        return '{"trial_id":"trial_observed"}'
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            assert body["tools"][0]["function"]["strict"] is True
            return httpx.Response(200, json={"id":"test", "object":"chat.completion", "created":0, "model":"fake",
                "choices":[{"index":0, "finish_reason":"tool_calls", "message":{"role":"assistant", "content":None,
                    "tool_calls":[{"id":"call_1", "type":"function", "function":{"name":"read_evidence", "arguments":"{}"}}]}}]})
        return httpx.Response(200, json=response())
    agent = AdvisoryAgentBase("fake", 0, 1, False)
    agent._llm = model(handler)
    result = agent._invoke_with_readonly_tools("Read evidence and return JSON", [read_evidence])
    assert decode_json_response(result) == {"ok": True}
    assert tool_calls == [1]
    assert len(calls) == 2
    assert "trial_observed" in agent._advisor_evidence_trial_ids
