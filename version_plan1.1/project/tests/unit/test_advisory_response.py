import json
from types import SimpleNamespace

import pytest

from agent.agents.advisory_response import (
    decode_json_response,
    extract_message_content,
)
from agent.agents.hpo_agent import HPOAgent
from agent.agents.base_agent import AdvisoryAgentBase
from agent.hpo.resources import ResourceBudgetExhausted
from agent.prompt import load_prompt_template


@pytest.mark.parametrize(
    "content",
    [
        '{"answer":1}',
        '```json\n{"answer":1}\n```',
        'Analysis first.\n\n```json\n{"answer":1}\n```\nFinal note.',
        'Analysis first.\r\n```JSON\r\n{"answer":1}\r\n```',
        [
            {"type": "text", "text": '{"answer":'},
            {"type": "text", "text": "1}"},
        ],
        [
            {"type": "reasoning", "text": "private reasoning"},
            {"type": "output_text", "text": '{"answer":1}'},
        ],
    ],
)
def test_text_and_block_responses_decode(content):
    assert decode_json_response(SimpleNamespace(content=content)) == {
        "answer": 1
    }


@pytest.mark.parametrize(
    "content",
    [None, "", "[]", "null", 'explanation {"answer":1}', '{"answer":',
     'Analysis\n```json\n{"answer":1}',
     'Analysis\n```json\n{"answer":1}\n```\n```json\n{"answer":2}\n```',
     'Analysis\n```python\n{"answer":1}\n```'],
)
def test_invalid_or_incomplete_responses_are_not_repaired(content):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        decode_json_response(SimpleNamespace(content=content))


def test_failed_proposal_records_bounded_response_and_finish_reason():
    agent = object.__new__(HPOAgent)
    agent._invoke_strategy_model = lambda *args, **kwargs: SimpleNamespace(
        content="x" * 5000,
        response_metadata={"finish_reason": "length"},
    )
    proposal = agent._request_strategy_proposal(
        "prompt", error_code="proposal_parse_error"
    )
    assert proposal.action == "invalid_proposal"
    assert proposal.evidence["failure_stage"] == "json_decode"
    assert proposal.evidence["finish_reason"] == "length"
    assert len(proposal.evidence["response_excerpt"]) == 2000


def test_valid_block_proposal_still_passes_strict_schema_validation():
    agent = object.__new__(HPOAgent)
    payload = load_prompt_template("hpo_strategy_proposal")["examples"][0][
        "output"
    ]
    agent._invoke_strategy_model = lambda *args, **kwargs: SimpleNamespace(
        content=[{"type": "text", "text": json.dumps(payload)}],
    )
    assert (
        agent._request_strategy_proposal(
            "prompt", error_code="parse_error"
        ).action
        == payload["action"]
    )
    agent._invoke_strategy_model = lambda *args, **kwargs: {
        "content": '{"action":"keep_strategy"}'
    }
    proposal = agent._request_strategy_proposal(
        "prompt", error_code="parse_error"
    )
    assert proposal.evidence["failure_stage"] == "schema_validation"


def test_resource_deadline_is_never_swallowed_by_response_parser():
    agent = object.__new__(HPOAgent)

    def expired(*args, **kwargs):
        raise ResourceBudgetExhausted("expired")

    agent._invoke_strategy_model = expired
    with pytest.raises(ResourceBudgetExhausted):
        agent._request_strategy_proposal("prompt", error_code="parse_error")


def test_request_failures_have_separate_diagnostics():
    agent = object.__new__(HPOAgent)

    def fail(*args, **kwargs):
        raise ConnectionError("connection unavailable")

    agent._invoke_strategy_model = fail
    proposal = agent._request_strategy_proposal(
        "prompt", error_code="parse_error"
    )
    assert proposal.evidence["failure_stage"] == "request"
    assert proposal.evidence["response_excerpt"] == ""
    assert extract_message_content(None) == ""


def test_shared_advisor_parser_accepts_fences_and_keeps_failure_diagnostics():
    agent = AdvisoryAgentBase("unused", 0, 1, False)
    agent._llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content='```json\n{"notes":[]}\n```'))
    assert agent._request_json_advice("test") == {"notes": []}
    agent._llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content="bad response"))
    failed = agent._request_json_advice("test")
    assert failed["advice_diagnostics"]["failure_stage"] == "json_decode"
    assert failed["advice_diagnostics"]["response_excerpt"] == "bad response"

    def expired(_):
        raise ResourceBudgetExhausted("expired")

    agent._llm = SimpleNamespace(invoke=expired)
    with pytest.raises(ResourceBudgetExhausted):
        agent._request_json_advice("test")
