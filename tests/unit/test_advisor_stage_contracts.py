"""Stage contracts reject dangerous shape mismatches before workflow consumption."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from agent.agents.advisory_response import correction_prompt
from agent.agents.base_agent import AdvisoryAgentBase
from agent.prompt import render_prompt


def data_prompt():
    return render_prompt(
        "data_processing_planning_advice",
        context={
            "dataset_type": "audio",
            "quality_metrics": {"invalid_rows": 2},
            "available_operations": {
                "clean": {
                    "supported_data_types": ["audio"],
                    "parameter_schema": {
                        "enabled": {"type": "boolean"},
                        "protected": {
                            "type": "boolean",
                            "advisor_allowed": False,
                        },
                    },
                }
            },
        },
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "wrapper",
        "missing",
        "string_array",
        "string_parameters",
        "unknown_operation",
        "unknown_parameter",
        "protected",
        "wrong_type",
        "duplicate",
        "too_many",
    ],
)
def test_data_processing_retries_shape_errors_once_with_original_context(
    invalid,
):
    valid = {
        "diagnostics": ["Two invalid rows were reported."],
        "suggested_operations": [
            {
                "operation": "clean",
                "parameters": {"enabled": True},
                "reason": "Address reported invalid rows.",
            }
        ],
        "notes": [],
    }
    bad = deepcopy(valid)
    if invalid == "wrapper":
        bad = {"schema": bad}
    elif invalid == "missing":
        bad.pop("notes")
    elif invalid == "string_array":
        bad["diagnostics"] = "not an array"
    elif invalid == "string_parameters":
        bad["suggested_operations"][0]["parameters"] = "enabled=true"
    elif invalid == "unknown_operation":
        bad["suggested_operations"][0]["operation"] = "invented"
    elif invalid == "unknown_parameter":
        bad["suggested_operations"][0]["parameters"] = {
            "_output_uri": "/tmp/output"
        }
    elif invalid == "protected":
        bad["suggested_operations"][0]["parameters"] = {"protected": True}
    elif invalid == "wrong_type":
        bad["suggested_operations"][0]["parameters"] = {"enabled": "true"}
    elif invalid == "duplicate":
        bad["suggested_operations"] *= 2
    elif invalid == "too_many":
        bad["suggested_operations"] *= 6
    agent = AdvisoryAgentBase("fake", 0, 1, False)
    calls, deadlines = [], []

    def invoke(prompt):
        calls.append(json.loads(prompt))
        deadlines.append(agent._advisory_deadline)
        return SimpleNamespace(
            content=json.dumps(bad if len(calls) == 1 else valid)
        )

    agent._llm = SimpleNamespace(invoke=invoke)
    assert agent._request_json_advice(data_prompt()) == valid
    assert len(calls) == 2
    assert deadlines[0] == deadlines[1]
    assert calls[0]["context"] == calls[1]["context"]
    assert (
        calls[1]["response_correction"]["failure_stage"] == "schema_validation"
    )
    assert [event["status"] for event in agent._advisor_format_events] == [
        "failed",
        "success",
    ]


def test_coordination_bad_reply_never_escapes_as_valid_advice():
    agent = AdvisoryAgentBase("fake", 0, 1, False)
    calls = []

    def invoke(prompt):
        calls.append(prompt)
        return SimpleNamespace(
            content='{"diagnostics":[],"risks":{},"notes":[]}'
        )

    agent._llm = SimpleNamespace(invoke=invoke)
    result = agent._request_json_advice(
        render_prompt("orchestration_coordination_advisor", context={})
    )
    assert len(calls) == 2
    assert "advice_error" in result
    assert "risks" not in result
    assert result["advice_diagnostics"]["failure_stage"] == "schema_validation"


def test_format_correction_keeps_planning_permissions_and_handles_truncation():
    prompt = render_prompt(
        "hpo_strategy_proposal", context={"phase": "study_planning"}
    )
    original = json.loads(prompt)
    corrected = json.loads(
        correction_prompt(
            prompt,
            {
                "failure_stage": "json_decode",
                "error": "truncated",
                "finish_reason": "length",
            },
        )
    )
    assert corrected["response_contract"] == original["response_contract"]
    assert "budgets" not in corrected["response_contract"]["locked_values"]
    assert "adjust_budget" in corrected["response_contract"]["allowed_actions"]
    assert "length_guidance" in corrected["response_correction"]
    assert (
        "Do not change budgets"
        not in corrected["response_correction"]["instruction"]
    )


def test_cross_stage_reply_gets_one_format_retry_without_changing_phase():
    from agent.hpo.cross_stage import validate_response

    prompt = render_prompt(
        "hpo_cross_stage_proposal", context={"phase": "paired_promotion"}
    )
    agent = AdvisoryAgentBase("fake", 0, 1, False)
    calls = []
    good = json.loads(prompt)["schema"]

    def invoke(text):
        payload = json.loads(text)
        calls.append(payload)
        return SimpleNamespace(
            content=json.dumps(
                {"action": "paired_promotion"} if len(calls) == 1 else good
            )
        )

    agent._llm = SimpleNamespace(invoke=invoke)
    assert (
        agent._request_json_advice(
            prompt, validator=lambda v: validate_response(v, "paired_promotion")
        )
        == good
    )
    assert len(calls) == 2
    assert calls[0]["context"] == calls[1]["context"]
    assert (
        calls[1]["response_correction"]["failure_stage"] == "schema_validation"
    )
    assert (
        "focus_region" not in calls[0]["response_contract"]["allowed_actions"]
    )
