import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_openai")

from agent.agents.base_agent import BaseLangChainAgent


def test_message_tool_calls_are_normalized_into_execution_trace() -> None:
    result = {
        "messages": [
            {
                "tool_calls": [
                    {"id": "call_1", "name": "TrainModel", "args": {"lr": 0.001}}
                ]
            },
            {
                "tool_call_id": "call_1",
                "name": "TrainModel",
                "content": "EER: 0.03",
            },
        ]
    }

    trace = BaseLangChainAgent._extract_execution_trace(result)

    assert trace == [
        {
            "tool": "TrainModel",
            "input": {"lr": 0.001},
            "output": "EER: 0.03",
            "tool_call_id": "call_1",
        }
    ]
