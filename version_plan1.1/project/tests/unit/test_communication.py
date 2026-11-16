from agent.agents.communication import AgentTaskRequest, MessageService
from agent.agents.coordination import (
    AgentRegistration,
    AgentRegistry,
    CompletionPolicy,
    TaskDispatcher,
)
from tests.fakes import FailingAgent, FakeAgent


def test_dispatch_preserves_request_correlation() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration("fake_agent", ("optimize",), FakeAgent))
    messages = MessageService("session")
    dispatcher = TaskDispatcher(registry, messages)
    request = AgentTaskRequest(action="optimize", objective="test")

    record = dispatcher.dispatch("fake_agent", request)
    conversation = messages.conversation(request.request_id)

    assert record.status == "success"
    assert len(conversation) == 2
    assert conversation[1].reply_to == conversation[0].message_id
    assert conversation[1].payload["request_id"] == request.request_id
    assert CompletionPolicy(["fake_agent"]).evaluate([record]).complete


def test_dispatch_converts_agent_exception_to_failed_result() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration("failing_agent", ("run",), FailingAgent))
    dispatcher = TaskDispatcher(registry, MessageService("session"))

    record = dispatcher.dispatch(
        "failing_agent",
        AgentTaskRequest(action="run", objective="fail"),
    )
    decision = CompletionPolicy(["failing_agent"]).evaluate([record])

    assert record.status == "failed"
    assert "fake failure" in record.result["error"]
    assert decision.status == "failed"
