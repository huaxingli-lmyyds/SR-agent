import pytest

pytest.importorskip("langgraph")

from agent.agents.communication import AgentTaskRequest, MessageService
from agent.agents.coordination import AgentRegistration, AgentRegistry, CompletionPolicy, TaskDispatcher
from agent.agents.orchestration_workflow import OrchestrationWorkflow
from agent.data_processing.workflow import DataProcessingWorkflow
from tests.fakes import FakeAgent


def test_data_processing_langgraph_publishes_valid_dataset(tmp_path, dataset_dir) -> None:
    output = tmp_path / "version.json"
    result = DataProcessingWorkflow().run({
        "dataset_uri": str(dataset_dir),
        "dataset_type": "text",
        "task_type": "test",
        "target_goal": "validate",
        "output_path": str(output),
    })

    assert result["status"] == "success"
    assert result["published_version"]["dataset_id"]
    assert output.exists()


def test_data_processing_advisor_failure_cannot_stop_workflow(tmp_path, dataset_dir) -> None:
    def failing_advisor(state):
        raise RuntimeError("advisor unavailable")

    result = DataProcessingWorkflow(strategy_advisor=failing_advisor).run({
        "dataset_uri": str(dataset_dir),
        "dataset_type": "text",
        "task_type": "test",
        "target_goal": "validate",
        "output_path": str(tmp_path / "version.json"),
    })

    assert result["status"] == "success"
    assert "advisor unavailable" in result["advice"]["advice_error"]


def test_orchestration_langgraph_automatically_includes_registered_agents() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration("extra_agent", ("run",), FakeAgent))
    dispatcher = TaskDispatcher(registry, MessageService("session"))
    workflow = OrchestrationWorkflow(
        registry,
        dispatcher,
        CompletionPolicy(["extra_agent"]),
        lambda context, budget: AgentTaskRequest(action="", objective="test", context=context, budget=budget),
    )

    state = workflow.run({}, {})

    assert state["completion"]["complete"]
    assert state["records"][0].agent_type == "extra_agent"


def test_orchestration_advisor_cannot_change_decision_policy_order() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration("hpo_agent", ("run",), FakeAgent))
    registry.register(AgentRegistration("data_processing_agent", ("run",), FakeAgent))
    dispatcher = TaskDispatcher(registry, MessageService("session"))
    workflow = OrchestrationWorkflow(
        registry,
        dispatcher,
        CompletionPolicy(["data_processing_agent", "hpo_agent"]),
        lambda context, budget: AgentTaskRequest(action="", objective="test", context=context, budget=budget),
        advisor=lambda agents, context: {"suggested_order": ["hpo_agent", "data_processing_agent"]},
    )

    state = workflow.run({}, {})

    assert [record.agent_type for record in state["records"]] == [
        "data_processing_agent",
        "hpo_agent",
    ]
    assert state["advice"]["suggested_order"][0] == "hpo_agent"
