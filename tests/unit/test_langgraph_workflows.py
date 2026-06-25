import pytest

pytest.importorskip("langgraph")

from agent.agents.communication import AgentTaskRequest, MessageService
from agent.agents.coordination import AgentRegistration, AgentRegistry, CompletionPolicy, TaskDispatcher
from agent.agents.orchestration_workflow import OrchestrationWorkflow
from agent.data_processing.workflow import DataProcessingWorkflow
from agent.agents.hpo_agent import HPOAgent
from agent.hpo import SearchParameter, SearchSpace, TrialBudget
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


def test_data_processing_advisor_unknown_operation_is_rejected_without_stopping_workflow(tmp_path, dataset_dir) -> None:
    workflow = DataProcessingWorkflow(
        strategy_advisor=lambda state: {
            "suggested_operations": [{"operation": "unregistered_operation"}]
        }
    )

    result = workflow.run({
        "dataset_uri": str(dataset_dir),
        "dataset_type": "text",
        "task_type": "test",
        "target_goal": "optimize",
        "output_path": str(tmp_path / "version.json"),
    })

    assert result["status"] == "success"
    assert all(
        operation["operation"] != "unregistered_operation"
        for operation in result["plan"]["operations"]
    )
    assert result["plan"]["rejected_operations"][0]["operation"] == "unregistered_operation"


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


def test_orchestration_passes_data_processing_result_to_hpo_context() -> None:
    captured = {}

    class DataAgent:
        def execute_task(self, request):
            from agent.agents.communication import AgentTaskResult

            return AgentTaskResult(
                status="success",
                summary={"data_handoff": {"consumer_uri": "derived", "consumption_status": "ready"}},
                request_id=request.request_id,
            )

    class HPOAgent:
        def execute_task(self, request):
            from agent.agents.communication import AgentTaskResult

            captured.update(request.context)
            return AgentTaskResult(status="success", request_id=request.request_id)

    registry = AgentRegistry()
    registry.register(AgentRegistration("data_processing_agent", ("prepare",), DataAgent))
    registry.register(AgentRegistration("hpo_agent", ("optimize",), HPOAgent))
    workflow = OrchestrationWorkflow(
        registry,
        TaskDispatcher(registry, MessageService("session")),
        CompletionPolicy(["data_processing_agent", "hpo_agent"]),
        lambda context, budget: AgentTaskRequest(action="", objective="test", context=context, budget=budget),
    )

    state = workflow.run({}, {})

    previous = captured["previous_results"]["data_processing_agent"]
    assert previous["summary"]["data_handoff"]["consumer_uri"] == "derived"
    assert state["completion"]["complete"]


def test_default_agent_search_space_does_not_include_budget_controlled_epochs() -> None:
    from agent.models import SpeechBrainEcapaAdapter

    names = {
        item.name
        for item in HPOAgent._build_search_space(
            SpeechBrainEcapaAdapter().default_search_space()
        ).parameters
    }

    assert "number_of_epochs" not in names


def test_agent_rejects_epoch_search_when_budget_controls_epochs() -> None:
    space = SearchSpace([SearchParameter("number_of_epochs", "int", low=1, high=5)])

    with pytest.raises(ValueError, match="conflicts with budget.epochs"):
        HPOAgent._validate_search_budget_compatibility(
            space,
            [TrialBudget("full", epochs=5)],
        )
