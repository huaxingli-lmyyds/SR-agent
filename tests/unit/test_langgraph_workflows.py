import json
from types import SimpleNamespace

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


def test_hpo_best_metric_record_separates_training_and_evaluation_metrics() -> None:
    from types import SimpleNamespace

    from agent.agents.hpo_agent import HPOAgent
    from agent.hpo import Objective

    trial = SimpleNamespace(
        trial_id="trial_1",
        metrics={
            "eer": 0.03,
            "min_dcf": 0.1,
            "valid_error_rate": 0.2,
            "final_train_loss": 1.5,
        },
    )

    record = HPOAgent._best_metric_record(trial, Objective("eer", "min"))

    assert record["primary_metric"] == "eer"
    assert record["primary_value"] == 0.03
    assert record["eer"] == 0.03
    assert record["training"] == {"valid_error_rate": 0.2, "final_train_loss": 1.5}
    assert record["evaluation"] == {"eer": 0.03, "min_dcf": 0.1}


def test_hpo_successive_halving_defaults_use_larger_startup_cohort() -> None:
    from agent.agents.hpo_agent import HPOAgent

    budgets = [
        TrialBudget("screening", epochs=3, data_fraction=0.25),
        TrialBudget("promotion", epochs=8, data_fraction=0.5),
        TrialBudget("confirmation", epochs=20, data_fraction=1.0),
    ]

    assert HPOAgent._default_initial_trial_count("successive_halving", budgets, 30) == 9
    assert HPOAgent._default_promotion_limits(9, budgets) == [3, 1]
    assert HPOAgent._default_initial_trial_count("successive_halving", budgets, 10) == 6
    assert HPOAgent._default_promotion_limits(6, budgets) == [2, 1]


def test_hpo_study_learning_summary_records_next_study_guidance() -> None:
    from agent.agents.hpo_agent import HPOAgent
    from agent.hpo import Objective

    search_space = SearchSpace([
        SearchParameter("lr", "float", low=3e-4, high=3e-3, scale="log"),
        SearchParameter("batch_size", "categorical", choices=[16, 24, 32]),
    ])
    study = SimpleNamespace(
        strategy="successive_halving",
        candidate_strategy="tpe",
        search_space=search_space,
        strategy_reviews=[{
            "trigger": "stage_end:screening",
            "proposal": {
                "action": "refine_search_space",
                "requested_strategy": "tpe",
                "reason_codes": ["localize_around_best"],
            },
            "decision": {
                "accepted_fields": ["requested_strategy", "search_space"],
                "rejected_fields": [],
            },
            "applied_candidate_strategy": "tpe",
        }],
    )
    best_trial = SimpleNamespace(
        trial_id="trial_7",
        parameters={"lr": 0.0012, "batch_size": 24},
        metrics={"eer": 0.31, "valid_error_rate": 0.35},
    )

    summary = HPOAgent._study_learning_summary(study, best_trial, Objective("eer", "min"))

    assert summary["local_search_anchor"]["parameters"] == {"lr": 0.0012, "batch_size": 24}
    assert summary["local_search_anchor"]["value"] == 0.31
    assert summary["final_candidate_strategy"] == "tpe"
    assert summary["last_review"]["accepted_fields"] == ["requested_strategy", "search_space"]
    assert summary["next_study_recommendation"]["strategy"] == "tpe"
    assert summary["next_study_recommendation"]["search_space"] == search_space.to_dict()


def test_hpo_cross_study_memory_exposes_recent_learning_for_next_planning() -> None:
    from agent.agents.hpo_agent import HPOAgent

    learning_summary = {
        "local_search_anchor": {
            "trial_id": "trial_7",
            "parameters": {"lr": 0.0012, "batch_size": 24},
            "metric": "eer",
            "mode": "min",
            "value": 0.31,
        },
        "next_study_recommendation": {
            "strategy": "tpe",
            "search_space": {"parameters": [{"name": "lr", "low": 8e-4, "high": 2e-3}]},
            "anchor_parameters": {"lr": 0.0012, "batch_size": 24},
        },
    }
    campaign = {
        "best_value": 0.31,
        "best_experiment_id": "exp_2",
        "study_summaries": [
            {"experiment_id": "exp_1", "study_id": "study_1", "best_value": 0.4},
            {
                "experiment_id": "exp_2",
                "study_id": "study_2",
                "best_value": 0.31,
                "best_parameters": {"lr": 0.0012, "batch_size": 24},
                "learning_summary": learning_summary,
            },
        ],
    }

    memory = HPOAgent._cross_study_memory(campaign)

    assert memory["prior_study_count"] == 2
    assert memory["best_parameters"] == {"lr": 0.0012, "batch_size": 24}
    assert memory["local_search_anchor"] == learning_summary["local_search_anchor"]
    assert memory["next_study_recommendation"] == learning_summary["next_study_recommendation"]
    assert memory["recent_learnings"][-1]["learning_summary"] == learning_summary


def test_hpo_strategy_prompt_is_json_only_and_compact() -> None:
    from agent.agents.hpo_agent import HPOAgent

    prompt = HPOAgent._strategy_proposal_prompt({
        "phase": "runtime_review",
        "hard_max_training_runs": 30,
        "study": {"trial_count": 12},
    })
    payload = json.loads(prompt)

    assert payload["schema"]["action"] == "keep_strategy"
    assert "Return raw JSON only." in payload["rules"]
    assert "runtime_review" == payload["context"]["phase"]
    assert "trusted recipe anchor" in " ".join(payload["rules"])
    assert "cross_study_memory.local_search_anchor" in " ".join(payload["rules"])
    assert "```" not in prompt


def test_hpo_reference_profile_guides_local_ecapa_search() -> None:
    from agent.agents.hpo_agent import HPOAgent

    profile = HPOAgent._reference_search_profile("ecapa_tdnn")
    params = {
        item["name"]: item
        for item in profile["stable_search_space"]["parameters"]
    }

    assert profile["baseline_parameters"] == {
        "lr": 0.001,
        "batch_size": 32,
        "margin": 0.2,
        "weight_decay": 2e-6,
    }
    assert params["lr"]["low"] == 3e-4
    assert params["lr"]["high"] == 3e-3
    assert params["weight_decay"]["low"] == 5e-7
    assert params["weight_decay"]["high"] == 2e-5
    assert profile["local_adjustment_policy"]["max_changed_parameters_per_review"] == 2


def test_hpo_runtime_prompt_uses_compact_study_summary() -> None:
    from agent.agents.hpo_agent import HPOAgent

    study = SimpleNamespace(
        study_id="study_1",
        experiment_id="exp_1",
        status="running",
        strategy="adaptive_search",
        candidate_strategy="tpe",
        best_trial_id="trial_9",
        trial_ids=[f"trial_{idx}" for idx in range(20)],
        max_training_runs=30,
        search_space=SearchSpace([SearchParameter("lr", "float", low=1e-5, high=1e-2)]),
        budgets=[TrialBudget("screen", epochs=3, data_fraction=0.25)],
        strategy_reviews=[{"trigger": str(idx)} for idx in range(5)],
    )

    compact = HPOAgent._compact_study(study)

    assert compact["trial_count"] == 20
    assert "trial_ids" not in compact
    assert compact["recent_reviews"] == [{"trigger": "2"}, {"trigger": "3"}, {"trigger": "4"}]


def test_data_processing_advice_prompt_allows_validated_suggestions(monkeypatch) -> None:
    from agent.agents.data_processing_agent import DataProcessingAgent

    captured = {}

    class FakeLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return {"content": '{"diagnostics":[],"suggested_operations":[],"notes":[]}'}

    agent = DataProcessingAgent(enable_llm_advisor=True)
    agent._llm = FakeLLM()
    advice = agent._planning_advice({
        "dataset_uri": "/data",
        "dataset_type": "audio",
        "task_type": "speaker_verification",
        "target_goal": "validate",
        "profile": {"large": "omitted"},
    })
    payload = json.loads(captured["prompt"])

    assert advice["suggested_operations"] == []
    assert "suggested_operations are advisory only" in " ".join(payload["rules"])
    assert "profile" not in payload["context"]
