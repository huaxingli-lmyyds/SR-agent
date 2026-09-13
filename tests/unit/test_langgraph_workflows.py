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


def test_data_processing_llm_advice_runs_after_profile(
    tmp_path, dataset_dir
) -> None:
    def advisor(state):
        assert state["profile"]["quality_metrics"]
        return {
            "diagnostics": [],
            "suggested_operations": [
                {"operation": "validate_dataset", "reason": "LLM check"}
            ],
            "notes": [],
        }

    result = DataProcessingWorkflow(strategy_advisor=advisor).run(
        {
            "dataset_uri": str(dataset_dir),
            "dataset_type": "text",
            "task_type": "test",
            "target_goal": "validate",
            "output_path": str(tmp_path / "version.json"),
        }
    )

    assert result["status"] == "success"
    assert result["advice"]["suggested_operations"][0]["operation"] == (
        "validate_dataset"
    )


def test_data_processing_rejects_unknown_llm_operation(
    tmp_path, dataset_dir
) -> None:
    result = DataProcessingWorkflow(
        strategy_advisor=lambda state: {
            "suggested_operations": [
                {"operation": "unregistered_operation"}
            ]
        }
    ).run(
        {
            "dataset_uri": str(dataset_dir),
            "dataset_type": "text",
            "task_type": "test",
            "target_goal": "validate",
            "output_path": str(tmp_path / "version.json"),
        }
    )

    assert result["status"] == "success"
    rejected = result["plan"]["rejected_operations"]
    assert rejected[0]["operation"] == "unregistered_operation"
    assert rejected[0]["source"] == "llm"


def test_data_processing_rejects_unknown_explicit_operation(
    tmp_path, dataset_dir
) -> None:
    with pytest.raises(KeyError, match="unregistered_operation"):
        DataProcessingWorkflow().run(
            {
                "dataset_uri": str(dataset_dir),
                "dataset_type": "text",
                "task_type": "test",
                "target_goal": "validate",
                "output_path": str(tmp_path / "version.json"),
                "requested_operations": [
                    {"operation": "unregistered_operation"}
                ],
            }
        )


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


def test_orchestration_resume_dispatches_only_hpo_agent() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration("hpo_agent", ("run",), FakeAgent))
    registry.register(AgentRegistration("data_processing_agent", ("run",), FakeAgent))
    workflow = OrchestrationWorkflow(
        registry,
        TaskDispatcher(registry, MessageService("session")),
        CompletionPolicy(["hpo_agent"]),
        lambda context, budget: AgentTaskRequest(
            action="",
            objective="resume",
            context=context,
            budget=budget,
        ),
    )

    state = workflow.run({"resume_experiment_id": "hpo_existing"}, {})

    assert [record.agent_type for record in state["records"]] == ["hpo_agent"]
    assert state["completion"]["complete"]


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


def test_hpo_agent_rejects_missing_validation_protocol_before_creating_study(
    tmp_path, minimal_config, dataset_dir
) -> None:
    agent = HPOAgent(
        config_path=str(minimal_config),
        experiments_dir=str(tmp_path / "experiments"),
        verbose=False,
    )
    result = agent.execute_task(AgentTaskRequest(
        action="optimize_hyperparameters",
        objective="test validation gate",
        context={
            "data_folder": str(dataset_dir),
            "runtime_options": {},
        },
        budget={"max_training_runs": 1},
    ))

    assert result.status == "failed"
    assert "verification_config" in result.error
    assert not list((tmp_path / "experiments").glob("*/experiment_record.json"))


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


def test_hpo_successive_halving_defaults_use_conservative_complete_bracket() -> None:
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
        sampler_config={"n_startup_trials": 4},
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
    assert summary["next_study_recommendation"]["strategy"] == "successive_halving"
    assert summary["next_study_recommendation"]["sampler"] == "tpe"
    assert summary["next_study_recommendation"]["sampler_config"] == {
        "n_startup_trials": 4
    }
    assert summary["next_study_recommendation"]["pruner"] == "successive_halving"
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


def test_hpo_next_study_proposal_is_inherited_only_by_policy_controllers() -> None:
    stored = {
        "action": "switch_strategy",
        "requested_sampler": "tpe",
        "sampler_config": {"n_startup_trials": 3},
        "reason_codes": ["plateau"],
        "proposal_id": "proposal_deferred",
        "created_at": "2026-09-01T00:00:00",
    }
    campaign = {"study_summaries": [{"next_study_proposal": stored}]}

    inherited = HPOAgent._next_study_proposal(campaign, "llm")

    assert inherited is not None
    assert inherited.requested_sampler == "tpe"
    assert inherited.sampler_config == {"n_startup_trials": 3}
    assert inherited.proposal_id == "proposal_deferred"
    assert HPOAgent._next_study_proposal(campaign, "rule") is not None
    assert HPOAgent._next_study_proposal(campaign, "fixed") is None


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



def test_data_processing_prompt_uses_compact_profile(monkeypatch) -> None:
    from agent.agents.data_processing_agent import DataProcessingAgent

    captured = {}

    class FakeLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return {
                "content": (
                    '{"diagnostics":[],"suggested_operations":[],'
                    '"notes":[]}'
                )
            }

    agent = DataProcessingAgent(enable_llm_advisor=True)
    agent._llm = FakeLLM()
    advice = agent._planning_advice(
        {
            "dataset_uri": "/data",
            "dataset_type": "audio",
            "task_type": "speaker_verification",
            "target_goal": "validate",
            "profile": {
                "quality_metrics": {"warning_count": 1},
                "distributions": {"sample_rates": {"16000": 2}},
                "issues": [
                    {
                        "code": "duration_out_of_range",
                        "severity": "warning",
                        "message": "duration",
                        "suggested_operation": "filter_by_duration",
                        "evidence": {"large": "omitted"},
                    }
                ],
            },
        }
    )
    payload = json.loads(captured["prompt"])

    assert advice["suggested_operations"] == []
    assert payload["context"]["quality_metrics"] == {"warning_count": 1}
    assert payload["context"]["issues"][0]["code"] == (
        "duration_out_of_range"
    )
    assert "evidence" not in payload["context"]["issues"][0]
    assert "available_operations" in payload["context"]
    assert "profile" not in payload["context"]


def test_hpo_campaign_history_keeps_valid_base_fidelity_and_deduplicates() -> None:
    from agent.hpo import Objective, Trial

    budget = TrialBudget("screening", epochs=3, data_fraction=0.25)
    existing = [{
        "trial_id": "warm_old",
        "parameters": {"lr": 0.001},
        "budget": budget.to_dict(),
        "status": "completed",
        "rung": 0,
        "metrics": {"eer": 0.4},
    }]
    trials = [
        Trial(
            "trial_new",
            {"lr": 0.001},
            budget,
            status="completed",
            rung=0,
            metrics={"eer": 0.3},
        ),
        Trial(
            "trial_promoted_child",
            {"lr": 0.002},
            TrialBudget("full", epochs=20, data_fraction=1.0),
            status="completed",
            rung=1,
            metrics={"eer": 0.2},
        ),
        Trial(
            "trial_failed",
            {"lr": 0.003},
            budget,
            status="failed",
            rung=0,
            metrics={"eer": 0.1},
        ),
    ]

    history = HPOAgent._merge_campaign_history(
        existing, trials, Objective("eer", "min"), "successive_halving"
    )

    assert len(history) == 1
    assert history[0]["trial_id"] == "warm_trial_new"
    assert history[0]["metrics"]["eer"] == 0.3


def test_hpo_resource_snapshot_and_budget_analysis_are_structured() -> None:
    budgets = [
        TrialBudget("screening", epochs=3, data_fraction=0.25),
        TrialBudget("promotion", epochs=8, data_fraction=0.5),
        TrialBudget("confirmation", epochs=20, data_fraction=1.0),
    ]
    space = SearchSpace([
        SearchParameter("lr", "float", low=3e-4, high=3e-3, scale="log"),
        SearchParameter("batch_size", "categorical", choices=[16, 24, 32]),
    ])

    resources = HPOAgent._resource_snapshot(
        {"device": "cuda:0", "precision": "fp16"},
        {"max_wall_time_seconds": 7200},
    )
    analysis = HPOAgent._search_budget_analysis(
        space,
        budgets,
        30,
        9,
        [3, 1],
        3,
        {
            "max_total_training_runs": 60,
            "study_summaries": [{"training_runs": 13}],
        },
    )

    assert resources["requested_runtime"] == {
        "device": "cuda:0",
        "precision": "fp16",
    }
    assert resources["declared_limits"] == {"max_wall_time_seconds": 7200}
    assert "cuda" in resources
    assert analysis["planned_training_runs"] == 13
    assert analysis["unused_training_run_capacity"] == 17
    assert analysis["estimated_relative_work_units"] == 38.75
    assert analysis["campaign_remaining_training_runs"] == 47
    assert analysis["resource_sensitive_parameters"] == ["batch_size"]
