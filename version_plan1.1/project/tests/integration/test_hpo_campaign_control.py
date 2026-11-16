import json

import pytest

from agent.agents.communication import AgentTaskRequest
from agent.agents.hpo_agent import HPOAgent
from agent.hpo import StrategyProposal
from agent.memory import MemoryService
from agent.runners import RUNNER_ADAPTERS
from tests.fakes import FakeRunnerAdapter


@pytest.fixture
def campaign_run(tmp_path, minimal_config, dataset_dir, monkeypatch):
    runner = FakeRunnerAdapter()
    monkeypatch.setitem(RUNNER_ADAPTERS, "fake", runner)
    pairs = tmp_path / "validation.txt"
    pairs.write_text(
        "1 val1/a.wav val1/b.wav\n0 val1/a.wav val2/a.wav\n", encoding="utf-8"
    )
    agent = HPOAgent(
        config_path=str(minimal_config),
        runner="fake",
        verbose=False,
        experiments_dir=str(tmp_path / "experiments"),
        memory_service=MemoryService(tmp_path / "memory"),
        enable_llm_advisor=True,
    )
    monkeypatch.setattr(
        agent,
        "_planning_proposal",
        lambda *a, **k: StrategyProposal(action="keep_strategy"),
    )
    monkeypatch.setattr(agent, "_runtime_strategy_reviewer", lambda *a: None)
    calls = []
    original = runner.run_training

    def train(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "run_training", train)
    request = AgentTaskRequest(
        action="optimize_hyperparameters",
        objective="campaign control",
        context={
            "data_folder": str(dataset_dir),
            "sampler": "random_search",
            "pruner": "none",
            "controller_mode": "llm",
            "search_space": {
                "parameters": [
                    {
                        "name": "lr",
                        "parameter_type": "float",
                        "low": 0.0001,
                        "high": 0.003,
                    }
                ]
            },
            "budgets": [{"stage": "full", "epochs": 1, "data_fraction": 1.0}],
            "runtime_options": {
                "verification_config": str(minimal_config),
                "validation_pairs": str(pairs),
                "training_exclusion_pairs": str(pairs),
            },
        },
        budget={
            "max_training_runs": 4,
            "max_total_training_runs": 12,
            "max_studies": 3,
            "campaign_patience": 3,
        },
    )
    return agent, request, calls


def test_llm_compact_and_stop_affect_execution_and_survive_resume(
    campaign_run, monkeypatch
):
    agent, request, training = campaign_run
    reviews = []

    def choose(prompt):
        context = json.loads(prompt)["context"]
        reviews.append(context)
        return {
            "plan_id": "compact" if len(reviews) == 1 else "stop",
            "evidence_study_ids": [s["study_id"] for s in context["studies"]],
            "reason": "no confirmed improvement",
        }

    monkeypatch.setattr(agent, "_request_campaign_decision", choose)
    result = agent.execute_task(request)
    assert result.status == "success", result.error
    campaign = result.summary["campaign"]
    assert [s["training_runs"] for s in campaign["study_summaries"]] == [4, 2]
    assert len(training) == 6 and len(reviews) == 2
    assert campaign["stop_reason"] == "controller_stop"
    decision = campaign["study_summaries"][0]["continuation_decision"]
    assert (
        decision["applied"]
        and decision["actual_plan"]["max_training_runs"] == 2
    )
    resumed = agent.execute_task(
        AgentTaskRequest(
            action="optimize_hyperparameters",
            objective="resume stopped campaign",
            context={
                "resume_experiment_id": campaign["study_summaries"][-1][
                    "experiment_id"
                ]
            },
        )
    )
    assert resumed.status == "success", resumed.error
    assert not resumed.summary["campaign_continuation_pending"]
    assert len(training) == 6 and len(reviews) == 2


@pytest.mark.parametrize("mode", ["fixed", "rule"])
def test_non_llm_modes_never_call_campaign_model(
    campaign_run, monkeypatch, mode
):
    agent, request, training = campaign_run
    request.context["controller_mode"] = mode
    monkeypatch.setattr(
        agent, "_request_campaign_decision", lambda _: pytest.fail("LLM called")
    )
    result = agent.execute_task(request)
    assert result.status == "success", result.error
    assert len(result.summary["campaign"]["study_summaries"]) == 3
    if mode == "fixed":
        assert len(training) == 12
        assert all(
            "continuation_decision" not in s
            for s in result.summary["campaign"]["study_summaries"]
        )


def test_invalid_campaign_advice_falls_back_to_bounded_rule(
    campaign_run, monkeypatch
):
    agent, request, training = campaign_run
    monkeypatch.setattr(
        agent,
        "_request_campaign_decision",
        lambda _: {"plan_id": "stop", "max_training_runs": 999},
    )
    result = agent.execute_task(request)
    assert result.status == "success", result.error
    assert len(training) <= 12
    decision = result.summary["campaign"]["study_summaries"][0][
        "continuation_decision"
    ]
    assert decision["source"] == "rule" and decision["error"]


def test_interrupted_campaign_review_retries_without_retraining(
    campaign_run, monkeypatch
):
    agent, request, training = campaign_run

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(agent, "_request_campaign_decision", interrupt)
    with pytest.raises(KeyboardInterrupt):
        agent.execute_task(request)
    eid = agent._resource_experiment_ids[-1]
    monkeypatch.setattr(
        agent,
        "_request_campaign_decision",
        lambda prompt: {"plan_id": "continue", "reason": "resume consultation",
                        "evidence_study_ids": [json.loads(prompt)["context"]["studies"][-1]["study_id"]]},
    )
    result = agent.execute_task(
        AgentTaskRequest(
            action="optimize_hyperparameters",
            objective="resume",
            context={"resume_experiment_id": eid},
        )
    )
    assert result.status == "success", result.error
    assert len(training) == 4
    decision = result.summary["campaign"]["study_summaries"][-1][
        "continuation_decision"
    ]
    assert decision["status"] == "decided" and decision["source"] == "llm"
    assert decision["consultation"]["attempts"] == 2


@pytest.mark.parametrize("total,expected", [(26, [13, 11]), (23, [13])])
def test_cross_stage_campaign_reserves_complete_promotion_brackets(
    campaign_run, monkeypatch, total, expected
):
    agent, request, training = campaign_run
    request.context.update(
        sampler="tpe",
        pruner="successive_halving",
        cross_stage_probes=True,
        budgets=[
            {"stage": str(n), "epochs": n, "data_fraction": 1.0}
            for n in (1, 2, 3)
        ],
    )
    request.budget.update(
        max_training_runs=13,
        max_total_training_runs=total,
        max_studies=2,
        initial_trial_count=9,
        promotion_limits=[3, 1],
        candidate_batch_size=3,
    )
    monkeypatch.setattr(
        agent,
        "_request_cross_stage_proposal",
        lambda _: pytest.fail("tied scores have no ranking contrast"),
    )

    def compact(prompt):
        context = json.loads(prompt)["context"]
        return {
            "plan_id": "compact",
            "reason": "retain full confirmation at lower cost",
            "evidence_study_ids": [context["studies"][-1]["study_id"]],
        }

    monkeypatch.setattr(agent, "_request_campaign_decision", compact)
    result = agent.execute_task(request)
    assert result.status == "success", result.error
    assert [
        s["training_runs"]
        for s in result.summary["campaign"]["study_summaries"]
    ] == expected
    assert len(training) == sum(expected)
    if total == 23:
        assert (
            result.summary["campaign"]["stop_reason"]
            == "no_feasible_next_study"
        )
