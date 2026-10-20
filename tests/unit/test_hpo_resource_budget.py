import json

import pytest

from agent.hpo.resources import (
    CampaignResourceBudget, ResourceBudgetExhausted, positive_limit, stage_gpu_count,
)


@pytest.mark.parametrize("value", [0, -1, True, "1", float("nan"), float("inf")])
def test_resource_limit_rejects_invalid(value):
    with pytest.raises(ValueError, match="finite positive"):
        positive_limit(value, "max_gpu_hours")


def test_ddp_evaluation_cpu_counts():
    runtime = {"device": "cuda:0", "ddp_devices": [0, 1, 2]}
    assert stage_gpu_count(runtime, "speechbrain") == 3
    assert stage_gpu_count(runtime, "speechbrain", evaluation=True) == 1
    assert stage_gpu_count({"device": "cpu"}, "speechbrain") == 0
    assert stage_gpu_count({}, "fake") == 0


def test_gpu_accounting_includes_training_evaluation_and_retries(tmp_path):
    now = [1000.0]
    budget = CampaignResourceBudget(tmp_path / "resources.json", limits={"max_gpu_hours": 0.1}, clock=lambda: now[0])
    try:
        for stage, attempt, gpus, duration in [("training", 1, 3, 20), ("evaluation", 1, 1, 10), ("training", 2, 3, 5)]:
            deadline = budget.start_stage(experiment_id="e", trial_id="t", attempt=attempt, stage=stage, gpu_count=gpus)
            assert deadline == pytest.approx(now[0] + (360 - budget.data["gpu_seconds"]) / gpus)
            now[0] += duration
            budget.finish_stage()
        assert budget.snapshot()["gpu_seconds"] == 85
        assert len(budget.data["stages"]) == 3
        # CPU execution consumes time, not GPU hours.
        budget.start_stage(experiment_id="e", trial_id="cpu", attempt=1, stage="training", gpu_count=0)
        now[0] += 50
        budget.finish_stage()
        assert budget.snapshot()["gpu_seconds"] == 85
    finally:
        budget.close()


def test_tightest_stage_deadline_and_frozen_limits_survive_resume(tmp_path):
    now = [1000.0]
    path = tmp_path / "resources.json"
    limits = {"max_gpu_hours": 0.1, "max_wall_seconds": 100}
    budget = CampaignResourceBudget(path, limits=limits, clock=lambda: now[0])
    assert budget.start_stage(experiment_id="e", trial_id="t", attempt=1, stage="training", gpu_count=8, max_seconds=60) == 1045
    now[0] += 10
    budget.finish_stage()
    budget.close()
    now[0] += 50  # offline pauses do not extend the original wall deadline
    budget = CampaignResourceBudget(path, limits={}, resume=True, clock=lambda: now[0])
    assert budget.snapshot()["wall_seconds_remaining"] == 40
    assert budget.snapshot()["gpu_seconds"] == 80
    budget.close()
    with pytest.raises(ValueError, match="cannot change persisted"):
        CampaignResourceBudget(path, limits={"max_gpu_hours": 2}, resume=True, clock=lambda: now[0])
    # Failed initialization releases the OS lock too.
    budget = CampaignResourceBudget(path, limits={}, resume=True, clock=lambda: now[0])
    now[0] = 1101
    with pytest.raises(ResourceBudgetExhausted, match="max_wall_seconds_reached"):
        budget.check()
    budget.close()


def test_crash_reservation_is_not_refunded_or_charged_twice(tmp_path):
    now = [1000.0]
    path = tmp_path / "resources.json"
    budget = CampaignResourceBudget(path, limits={"max_gpu_hours": 0.01}, clock=lambda: now[0])
    budget.start_stage(experiment_id="e", trial_id="t", attempt=1, stage="training", gpu_count=3)
    budget.close()  # emulate a crash without settling the stage
    now[0] += 2
    for _ in range(2):
        budget = CampaignResourceBudget(path, limits={}, resume=True, clock=lambda: now[0])
        assert budget.snapshot()["gpu_seconds"] == 36
        assert budget.snapshot()["cost_may_be_incomplete"] is True
        assert len(budget.data["stages"]) == 1
        with pytest.raises(ResourceBudgetExhausted, match="max_gpu_hours_reached"):
            budget.check()
        budget.close()
    assert json.loads(path.read_text())["active_stage"] is None


def test_ledger_cannot_be_spent_by_two_controllers(tmp_path):
    path = tmp_path / "resources.json"
    budget = CampaignResourceBudget(path, limits={"max_wall_seconds": 100})
    try:
        with pytest.raises(RuntimeError, match="in use"):
            CampaignResourceBudget(path, limits={}, resume=True)
    finally:
        budget.close()


def test_advisor_timeout_is_bounded_and_exhaustion_does_not_fallback(tmp_path):
    from agent.agents.base_agent import AdvisoryAgentBase
    now = [1000.0]
    budget = CampaignResourceBudget(tmp_path / "r.json", limits={"max_wall_seconds": 5}, clock=lambda: now[0])
    calls = []

    class Model:
        def bind(self, **options):
            calls.append(options)
            return self

        def invoke(self, prompt):
            calls.append("invoke")
            now[0] += 6
            return "late result"

    agent = AdvisoryAgentBase("fake", 0, 1, False)
    agent._llm = Model()
    agent._resource_budget = budget
    try:
        with pytest.raises(ResourceBudgetExhausted):
            agent._invoke_with_readonly_tools("prompt", [])
        assert calls == [{"timeout": 5.0}, "invoke"]
    finally:
        budget.close()
