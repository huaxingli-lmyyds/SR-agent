"""Interface failures must stay structured and must not become valid results."""

from types import SimpleNamespace

import pytest

from agent.agents.communication import (
    AgentTaskRequest,
    AgentTaskResult,
    MessageService,
)
from agent.agents.coordination import (
    AgentRegistration,
    AgentRegistry,
    TaskDispatcher,
)
from agent.runners.contracts import (
    collect_training_result,
    validate_runner_compatibility,
    validate_runner_result,
)
from agent.tasks import (
    get_task_adapter,
    resolve_task_metrics,
    task_metric_metadata,
)
from agent.tools.runtime import apply_runtime_deadline, build_run_options


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {},
        {"status": "completed"},
        {"status": []},
        {"status": "success", "metrics": []},
    ],
)
def test_runner_rejects_malformed_responses(raw):
    with pytest.raises(ValueError, match="runner"):
        validate_runner_result(raw, "training")


def test_optional_result_collector_does_not_mutate_backend_metrics(tmp_path):
    raw = {"status": "success", "metrics": {}, "valid_error_rate": 0.1}
    result = collect_training_result(SimpleNamespace(), raw, tmp_path, tmp_path)
    assert result["metrics"]["valid_error_rate"] == 0.1
    assert raw["metrics"] == {}


def test_runner_compatibility_accepts_single_string_capabilities():
    runner = SimpleNamespace(runner="demo", supported_implementations="demo")
    validate_runner_compatibility(runner, implementation="demo")
    with pytest.raises(ValueError, match="does not support"):
        validate_runner_compatibility(runner, implementation="d")


@pytest.mark.parametrize(
    "returned",
    [
        {"status": "success"},
        AgentTaskResult("unexpected"),
        AgentTaskResult("success", request_id="another_request"),
    ],
)
def test_bad_agent_results_become_correlated_failures(returned):
    registry = AgentRegistry()
    registry.register(
        AgentRegistration(
            "bad",
            ["run"],
            lambda: SimpleNamespace(execute_task=lambda _: returned),
        )
    )
    messages = MessageService("session")
    request = AgentTaskRequest("run", "test")
    result = TaskDispatcher(registry, messages).dispatch("bad", request)
    assert result.status == "failed"
    assert result.result["request_id"] == request.request_id
    assert messages.history()[-1].message_type == "error"


def test_generic_tasks_keep_their_own_metrics_and_units():
    task = SimpleNamespace(task_type="classification")
    assert resolve_task_metrics(
        task, {"metrics": {"accuracy": 0.9}, "eer": 0.01}
    ) == {"accuracy": 0.9}
    assert task_metric_metadata(task, {}) == {
        "metric_protocol": None,
        "metric_units": {},
    }
    assert task_metric_metadata(
        task,
        {"metric_protocol": "frozen", "metric_units": {"accuracy": "percent"}},
    ) == {
        "metric_protocol": "frozen",
        "metric_units": {"accuracy": "percent"},
    }
    speaker = get_task_adapter("speaker_verification")
    assert resolve_task_metrics(speaker, {"status": "success", "eer": 0.1}) == {
        "eer": 0.1
    }
    assert (
        task_metric_metadata(speaker, {})["metric_units"]["eer"] == "ratio_0_1"
    )


def test_runtime_options_and_tighter_deadline_are_shared(monkeypatch):
    monkeypatch.setattr("agent.tools.runtime.time", lambda: 100)
    options = build_run_options(
        device="cuda", precision="fp32", ddp_devices=[0, 1]
    )
    assert options["batch_size_semantics"] == "global"
    overrides = {"_hpo_max_duration_seconds": 3}
    apply_runtime_deadline(overrides, 110, "training")
    assert overrides["_hpo_max_duration_seconds"] == 3
    apply_runtime_deadline(overrides, 102, "evaluation")
    assert overrides["_hpo_max_duration_seconds"] == 2
    with pytest.raises(RuntimeError, match="expired"):
        apply_runtime_deadline(overrides, 100, "evaluation")
