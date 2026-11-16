import json
from types import SimpleNamespace

import pytest

from agent.agents.communication import AgentTaskRequest, AgentTaskResult, MessageService
from agent.agents.coordination import AgentRegistration, AgentRegistry, CompletionPolicy, TaskDispatcher
from agent.agents.orchestration_workflow import OrchestrationWorkflow
from agent.data_processing.contracts import DataOperation, DataProcessingPlan, OperationImpact, QualityPolicy
from agent.data_processing.feedback import normalize_feedback
from agent.data_processing.impact import impact_rejection
from agent.data_processing.registry import PROCESSORS
from agent.data_processing.service import execute_plan, infer_dataset_spec, profile_dataset, publish_dataset_version
from agent.data_processing.workflow import DataProcessingWorkflow


def workflow(registry, advisor=None):
    return OrchestrationWorkflow(
        registry, TaskDispatcher(registry, MessageService("test")),
        CompletionPolicy([item["agent_type"] for item in registry.describe()]),
        lambda context, budget: AgentTaskRequest("run", "test", context, budget),
        advisor=advisor,
    )


def register(registry, name, calls, *, requires=(), status="success"):
    def execute(request):
        calls.append(name)
        return AgentTaskResult(status, request_id=request.request_id)
    registry.register(AgentRegistration(name, ("run",), lambda: SimpleNamespace(execute_task=execute), requires=requires))


def test_dependency_order_and_scalable_graph():
    registry, calls = AgentRegistry(), []
    register(registry, "a_downstream", calls, requires=("z_upstream",))
    register(registry, "z_upstream", calls)
    for index in range(15):
        register(registry, f"extra{index}", calls)
    result = workflow(registry).run({}, {})
    assert result["completion"]["complete"]
    assert calls.index("z_upstream") < calls.index("a_downstream")
    assert len(calls) == 17


@pytest.mark.parametrize("missing", [True, False])
def test_invalid_dependency_graph_fails_before_advice(missing):
    registry, calls = AgentRegistry(), []
    register(registry, "a", calls, requires=("missing" if missing else "b",))
    if not missing:
        register(registry, "b", calls, requires=("a",))
    with pytest.raises(ValueError, match="missing dependencies|cyclic"):
        workflow(registry, lambda *args: pytest.fail("consulted invalid graph")).run({}, {})
    assert calls == []


def test_failed_dependency_blocks_transitively_but_not_independent_agent():
    registry, calls = AgentRegistry(), []
    register(registry, "a", calls, status="failed")
    register(registry, "b", calls, requires=("a",))
    register(registry, "c", calls, requires=("b",))
    register(registry, "d", calls)
    result = workflow(registry).run({}, {})
    assert calls == ["a", "d"]
    assert result["latest_results"]["c"]["summary"]["blocked_by"] == ["b"]
    assert result["completion"]["status"] == "failed"


def test_hpo_resume_uses_saved_protocol_without_reprocessing_data():
    registry, calls = AgentRegistry(), []
    register(registry, "data_processing_agent", calls)
    register(registry, "hpo_agent", calls, requires=("data_processing_agent",))
    graph = workflow(registry)
    graph.completion_policy = CompletionPolicy(["hpo_agent"])
    assert graph.run({"resume_experiment_id": "saved"}, {})["completion"]["complete"]
    assert calls == ["hpo_agent"]


def feedback(dataset, **updates):
    return {"dataset_uri": str(dataset), "evaluation_role": "hpo_validation", "study_id": "study1",
            "trial_count": 3, "diagnostics": ["short utterance degradation"], **updates}


@pytest.mark.parametrize("updates,reason", [
    ({"evaluation_role": "final_test"}, "not_hpo_validation"),
    ({"dataset_uri": "some_other_dataset"}, "dataset_mismatch"),
    ({"trial_count": 0}, "missing_trial_evidence"),
    ({"trial_count": True}, "missing_trial_evidence"),
    ({"study_id": ""}, "missing_or_invalid_study_id"),
    ({"diagnostics": "unstructured"}, "invalid_diagnostics"),
])
def test_feedback_rejects_unscoped_evidence(tmp_path, updates, reason):
    assert normalize_feedback(feedback(tmp_path, **updates), str(tmp_path))["reason"] == reason


def test_feedback_bounds_fields_and_preserves_provenance(tmp_path):
    result = normalize_feedback(feedback(tmp_path, diagnostics=["x" * 500] * 20,
        metrics={"eer": .12, "min_dcf": float("nan")}, commands=["delete"]), str(tmp_path))
    evidence = result["evidence"]
    assert evidence["study_id"] == "study1"
    assert len(evidence["diagnostics"]) == 5 and len(evidence["diagnostics"][0]) == 240
    assert evidence["diagnostics_omitted"] == 15
    assert evidence["metrics"] == {"eer": .12}
    assert "commands" not in evidence


def test_workflow_feedback_reaches_advisor_and_gates_opportunities(tmp_path, dataset_dir):
    calls = []
    graph = DataProcessingWorkflow(strategy_advisor=lambda state: calls.append(state) or {})
    state = {"dataset_uri": str(dataset_dir), "dataset_type": "text", "output_path": str(tmp_path / "version.json"),
             "consultation_mode": "evidence"}
    skipped = graph.run(state)
    assert not calls and skipped["decision"]["status"] == "skipped"
    completed = graph.run({**state, "hpo_feedback": feedback(dataset_dir)})
    assert calls[0]["hpo_feedback"]["evidence"]["study_id"] == "study1"
    assert completed["decision"]["status"] == "completed" and completed["status"] == "success"
    calls.clear()
    rejected = graph.run({**state, "hpo_feedback": feedback(dataset_dir, evaluation_role="final_test")})
    assert rejected["hpo_feedback"]["status"] == "rejected" and not calls


def test_advisor_failure_remains_visible_and_does_not_replace_quality_gate(tmp_path, dataset_dir):
    def fail(state):
        raise RuntimeError("offline")
    result = DataProcessingWorkflow(strategy_advisor=fail).run({"dataset_uri": str(dataset_dir),
        "dataset_type": "text", "output_path": str(tmp_path / "v.json")})
    assert result["decision"]["status"] == "failed"
    assert "offline" in result["advice"]["advice_error"]
    assert result["status"] == "success" and result["results"][-1]["operation"] == "quality_gate"


@pytest.mark.parametrize("kwargs", [
    {"max_operation_affected_fraction": float("nan")},
    {"max_operation_affected_fraction": 1.01},
    {"max_operation_output_bytes": -1}, {"max_operation_output_bytes": True},
])
def test_invalid_impact_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        QualityPolicy(**kwargs)


def test_impact_limits_boundary_unknown_and_legacy():
    impact = OperationImpact("filter", 10, 2, 100, True, {"scan_complete": True})
    assert impact_rejection(impact, QualityPolicy(max_operation_affected_fraction=.2, max_operation_output_bytes=100)) is None
    assert impact_rejection(impact, QualityPolicy(max_operation_affected_fraction=.1)) == "affected_fraction_exceeded"
    assert impact_rejection(impact, QualityPolicy(max_operation_output_bytes=99)) == "estimated_output_bytes_exceeded"
    impact.details = {}
    assert impact_rejection(impact, QualityPolicy(max_operation_affected_fraction=.2)) == "impact_unknown_or_incomplete"
    assert impact_rejection(impact, QualityPolicy()) is None


def test_impact_rejection_prevents_execution_and_publication(monkeypatch, tmp_path, dataset_dir):
    processor = SimpleNamespace(
        supported_data_types={"text"}, validate=lambda *args: None,
        preview=lambda *args: OperationImpact("controlled", 10, 5, 100, True, {"scan_complete": True}),
        execute=lambda *args: pytest.fail("rejected mutation executed"),
    )
    monkeypatch.setitem(PROCESSORS._processors, "controlled", processor)
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="text")
    plan = DataProcessingPlan(dataset, [DataOperation("controlled")], quality_policy={"max_operation_affected_fraction": .2})
    results = execute_plan(plan, output_root=tmp_path / "output")
    assert results[0].details["rejection_reason"] == "affected_fraction_exceeded"
    with pytest.raises(ValueError, match="failed operations"):
        publish_dataset_version(dataset, results, tmp_path / "v.json")
    assert not (tmp_path / "v.json").exists()


def test_atomic_publication_failure_preserves_previous_version(monkeypatch, tmp_path, dataset_dir):
    from agent.data_processing import service
    dataset = infer_dataset_spec(str(dataset_dir), dataset_type="text")
    results = execute_plan(DataProcessingPlan(dataset), initial_profile=profile_dataset(dataset))
    target = tmp_path / "v.json"
    publish_dataset_version(dataset, results, target)
    previous = target.read_bytes()
    def failed_replace(*args):
        raise OSError("simulated crash before commit")
    monkeypatch.setattr(service.os, "replace", failed_replace)
    with pytest.raises(OSError):
        publish_dataset_version(dataset, results, target)
    assert target.read_bytes() == previous
    assert not list(tmp_path.glob(".dataset-*.tmp"))


def test_advisor_prompt_exposes_coverage_and_scoped_feedback(monkeypatch, tmp_path):
    from agent.agents.data_processing_agent import DataProcessingAgent
    agent = DataProcessingAgent.__new__(DataProcessingAgent)
    captured = []
    monkeypatch.setattr(agent, "_request_json_advice", lambda prompt: captured.append(json.loads(prompt)) or {})
    scoped = normalize_feedback(feedback(tmp_path), str(tmp_path))
    agent._planning_advice({"hpo_feedback": scoped, "profile": {
        "extensions": {"audio": {"probe_limit": 10, "probe_limited": True, "signal_probe_count": 2}}}})
    context = captured[0]["context"]
    assert context["profile_coverage"]["probe_limited"] is True
    assert context["hpo_feedback"] == scoped


def test_compact_orchestration_record_preserves_blocking_and_data_decisions():
    from agent.agents.orchestrator import CoordinatorAgent
    blocked = {"dispatch_status": "blocked", "blocked_by": ["data_processing_agent"]}
    assert CoordinatorAgent._compact_agent_summary("hpo_agent", blocked) == blocked
    summary = {"data_handoff": {}, "data_decision": {"status": "skipped"}, "feedback_status": "absent"}
    assert CoordinatorAgent._compact_agent_summary("data_processing_agent", summary) == summary


def test_real_audio_filter_guard_before_materialization(tmp_path):
    import wave
    from agent.data_processing.service import build_processing_plan
    source = tmp_path / "audio"
    source.mkdir()
    for name, frames in (("short", 160), ("normal", 16000)):
        with wave.open(str(source / f"{name}.wav"), "wb") as stream:
            stream.setparams((1, 2, 16000, frames, "NONE", "not compressed"))
            stream.writeframes(b"\x20\x00" * frames)
    dataset = infer_dataset_spec(str(source), dataset_type="audio")
    policy = QualityPolicy(max_operation_affected_fraction=.25)
    plan = build_processing_plan(profile_dataset(dataset, policy), requested_operations=[
        {"operation": "filter_by_duration", "parameters": {"min_seconds": .1, "max_seconds": 10}}
    ], policy=policy)
    results = execute_plan(plan, output_root=tmp_path / "derived")
    assert results[-1].details["rejection_reason"] == "affected_fraction_exceeded"
    assert len(list(source.glob("*.wav"))) == 2
    assert not list((tmp_path / "derived").rglob("*.wav"))
    assert not list((tmp_path / "derived").rglob("*.selection.jsonl"))


def test_bad_coordination_advice_does_not_prevent_deterministic_execution():
    registry, calls = AgentRegistry(), []
    register(registry, "agent", calls)
    result = workflow(registry, lambda *args: ["wrong type"]).run({}, {})
    assert result["completion"]["complete"] and calls == ["agent"]
    assert "advice_error" in result["advice"]
