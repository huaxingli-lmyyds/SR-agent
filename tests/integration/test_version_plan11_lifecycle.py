"""Real agent/record boundaries; no external LLM or training calls."""

import json

import pytest

from agent.agents.communication import AgentTaskRequest
from agent.agents.coordination import AgentRegistration, AgentRegistry
from agent.agents.data_processing_agent import DataProcessingAgent
from agent.agents.orchestrator import CoordinatorAgent
from agent.memory import MemoryService


def test_feedback_and_decision_survive_agent_and_record_boundary(
    tmp_path, minimal_config, dataset_dir, monkeypatch
):
    agent = DataProcessingAgent(
        config_path=str(minimal_config),
        experiments_dir=tmp_path / "dp",
        memory_service=MemoryService(tmp_path / "memory"),
        enable_llm_advisor=True,
        verbose=False,
    )
    captured = []
    monkeypatch.setattr(
        agent, "_planning_advice", lambda state: captured.append(state) or {}
    )
    result = agent.run_workflow(
        AgentTaskRequest(
            agent.action,
            "validate",
            context={
                "dataset_uri": str(dataset_dir),
                "data_processing": {"consultation_mode": "evidence"},
                "hpo_feedback": {
                    "dataset_uri": str(dataset_dir),
                    "study_id": "old_study",
                    "trial_count": 2,
                    "evaluation_role": "hpo_validation",
                    "diagnostics": ["inspect durations"],
                },
            },
        )
    )
    assert (
        result.status == "success"
        and result.summary["feedback_status"] == "accepted"
    )
    assert captured[0]["hpo_feedback"]["evidence"]["study_id"] == "old_study"
    record = json.loads(
        next((tmp_path / "dp").glob("*/experiment_record.json")).read_text(
            encoding="utf-8"
        )
    )
    assert (
        record["extensions"]["data_lifecycle"]["decision"]["status"]
        == "completed"
    )
    assert (
        record["extensions"]["data_processing"]["hpo_feedback"]["status"]
        == "accepted"
    )


def test_invalid_registry_marks_management_record_failed(
    tmp_path, minimal_config, dataset_dir
):
    registry = AgentRegistry()
    registry.register(
        AgentRegistration(
            "bad",
            ("run",),
            lambda: pytest.fail("must not construct"),
            requires=("missing",),
        )
    )
    agent = CoordinatorAgent(
        config_path=str(minimal_config),
        output_dir=str(tmp_path / "out"),
        registry=registry,
        verbose=False,
    )
    with pytest.raises(ValueError, match="missing dependencies"):
        agent.run(context={"data_folder": str(dataset_dir)})
    record = json.loads(
        next(
            (tmp_path / "out/manage").glob("*/experiment_record.json")
        ).read_text(encoding="utf-8")
    )
    assert (
        record["status"] == "failed"
        and "missing dependencies" in record["error"]
    )


def test_data_workflow_exception_does_not_leave_running_record(
    tmp_path, minimal_config, dataset_dir
):
    agent = DataProcessingAgent(
        config_path=str(minimal_config),
        experiments_dir=tmp_path / "dp",
        memory_service=MemoryService(tmp_path / "memory"),
        verbose=False,
    )
    with pytest.raises(ValueError, match="consultation_mode"):
        agent.run_workflow(
            AgentTaskRequest(
                agent.action,
                "validate",
                context={
                    "dataset_uri": str(dataset_dir),
                    "data_processing": {"consultation_mode": "invalid"},
                },
            )
        )
    record = json.loads(
        next((tmp_path / "dp").glob("*/experiment_record.json")).read_text(
            encoding="utf-8"
        )
    )
    assert (
        record["status"] == "failed" and "consultation_mode" in record["error"]
    )
