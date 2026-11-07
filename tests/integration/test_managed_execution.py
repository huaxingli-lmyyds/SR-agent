import json
import os
from pathlib import Path
import pytest
from agent.agents.hpo_agent import HPOAgent
from agent.agents.communication import AgentTaskRequest
from agent.memory import MemoryService
from agent.runners import RUNNER_ADAPTERS
from tests.managed_fakes import ProcessRunner


def test_process_batch_and_confirmation_are_budgeted_and_resumable(tmp_path, minimal_config, dataset_dir, monkeypatch):
    monkeypatch.setitem(RUNNER_ADAPTERS, "process_fake", ProcessRunner())
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("1 val1/a.wav val1/b.wav\n0 val1/a.wav val2/a.wav\n")
    agent = HPOAgent(config_path=str(minimal_config), runner="process_fake", verbose=False,
        experiments_dir=str(tmp_path / "experiments"), memory_service=MemoryService(tmp_path / "memory"))
    request = AgentTaskRequest(action=agent.action, objective="test", context={
        "data_folder": str(dataset_dir), "strategy": "grid_search", "controller_mode": "fixed",
        "search_space": {"parameters": [{"name": "lr", "parameter_type": "categorical", "choices": [.001, .002]}]},
        "budgets": [{"stage": "full", "epochs": 1, "data_fraction": 1.0}],
        "runtime_options": {"managed_execution": True, "max_concurrent_trials": 2, "device": "cpu",
            "confirmation_enabled": True, "verification_config": str(minimal_config),
            "validation_pairs": str(pairs), "training_exclusion_pairs": str(pairs)}},
        budget={"max_training_runs": 8, "max_total_training_runs": 8, "max_studies": 1})
    result = agent.run_workflow(request)
    assert result.status == "success", result.to_dict()
    confirmation = result.summary["confirmation"]
    assert confirmation["status"] == "completed", confirmation
    assert confirmation["training_runs"] == 6
    assert len(confirmation["groups"]) == 2
    assert all(g["sample_count"] == 3 for g in confirmation["groups"])
    assert result.summary["best_parameters"]["lr"] == .001
    checkpoints = list((tmp_path / "experiments").rglob("fake.ckpt"))
    assert len(checkpoints) == 8
    assert all(json.loads(p.read_text())["pid"] != os.getpid() for p in checkpoints)
    ledger = json.loads(next((tmp_path / "experiments" / "campaign_resources").glob("*.json")).read_text())
    if "stages" not in ledger:
        ledger = next(json.loads(p.read_text()) for p in (tmp_path / "experiments" / "campaign_resources").glob("*.json") if "stages" in json.loads(p.read_text()))
    assert len(ledger["stages"]) == 16
    assert ledger["gpu_seconds"] == 0
    from agent.hpo.service import HPOService
    from agent.hpo.feedback import HPOFeedbackAnalyzer
    from agent.utils.experiment_tracker import ExperimentTracker
    service = HPOService(ExperimentTracker(tmp_path / "experiments"))
    for experiment in {row["experiment_id"] for row in ledger["stages"]}:
        study = service.load_study(experiment)
        feedback = HPOFeedbackAnalyzer().analyze(study, service.list_trials(experiment))
        costs = feedback["cost_summary"]
        rows = [r for r in ledger["stages"] if r["experiment_id"] == experiment]
        assert costs["sources"] == ["campaign_ledger"]
        assert costs["stage_count"] == len(rows)
        assert costs["training_seconds"] == pytest.approx(sum(r["seconds"] for r in rows if r["stage"] == "training"))
        assert costs["evaluation_seconds"] == pytest.approx(sum(r["seconds"] for r in rows if r["stage"] != "training"))
    search_id = result.experiment_ids["campaign"][0]
    resumed = agent.run_workflow(AgentTaskRequest(action=agent.action, objective="resume",
        context={"resume_experiment_id": search_id}))
    assert resumed.status == "success", resumed.to_dict()
    assert len(list((tmp_path / "experiments").rglob("fake.ckpt"))) == 8
    assert resumed.summary["confirmation"]["confirmed_best"] == confirmation["confirmed_best"]
