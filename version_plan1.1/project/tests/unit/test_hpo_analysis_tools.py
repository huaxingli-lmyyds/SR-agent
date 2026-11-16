import json

import pytest

pytest.importorskip("langchain_core")

from agent.hpo import HPOService, Objective, SearchParameter, SearchSpace, TrialBudget
from agent.hpo.protocol import METRIC_PROTOCOL_ID, METRIC_UNITS
from agent.tools.hpo_analysis_tools import build_hpo_analysis_tools
from agent.utils.experiment_tracker import ExperimentTracker
import agent.hpo.service as hpo_service_module


def test_hpo_analysis_tools_are_read_only_and_see_pending_candidates(
    tmp_path,
    minimal_config,
    dataset_dir,
    monkeypatch,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
    )
    monkeypatch.setattr(
        hpo_service_module,
        "get_experiment_artifact_dir",
        lambda *args, **kwargs: tmp_path / "hpo_artifacts",
    )
    service = HPOService(tracker)
    study = service.create_study(
        experiment_id,
        SearchSpace([SearchParameter("lr", "float", low=0.0, high=1.0)]),
        [Objective("eer", "min")],
        [TrialBudget("full", epochs=1)],
        strategy="agent_proposal",
        max_training_runs=2,
        candidate_proposals=[{"parameters": {"lr": 0.25}}],
        proposal_id="proposal_a",
    )
    before = service.load_study(experiment_id).to_dict()
    tools = {item.name: item for item in build_hpo_analysis_tools(tracker)}

    duplicate = json.loads(tools["validate_hpo_candidate"].invoke({
        "experiment_id": experiment_id,
        "parameters": {"lr": 0.25},
    }))
    valid = json.loads(tools["validate_hpo_candidate"].invoke({
        "experiment_id": experiment_id,
        "parameters": {"lr": 0.5},
    }))
    inspected = json.loads(tools["inspect_hpo_study"].invoke({
        "experiment_id": experiment_id,
    }))
    wrong_metric = json.loads(tools["compare_hpo_trials"].invoke({
        "experiment_id": experiment_id,
        "metric": "min_dcf",
    }))

    assert duplicate == {"valid": False, "duplicate": True, "parameters": {"lr": 0.25}}
    assert valid == {"valid": True, "duplicate": False, "parameters": {"lr": 0.5}}
    assert inspected["study"]["pending_agent_candidate_count"] == 1
    assert wrong_metric["authoritative_metric"] == "eer"
    assert service.load_study(experiment_id).to_dict() == before

    traversal = json.loads(tools["inspect_hpo_study"].invoke({
        "experiment_id": "../outside",
    }))
    assert "invalid experiment_id" in traversal["error"]


def test_agent_history_tool_requires_exact_frozen_confirmation_signature(
    tmp_path, minimal_config, dataset_dir
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    signature = {
        "objective": {"metric": "eer", "mode": "min"},
        "final_budget": {
            "epochs": 10,
            "data_fraction": 1.0,
            "max_duration_seconds": None,
        },
        "metric_protocol": METRIC_PROTOCOL_ID,
        "metric_units": dict(METRIC_UNITS),
        "dataset": str(dataset_dir),
        "dataset_id": "vox1",
        "dataset_version": "v1",
        "task_type": "speaker_verification",
        "model_family": "unknown",
        "implementation": "speechbrain",
        "runner": "speechbrain",
        "config_sha256": "config-a",
        "verification_config_sha256": "verification-a",
        "validation_pairs_sha256": "validation-a",
        "training_exclusion_pairs_sha256": "exclusion-a",
    }
    legacy_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
        task={
            "type": "speaker_verification",
            "dataset": str(dataset_dir),
            "primary_metric": "eer",
            "metric_mode": "min",
        },
    )
    current_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
        task={
            "type": "speaker_verification",
            "dataset": str(dataset_dir),
            "primary_metric": "eer",
            "metric_mode": "min",
            "metric_protocol": METRIC_PROTOCOL_ID,
        },
    )
    different_budget_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
        task={
            "type": "speaker_verification",
            "dataset": str(dataset_dir),
            "primary_metric": "eer",
            "metric_mode": "min",
            "metric_protocol": METRIC_PROTOCOL_ID,
        },
    )
    tracker.update_hpo_experiment(
        current_id,
        status="success",
        metrics={"best": {"primary_value": 0.2, "eer": 0.2}},
        extensions={"optimization": {"campaign": {
            "confirmation_signature": signature,
        }}},
    )
    incompatible = dict(signature)
    incompatible["final_budget"] = dict(signature["final_budget"], epochs=5)
    tracker.update_hpo_experiment(
        different_budget_id,
        status="success",
        metrics={"best": {"primary_value": 0.25, "eer": 0.25}},
        extensions={"optimization": {"campaign": {
            "confirmation_signature": incompatible,
        }}},
    )
    tools = {
        item.name: item
        for item in build_hpo_analysis_tools(
            tracker, confirmation_signature=signature
        )
    }
    rows = json.loads(tools["list_comparable_hpo_experiments"].invoke({
        "model_family": "unknown",
        "dataset": str(dataset_dir),
        "limit": 10,
    }))

    assert [row["experiment_id"] for row in rows] == [current_id]
    assert legacy_id not in {row["experiment_id"] for row in rows}
    assert different_budget_id not in {row["experiment_id"] for row in rows}


def test_agent_history_tool_rejects_unbound_comparison(
    tmp_path,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    tools = {item.name: item for item in build_hpo_analysis_tools(tracker)}
    result = json.loads(tools["list_comparable_hpo_experiments"].invoke({}))

    assert "frozen confirmation_signature" in result["error"]
