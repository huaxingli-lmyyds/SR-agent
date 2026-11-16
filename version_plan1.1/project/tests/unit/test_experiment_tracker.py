import json
from pathlib import Path

import pytest

from agent.core.contracts import Artifact, OperationResult
from agent.core.experiment_service import ExperimentService
from agent.utils.experiment_tracker import ExperimentTracker


def _create_record(tracker, config, dataset, model_family, runner, eer):
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(config),
        data_folder=str(dataset),
        task={
            "type": "speaker_verification",
            "dataset": str(dataset),
            "primary_metric": "eer",
            "metric_mode": "min",
        },
        model={"family": model_family, "implementation": "fake"},
        execution={"runner": runner},
    )
    result = OperationResult(
        status="success",
        stage="evaluation",
        task={"type": "speaker_verification", "dataset": str(dataset)},
        model={"family": model_family, "implementation": "fake"},
        execution={"runner": runner},
        metrics={"test": {"eer": eer}},
        artifacts=[Artifact("report", "evaluation", f"{experiment_id}.json")],
    )
    assert ExperimentService(tracker).record_result(experiment_id, result)
    return experiment_id


def test_custom_store_never_falls_back_to_global_records(tmp_path, monkeypatch, minimal_config, dataset_dir):
    from agent.utils import experiment_tracker as module

    foreign = ExperimentTracker(tmp_path / "global")
    experiment_id = _create_record(foreign, minimal_config, dataset_dir, "ecapa_tdnn", "fake", 0.01)
    isolated = ExperimentTracker(tmp_path / "isolated")
    monkeypatch.setattr(module, "get_experiment_type_dir", lambda kind: foreign.experiments_dir)
    before = (foreign.experiments_dir / experiment_id / "experiment_record.json").read_bytes()
    assert isolated.get_experiment(experiment_id) is None
    assert not isolated.update_experiment(experiment_id, status="failed")
    assert isolated.get_experiment(f"../global/{experiment_id}") is None
    assert (foreign.experiments_dir / experiment_id / "experiment_record.json").read_bytes() == before
    # The default (unscoped) tracker still supports lookup across experiment types.
    monkeypatch.setattr(module, "get_hpo_experiments_dir", lambda: tmp_path / "default")
    assert ExperimentTracker().get_experiment(experiment_id)["experiment_id"] == experiment_id


def test_experiment_lifecycle_and_scoped_best_query(tmp_path, minimal_config, dataset_dir) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    expected = _create_record(tracker, minimal_config, dataset_dir, "ecapa_tdnn", "fake", 0.04)
    _create_record(tracker, minimal_config, dataset_dir, "other_model", "fake", 0.01)

    best = tracker.find_best_experiment(
        metric="eer",
        model_family="ecapa_tdnn",
        dataset=str(dataset_dir),
        implementation="fake",
        runner="fake",
    )

    assert best[0]["experiment_id"] == expected
    assert best[0]["status"] == "success"
    assert best[0]["metrics"]["test"]["eer"] == 0.04


def test_best_metric_split_is_queryable(tmp_path, minimal_config, dataset_dir) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
        task={
            "type": "speaker_verification",
            "dataset": str(dataset_dir),
            "primary_metric": "eer",
            "metric_mode": "min",
        },
        model={"family": "ecapa_tdnn", "implementation": "fake"},
        execution={"runner": "fake"},
    )
    tracker.update_hpo_experiment(
        experiment_id,
        status="success",
        metrics={"best": {"primary_metric": "eer", "primary_value": 0.02}},
    )

    best = tracker.find_best_experiment(metric="eer", model_family="ecapa_tdnn")

    assert best[0]["experiment_id"] == experiment_id


@pytest.mark.parametrize("legacy_reference", [False, True])
def test_config_snapshot_survives_custom_execution_and_source_changes(
    tmp_path, minimal_config, dataset_dir, legacy_reference,
) -> None:
    original = minimal_config.read_text(encoding="utf-8")
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config), data_folder=str(dataset_dir),
        execution={"runner": "fake", "config_backup_path": "not-the-snapshot.yaml"},
    )
    record = tracker.get_experiment(experiment_id)
    snapshot = tracker.get_config_snapshot(experiment_id)
    assert Path(record["execution"]["config_backup_path"]) == snapshot
    assert record["execution"]["runner"] == "fake"
    if legacy_reference:
        record["execution"].pop("config_backup_path")
        record_path = tracker.experiments_dir / experiment_id / "experiment_record.json"
        assert record_path.is_file()
        record_path.write_text(json.dumps(record), encoding="utf-8")
    minimal_config.write_text("changed: true\n", encoding="utf-8")
    assert tracker.get_config_snapshot(experiment_id).read_text(encoding="utf-8") == original
    minimal_config.unlink()
    assert tracker.get_config_snapshot(experiment_id) == snapshot


def test_missing_snapshot_fails_instead_of_using_mutable_source(
    tmp_path, minimal_config, dataset_dir,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config), data_folder=str(dataset_dir),
    )
    tracker.get_config_snapshot(experiment_id).unlink()
    assert minimal_config.is_file()
    with pytest.raises(FileNotFoundError, match="config snapshot missing"):
        tracker.get_config_snapshot(experiment_id)
