import json

from agent.utils.experiment_tracker import ExperimentTracker


def test_tracker_writes_model_aware_version_manifest(tmp_path, minimal_config, dataset_dir) -> None:
    root = tmp_path / "hpo"
    tracker = ExperimentTracker(root)
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
        task={"type": "speaker_verification", "dataset": str(dataset_dir), "primary_metric": "eer"},
        model={"family": "ecapa_tdnn", "implementation": "speechbrain"},
        execution={"runner": "speechbrain"},
        extra_fields={"version": {"campaign_id": "campaign_001", "study_index": 2}},
    )

    manifest = json.loads((root / experiment_id / "version_manifest.json").read_text(encoding="utf-8"))
    pointer = (
        root
        / "_catalog"
        / "speaker_verification"
        / "ecapa_tdnn"
        / "speechbrain"
        / "campaign_001"
        / f"{experiment_id}.json"
    )

    assert manifest["campaign_id"] == "campaign_001"
    assert manifest["study_index"] == 2
    assert manifest["scope"]["model_family"] == "ecapa_tdnn"
    assert pointer.exists()


def test_catalog_pointer_tracks_terminal_status(tmp_path, minimal_config, dataset_dir) -> None:
    root = tmp_path / "hpo"
    tracker = ExperimentTracker(root)
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
        model={"family": "resnetse34", "implementation": "custom"},
        extra_fields={"version": {"campaign_id": "campaign_002", "study_index": 1}},
    )

    assert tracker.update_hpo_experiment(
        experiment_id,
        status="success",
        metrics={"test": {"eer": 0.03}},
    )
    pointer = next((root / "_catalog").rglob(f"{experiment_id}.json"))
    value = json.loads(pointer.read_text(encoding="utf-8"))

    assert value["status"] == "success"
    assert value["metrics"]["test"]["eer"] == 0.03
