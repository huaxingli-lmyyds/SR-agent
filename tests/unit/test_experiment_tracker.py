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
