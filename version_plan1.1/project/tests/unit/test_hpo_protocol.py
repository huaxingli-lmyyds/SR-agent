import pytest

from agent.hpo import HPOService, Objective, SearchParameter, SearchSpace, TrialBudget
from agent.hpo.protocol import (
    METRIC_PROTOCOL_ID,
    resolve_hpo_validation_protocol,
)


def _inputs(tmp_path):
    config = tmp_path / "validation.yaml"
    config.write_text("verification_file: unused.txt\n", encoding="utf-8")
    pairs = tmp_path / "validation_pairs.txt"
    pairs.write_text(
        "1 id00001/session/a.wav id00001/session/b.wav\n"
        "0 id00001/session/a.wav id00002/session/c.wav\n",
        encoding="utf-8",
    )
    return config, pairs


@pytest.mark.parametrize(
    "runtime,missing",
    [
        ({}, "verification_config"),
        ({"verification_config": "x"}, "validation_pairs"),
    ],
)
def test_hpo_protocol_requires_explicit_validation_inputs(tmp_path, runtime, missing):
    if runtime:
        config, _ = _inputs(tmp_path)
        runtime["verification_config"] = str(config)
    with pytest.raises(ValueError, match=missing):
        resolve_hpo_validation_protocol(runtime, require_explicit=True)


def test_hpo_protocol_rejects_test_pairs_and_incomplete_training_exclusion(tmp_path):
    config, pairs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="test_pairs cannot be supplied"):
        resolve_hpo_validation_protocol(
            {
                "verification_config": str(config),
                "validation_pairs": str(pairs),
                "training_exclusion_pairs": str(pairs),
                "test_pairs": str(pairs),
            },
            require_explicit=True,
        )

    exclusion = tmp_path / "exclusion.txt"
    exclusion.write_text(
        "1 id99999/session/a.wav id99999/session/b.wav\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not cover all validation speakers"):
        resolve_hpo_validation_protocol(
            {
                "verification_config": str(config),
                "validation_pairs": str(pairs),
                "training_exclusion_pairs": str(exclusion),
            },
            require_explicit=True,
        )


def test_hpo_protocol_requires_explicit_training_exclusion(tmp_path):
    config, pairs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="training_exclusion_pairs"):
        resolve_hpo_validation_protocol(
            {
                "verification_config": str(config),
                "validation_pairs": str(pairs),
            },
            require_explicit=True,
        )


def test_hpo_protocol_persists_hashes_and_rejects_legacy_or_changed_resume(tmp_path):
    config, pairs = _inputs(tmp_path)
    resolved = resolve_hpo_validation_protocol(
        {
            "verification_config": str(config),
            "validation_pairs": str(pairs),
            "training_exclusion_pairs": str(pairs),
        },
        require_explicit=True,
    )
    persisted = {
        "evaluation_config_path": resolved["verification_config"],
        "validation_pairs_path": resolved["validation_pairs"],
        "training_exclusion_pairs_path": resolved["training_exclusion_pairs"],
        **{
            key: resolved[key]
            for key in (
                "verification_config_sha256",
                "validation_pairs_sha256",
                "training_exclusion_pairs_sha256",
                "metric_protocol",
            )
        },
    }
    resumed = resolve_hpo_validation_protocol(
        {}, persisted_execution=persisted, require_explicit=False
    )
    assert resumed["metric_protocol"] == METRIC_PROTOCOL_ID

    with pytest.raises(ValueError, match="legacy or incompatible"):
        resolve_hpo_validation_protocol(
            {},
            persisted_execution={
                **persisted,
                "metric_protocol": None,
            },
            require_explicit=False,
        )

    pairs.write_text(
        pairs.read_text(encoding="utf-8")
        + "0 id00003/session/a.wav id00004/session/b.wav\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation input changed"):
        resolve_hpo_validation_protocol(
            {}, persisted_execution=persisted, require_explicit=False
        )


def test_study_creation_rejects_objective_that_conflicts_with_record(
    tmp_path, minimal_config, dataset_dir
):
    from agent.utils import ExperimentTracker

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
    )
    with pytest.raises(ValueError, match="conflicts with experiment primary_metric"):
        HPOService(tracker).create_study(
            experiment_id,
            SearchSpace([SearchParameter("lr", "categorical", choices=[0.1])]),
            [Objective("min_dcf", "min")],
            [TrialBudget("full", epochs=1)],
            max_training_runs=1,
        )


def test_study_rejects_multiple_objectives_that_scheduler_cannot_honor(
    tmp_path, minimal_config, dataset_dir
):
    from agent.utils import ExperimentTracker

    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config), data_folder=str(dataset_dir)
    )
    with pytest.raises(ValueError, match="exactly one authoritative"):
        HPOService(tracker).create_study(
            experiment_id,
            SearchSpace([SearchParameter("lr", "categorical", choices=[0.1])]),
            [Objective("eer", "min"), Objective("min_dcf", "min")],
            [TrialBudget("full", epochs=1)],
            max_training_runs=1,
        )
