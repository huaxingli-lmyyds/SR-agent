"""Exercise public tool interfaces rather than private forwarding functions."""

import json

import pytest

from agent.core.contracts import OperationResult
from agent.runners import RUNNER_ADAPTERS
from agent.tasks import TASK_ADAPTERS
from agent.tools import evaluation_tools, training_tools
from agent.tools.dataset_tools import BuildDataProcessingPlan, InspectDataset
from agent.tools.speechbrain_data_tools import PrepareVoxCelebData
from agent.utils.experiment_tracker import ExperimentTracker
from tests.fakes import FakeRunnerAdapter


@pytest.mark.parametrize("explicit", [False, True])
def test_preparation_tool_uses_training_yaml_and_explicit_overrides(
    tmp_path, monkeypatch, explicit
):
    from agent.runners import speechbrain_backend
    from agent.tools import speechbrain_data_tools

    tracker = ExperimentTracker(tmp_path / "experiments")
    monkeypatch.setattr(
        speechbrain_data_tools, "ExperimentTracker", lambda: tracker
    )
    config = tmp_path / "train.yaml"
    output = tmp_path / "prepared"
    output.mkdir()
    config.write_text(f"""
data_folder: {tmp_path}
save_folder: {output}
verification_file: {tmp_path / "pairs.txt"}
split_ratio: [90, 10]
split_unit: session
data_prep_seed: 0
random_chunk: true
amp_th: 0
""")
    captured = {}

    def prepare(**options):
        captured.update(options)
        (output / "train.csv").write_text(
            'ID,wav,spk_id\n"multiline\nID",a.wav,id1\n'
        )
        return {"status": "success", "save_folder": str(output)}

    monkeypatch.setattr(speechbrain_backend, "run_data_prep", prepare)
    options = {"config_path": str(config)}
    if explicit:
        options.update(
            split_ratio="80,20",
            split_unit="utterance",
            random_segment="false",
            data_prep_seed=7,
        )
    result = json.loads(PrepareVoxCelebData.invoke(options))
    assert result["status"] == "success", result.get("error")
    assert captured["split_unit"] == ("utterance" if explicit else "session")
    assert captured["split_ratio"] == ([80, 20] if explicit else [90, 10])
    assert captured["random_segment"] is not explicit
    assert captured["data_prep_seed"] == (7 if explicit else 0)
    assert captured["amp_th"] == 0
    assert result["metrics"]["summary"]["train"] == 1
    assert result["parameters"]["split_unit"] == captured["split_unit"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"split_ratio": "0,100"},
        {"split_ratio": "90,-10"},
        {"split_ratio": "nan,10"},
        {"split_ratio": "90"},
        {"skip_prep": "maybe"},
        {"split_unit": "unknown"},
        {"split_unit": "session", "split_speaker": "true"},
    ],
)
def test_preparation_tool_rejects_invalid_options_before_backend(
    tmp_path, monkeypatch, arguments
):
    from agent.runners import speechbrain_backend
    from agent.tools import speechbrain_data_tools

    monkeypatch.setattr(
        speechbrain_data_tools, "ExperimentTracker", lambda: object()
    )
    monkeypatch.setattr(
        speechbrain_backend,
        "run_data_prep",
        lambda **_: pytest.fail("backend should not run"),
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"save_folder: {tmp_path / 'save'}\nverification_file: pairs.txt\n"
    )
    result = json.loads(
        PrepareVoxCelebData.invoke({"config_path": str(config), **arguments})
    )
    assert result["status"] == "failed"
    assert "ValueError" in result["error"]


@pytest.mark.parametrize("tool_name", ["RunEvaluation", "EvaluateModel"])
def test_evaluation_tools_support_task_metrics_and_preserve_runner_fields(
    tmp_path, minimal_config, dataset_dir, monkeypatch, tool_name
):
    class ClassificationTask:
        task_type = "classification"
        primary_metric = "accuracy"
        metric_mode = "max"

        def validate_metrics(self, metrics):
            assert metrics == {"accuracy": 0.9}

    captured = {}

    class ClassificationRunner(FakeRunnerAdapter):
        def run_evaluation(self, *args, **kwargs):
            return {
                "status": "success",
                "metrics": {"accuracy": 0.9},
                "scores_path": str(tmp_path / "classification.csv"),
                "timeout_seconds": 12,
                "process_exitcode": 0,
                "runtime": {"worker": "classification"},
            }

        def normalize_evaluation_result(self, raw):
            captured.update(raw)
            return OperationResult(
                status=raw["status"],
                stage="evaluation",
                metrics={"test": raw["metrics"]},
                extensions={"custom": raw["runtime"]},
            )

    tracker = ExperimentTracker(tmp_path / "experiments")
    monkeypatch.setattr(
        evaluation_tools, "ExperimentTracker", lambda *_: tracker
    )
    monkeypatch.setitem(TASK_ADAPTERS, "classification", ClassificationTask())
    monkeypatch.setitem(
        RUNNER_ADAPTERS, "classification", ClassificationRunner()
    )
    config = str(minimal_config)
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("0 id1/a.wav id2/b.wav\n")
    (tmp_path / "classification.csv").write_text("label,prediction\na,a\n")
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_text("fake")
    experiment = tracker.create_hpo_experiment(
        config_path=config,
        data_folder=str(dataset_dir),
        task={"type": "classification", "dataset": str(dataset_dir)},
        model={"family": "ecapa_tdnn", "implementation": "fake"},
        execution={"runner": "classification"},
    )
    result = json.loads(
        getattr(evaluation_tools, tool_name).invoke(
            {
                "experiment_id": experiment,
                "model_path": str(checkpoint),
                "verification_config": config,
                "verification_pairs": str(pairs),
                "experiments_dir": str(tracker.experiments_dir),
            }
        )
    )
    assert result["status"] == "success", result.get("error")
    assert result["metrics"] == {"test": {"accuracy": 0.9}}
    assert result["task"]["metric_units"] == {}
    assert captured["timeout_seconds"] == 12
    assert captured["process_exitcode"] == 0
    assert result["extensions"]["custom"]["worker"] == "classification"
    assert training_tools.EvaluateModel.name == "EvaluateModel"


def test_inspection_tool_preserves_policy_and_scope_in_plan_json(tmp_path):
    (tmp_path / "wav" / "id10001").mkdir(parents=True)
    pairs = tmp_path / "exclude.txt"
    pairs.write_text("1 id10001/a.wav id10001/b.wav\n")
    profile = InspectDataset.invoke(
        {
            "dataset_uri": str(tmp_path),
            "dataset_type": "audio",
            "task_type": "speaker_verification",
            "training_exclusion_pairs": str(pairs),
            "quality_policy_json": json.dumps({"max_audio_files": 5}),
        }
    )
    payload = json.loads(profile)
    assert payload["dataset"]["metadata"]["contains_protected_speakers"]
    assert payload["extensions"]["quality_policy"]["max_audio_files"] == 5
    result = json.loads(
        BuildDataProcessingPlan.invoke(
            {
                "profile_json": profile,
                "requested_operations_json": json.dumps(
                    [{"operation": "build_debug_subset"}]
                ),
            }
        )
    )
    assert result["status"] == "failed"
    assert "protected evaluation speakers" in result["error"]
