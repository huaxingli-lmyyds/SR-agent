from pathlib import Path

import pytest

from agent.core.contracts import OperationResult
from agent.runners.contracts import collect_training_result, validate_runner_compatibility
from agent.runners.registry import RunnerRegistry
from agent.runners.speechbrain import SpeechBrainRunnerAdapter


class ExternalRunner:
    runner = "external"
    default_evaluation_config = None
    supported_implementations = {"external"}
    supported_model_families = {"demo"}

    def run_training(self, config_path, overrides):
        return {"status": "success"}

    def run_evaluation(self, config_path, model_path, data_path, overrides):
        return {"status": "success"}

    def collect_training_result(self, raw, output_folder, experiment_dir):
        return {
            **raw,
            "metrics": {"accuracy": 0.9},
            "model_paths": [str(experiment_dir / "model.bin")],
            "output_folder": str(output_folder),
        }

    def normalize_training_result(self, raw):
        return OperationResult(status=raw["status"], stage="training", metrics={"validation": raw["metrics"]})

    def normalize_evaluation_result(self, raw):
        return OperationResult(status=raw["status"], stage="evaluation")


def test_runner_registry_supports_external_runtime_without_tool_changes(tmp_path: Path) -> None:
    registry = RunnerRegistry()
    runner = ExternalRunner()
    registry.register(runner)

    collected = collect_training_result(
        registry.get("external"),
        runner.run_training("config.json", {}),
        tmp_path / "output",
        tmp_path,
    )

    assert collected["metrics"]["accuracy"] == 0.9
    assert registry.describe()["external"]["adapter_type"] == "ExternalRunner"


def test_runner_registry_rejects_unknown_runner() -> None:
    with pytest.raises(ValueError, match="unknown runner adapter"):
        RunnerRegistry().get("missing")


def test_runner_compatibility_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="does not support model family"):
        validate_runner_compatibility(ExternalRunner(), model_family="other")


def test_speechbrain_runner_owns_log_and_checkpoint_discovery(tmp_path: Path) -> None:
    output = tmp_path / "output"
    checkpoint = output / "save" / "CKPT+best"
    checkpoint.mkdir(parents=True)
    (checkpoint / "CKPT.yaml").write_text("ErrorRate: 0.05\n", encoding="utf-8")
    (output / "train_log.txt").write_text(
        "epoch: 1, lr: 0.001 - train loss: 0.4 - valid loss: 0.3, valid ErrorRate: 0.05\n",
        encoding="utf-8",
    )

    collected = SpeechBrainRunnerAdapter().collect_training_result(
        {"status": "success"},
        output,
        tmp_path,
    )

    assert collected["model_paths"] == [str(checkpoint)]
    assert collected["metrics"]["best_error_rate"] == 0.05
