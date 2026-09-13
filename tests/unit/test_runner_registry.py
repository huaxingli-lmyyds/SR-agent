from pathlib import Path

import pytest

from agent.core.contracts import OperationResult
from agent.runners.contracts import collect_training_result, validate_runner_compatibility
from agent.runners.registry import RunnerRegistry
from agent.runners.speechbrain import SpeechBrainRunnerAdapter
import agent.runners.speechbrain as speechbrain_runner_module


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


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1"])
def test_speechbrain_runner_rejects_invalid_training_deadlines(value) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        SpeechBrainRunnerAdapter().run_training(
            "unused.yaml", {"_hpo_max_duration_seconds": value}
        )


def test_speechbrain_runner_terminates_training_at_deadline(monkeypatch) -> None:
    class FakeConnection:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeProcess:
        exitcode = None

        def __init__(self):
            self.alive = True
            self.joins = []
            self.terminated = False
            self.sentinel = object()

        def start(self):
            pass

        def join(self, timeout):
            self.joins.append(timeout)

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False
            self.exitcode = -15

    receive_connection = FakeConnection()
    send_connection = FakeConnection()
    process = FakeProcess()

    class FakeContext:
        def Pipe(self, duplex):
            assert duplex is False
            return receive_connection, send_connection

        def Process(self, **kwargs):
            assert kwargs["args"][2]["_hpo_max_duration_seconds"] == 2.5
            return process

    monkeypatch.setattr(
        speechbrain_runner_module.multiprocessing,
        "get_context",
        lambda method: FakeContext(),
    )
    monkeypatch.setattr(
        speechbrain_runner_module,
        "wait_for_process_io",
        lambda objects, timeout: [],
    )
    result = SpeechBrainRunnerAdapter().run_training(
        "unused.yaml", {"_hpo_max_duration_seconds": 2.5}
    )

    assert result["status"] == "failed"
    assert result["terminated_by_budget"] is True
    assert result["timeout_seconds"] == 2.5
    assert process.terminated is True
    assert process.joins == [10.0]
    assert receive_connection.closed and send_connection.closed


def test_speechbrain_runner_returns_child_result_before_join(monkeypatch) -> None:
    expected = {"status": "success", "valid_error_rate": 0.2}

    class FakeReceiveConnection:
        def recv(self):
            return expected

        def close(self):
            pass

    class FakeSendConnection:
        def close(self):
            pass

    class FakeProcess:
        exitcode = 0
        sentinel = object()

        def start(self):
            pass

        def join(self, timeout):
            assert timeout == 10.0

        def is_alive(self):
            return False

    class FakeContext:
        def Pipe(self, duplex):
            assert duplex is False
            return receive_connection, FakeSendConnection()

        def Process(self, **kwargs):
            return FakeProcess()

    receive_connection = FakeReceiveConnection()
    monkeypatch.setattr(
        speechbrain_runner_module.multiprocessing,
        "get_context",
        lambda method: FakeContext(),
    )
    monkeypatch.setattr(
        speechbrain_runner_module,
        "wait_for_process_io",
        lambda objects, timeout: [receive_connection],
    )

    assert SpeechBrainRunnerAdapter().run_training(
        "unused.yaml", {"_hpo_max_duration_seconds": 3.0}
    ) == expected


def test_speechbrain_runner_reports_early_child_crash_without_waiting_for_deadline(
    monkeypatch,
) -> None:
    class FakeReceiveConnection:
        def poll(self):
            return False

        def close(self):
            pass

    class FakeSendConnection:
        def close(self):
            pass

    class FakeProcess:
        exitcode = 9
        sentinel = object()

        def start(self):
            pass

        def join(self, timeout):
            assert timeout == 10.0

        def is_alive(self):
            return False

    process = FakeProcess()

    class FakeContext:
        def Pipe(self, duplex):
            return FakeReceiveConnection(), FakeSendConnection()

        def Process(self, **kwargs):
            return process

    monkeypatch.setattr(
        speechbrain_runner_module.multiprocessing,
        "get_context",
        lambda method: FakeContext(),
    )
    monkeypatch.setattr(
        speechbrain_runner_module,
        "wait_for_process_io",
        lambda objects, timeout: [process.sentinel],
    )

    result = SpeechBrainRunnerAdapter().run_training(
        "unused.yaml", {"_hpo_max_duration_seconds": 36000.0}
    )
    assert result["status"] == "failed"
    assert result["process_exitcode"] == 9
    assert result["terminated_by_budget"] is False
