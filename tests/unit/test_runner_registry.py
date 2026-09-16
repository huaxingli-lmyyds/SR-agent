from pathlib import Path
import sys

import pytest

from agent.core.contracts import OperationResult
from agent.runners.contracts import collect_training_result, validate_runner_compatibility
from agent.runners.registry import RunnerRegistry
from agent.runners.speechbrain import SpeechBrainRunnerAdapter
import agent.runners.speechbrain as speechbrain_runner_module
from agent.runners.speechbrain_distributed import (
    parse_ddp_devices,
    resolve_ddp_plan,
    training_runtime_signature,
)
from agent.runners.speechbrain_backend import _apply_distributed_batch_size


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


def test_ddp_device_parser_and_explicit_plan_are_deterministic() -> None:
    class FakeDistributed:
        @staticmethod
        def is_nccl_available():
            return True

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 4

    fake_torch = type(
        "FakeTorch", (), {"cuda": FakeCuda(), "distributed": FakeDistributed()}
    )()

    assert parse_ddp_devices("3,1") == [3, 1]
    plan = resolve_ddp_plan(fake_torch, {
        "device": "cuda",
        "ddp_devices": [3, 1],
        "distributed_world_size": 2,
        "distributed_backend": "gloo",
        "batch_size_semantics": "global",
    })

    assert plan["enabled"] is True
    assert plan["devices"] == [3, 1]
    assert plan["world_size"] == 2


def test_ddp_auto_falls_back_when_only_one_gpu_is_visible() -> None:
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda()})()

    plan = resolve_ddp_plan(fake_torch, {"ddp_devices": "auto"})

    assert plan["enabled"] is False
    assert plan["world_size"] == 1
    assert plan["fallback_reason"] == "fewer_than_two_visible_cuda_devices"


@pytest.mark.parametrize("value", ["0", "0,0", "-1,1", "0,x", [0, True]])
def test_ddp_rejects_invalid_explicit_devices(value) -> None:
    with pytest.raises(ValueError):
        parse_ddp_devices(value)


def test_training_runtime_signature_separates_ddp_from_single_gpu() -> None:
    single = training_runtime_signature({"precision": "fp16"})
    ddp = training_runtime_signature({
        "precision": "fp16",
        "ddp_plan": {
            "enabled": True,
            "world_size": 2,
            "backend": "nccl",
            "batch_size_semantics": "global",
        },
    })

    assert single["world_size"] == 1
    assert ddp["world_size"] == 2
    assert single != ddp


def test_training_runtime_signature_handles_persisted_ddp_fields() -> None:
    signature = training_runtime_signature({
        "distributed_world_size": 4,
        "distributed_backend": "nccl",
        "batch_size_semantics": "per_device",
    })

    assert signature == {
        "distributed": True,
        "world_size": 4,
        "backend": "nccl",
        "batch_size_semantics": "per_device",
        "precision": "fp32",
        "eval_precision": "fp32",
    }


def test_ddp_global_batch_is_split_without_changing_effective_batch() -> None:
    hparams = {
        "batch_size": 32,
        "dataloader_options": {"batch_size": 32, "num_workers": 4},
    }

    _apply_distributed_batch_size(hparams, {
        "enabled": True,
        "world_size": 4,
        "batch_size_semantics": "global",
    })

    assert hparams["batch_size"] == 8
    assert hparams["batch_size_per_device"] == 8
    assert hparams["global_batch_size"] == 32
    assert hparams["dataloader_options"] == {"batch_size": 8, "num_workers": 4}


def test_ddp_global_batch_rejects_non_divisible_value() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        _apply_distributed_batch_size(
            {"batch_size": 30},
            {
                "enabled": True,
                "world_size": 4,
                "batch_size_semantics": "global",
            },
        )


def test_speechbrain_runner_routes_optional_ddp_before_single_process_timeout(
    monkeypatch,
) -> None:
    import agent.runners.speechbrain_distributed as distributed_module

    captured = {}

    def fake_run(config_path, overrides):
        captured.update({"config_path": config_path, "overrides": overrides})
        return {"status": "success", "valid_error_rate": 0.2}

    monkeypatch.setattr(distributed_module, "run_distributed_training", fake_run)
    overrides = {
        "_hpo_max_duration_seconds": 5,
        "_run_opts": {"ddp_devices": [0, 1]},
    }

    result = SpeechBrainRunnerAdapter().run_training("train.yaml", overrides)

    assert result["status"] == "success"
    assert captured == {"config_path": "train.yaml", "overrides": overrides}


def test_ddp_auto_single_gpu_fallback_keeps_timeout_guard(monkeypatch) -> None:
    import agent.runners.speechbrain as runner_module
    import agent.runners.speechbrain_distributed as distributed_module

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda()})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    captured = {}

    def guarded(config_path, overrides):
        captured.update({"config_path": config_path, "overrides": overrides})
        return {"status": "success", "valid_error_rate": 0.2}

    monkeypatch.setattr(
        runner_module, "_run_speechbrain_training_with_timeout", guarded
    )
    overrides = {
        "_hpo_max_duration_seconds": 5,
        "_run_opts": {
            "ddp_devices": "auto",
            "distributed_backend": "nccl",
            "batch_size_semantics": "global",
        },
    }

    result = distributed_module.run_distributed_training("train.yaml", overrides)

    assert result["status"] == "success"
    assert result["runtime"]["ddp"]["fallback_reason"] == (
        "fewer_than_two_visible_cuda_devices"
    )
    assert captured["overrides"]["_hpo_max_duration_seconds"] == 5
    assert captured["overrides"]["_run_opts"] == {}
