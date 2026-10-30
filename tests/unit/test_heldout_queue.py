import json

import pytest

from scripts.experiments import run_heldout_queue as queue
from scripts.experiments import run_hpo_benchmark as benchmark

GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"


def test_evaluation_can_use_any_idle_gpu_in_explicit_pool():
    rows = [
        {"index": 4, "uuid": GPU_A, "free_mib": 21000, "utilization_percent": 95},
        {"index": 5, "uuid": GPU_B, "free_mib": 8000, "utilization_percent": 0},
    ]
    assert queue.eligible_gpus(rows, {4, 5}, set(), 4096, 10) == [rows[1]]
    assert queue.eligible_gpus(rows, {4}, set(), 4096, 10) == []
    assert queue.eligible_gpus(rows, {4, 5}, {GPU_B}, 4096, 10) == []


def plan(tmp_path, monkeypatch):
    exclusion = tmp_path / "exclude.txt"
    exclusion.write_text("frozen")
    monkeypatch.setattr(benchmark, "package_versions", lambda: {})
    return {"schema_version": benchmark.SCHEMA_VERSION, "output_root": str(tmp_path),
            "config": {"parallel_single_gpu": True, "device": "cuda:0"},
            "gpu_assignment": {"physical_gpu_uuid": GPU_A},
            "environment": {"packages": {}, "source_sha256": {}},
            "frozen_inputs": {}, "input_sha256": {},
            "training_exclusions": str(exclusion), "training_exclusions_sha256": benchmark.file_hash(exclusion)}


def test_evaluation_rebinding_preserves_training_plan_and_hash_checks(tmp_path, monkeypatch):
    frozen = plan(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_B)
    with pytest.raises(ValueError, match="GPU assignment changed"):
        benchmark.verify_frozen(frozen, tmp_path)
    benchmark.verify_frozen(frozen, tmp_path, evaluation_gpu=GPU_B)
    assert frozen["gpu_assignment"] == {"physical_gpu_uuid": GPU_A}
    frozen["environment"]["source_sha256"]["README.md"] = "wrong"
    with pytest.raises(ValueError, match="source changed"):
        benchmark.verify_frozen(frozen, tmp_path, evaluation_gpu=GPU_B)


def test_rebinding_is_never_allowed_in_training_mode():
    with pytest.raises(ValueError, match="requires --evaluate-test"):
        benchmark.main(["--resume", "--evaluation-gpu", GPU_B])


def test_mismatching_evaluation_binding_rejected(tmp_path, monkeypatch):
    frozen = plan(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_A)
    with pytest.raises(ValueError, match="matching single-GPU"):
        benchmark.verify_frozen(frozen, tmp_path, evaluation_gpu=GPU_B)


def test_queue_wait_has_deadline_and_durable_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "pending_jobs", lambda _: [{"run_id": "job", "output": "unused", "state": "pending"}])
    monkeypatch.setattr(queue, "sample_gpus", lambda: [{"index": 4, "uuid": GPU_A, "free_mib": 100, "utilization_percent": 100}])
    now = [0.0]
    monkeypatch.setattr(queue.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert queue.run_queue(tmp_path, {4}, max_wait_seconds=20, poll_seconds=10) == 2
    status = json.loads((tmp_path / "heldout_queue_status.json").read_text())
    assert status["stage"] == "resource_wait_timeout"
    assert status["waiting_reason"] == "no eligible GPU in selected pool"
    assert status["jobs"][0]["state"] == "pending"


def campaign(tmp_path, *, finished=False):
    output = tmp_path / "jobs" / "test"
    output.mkdir(parents=True)
    benchmark.write_json(tmp_path / "launch_manifest.json", {"jobs": [{"run_id": "test", "output": str(output)}]})
    benchmark.write_json(output / "benchmark_plan.json", {"config": {"parallel_single_gpu": True},
        "run_order": [{"run_id": "test"}]})
    if finished:
        benchmark.write_json(output / "results.json", [{"run_id": "test", "variant": "system", "seed": 1, "status": "success"}])
    return output


def test_global_test_barrier_is_kept(tmp_path):
    campaign(tmp_path)
    with pytest.raises(ValueError, match="all search jobs"):
        queue.pending_jobs(tmp_path)


def test_committed_heldout_failures_are_not_selectively_rerun(tmp_path):
    output = campaign(tmp_path, finished=True)
    assert len(queue.pending_jobs(tmp_path)) == 1
    benchmark.write_json(output / "heldout_results.json", [{"variant": "system", "seed": 1, "status": "failed"}])
    assert queue.pending_jobs(tmp_path) == []
