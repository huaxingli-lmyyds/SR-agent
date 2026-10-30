"""Evaluate a completed campaign on any idle GPU in an explicitly selected pool.

Training plans, source fingerprints and winners stay frozen. Evaluation-only
GPU reassignment is explicit and audited by run_hpo_benchmark. No retraining.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.experiments import run_hpo_benchmark as benchmark


def sample_gpus():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, free, utilization = [x.strip() for x in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "free_mib": int(free),
                     "utilization_percent": int(utilization)})
    return rows


def eligible_gpus(observations, pool, occupied, min_free_mib, max_utilization):
    return sorted(
        (row for row in observations if row["index"] in pool
         and row["uuid"] not in occupied and row["free_mib"] >= min_free_mib
         and row["utilization_percent"] <= max_utilization),
        key=lambda row: (row["utilization_percent"], -row["free_mib"], row["index"]),
    )


def pending_jobs(campaign):
    manifest = benchmark.read_json(campaign / "launch_manifest.json")
    pending = []
    for job in manifest["jobs"]:
        output = Path(job["output"])
        plan = benchmark.read_json(output / "benchmark_plan.json")
        results_path = output / "results.json"
        if not results_path.exists():
            raise ValueError("all search jobs must finish before held-out evaluation")
        results = benchmark.read_json(results_path)
        expected = {r["run_id"] for r in plan["run_order"]}
        if {r["run_id"] for r in results} != expected or any(r.get("status") not in {"success", "failed"} for r in results):
            raise ValueError("all search jobs must commit terminal results")
        heldout = output / "heldout_results.json"
        if heldout.exists():
            rows = benchmark.read_json(heldout)
            if {(r["variant"], r["seed"]) for r in rows} == {(r["variant"], r["seed"]) for r in results}:
                continue  # Committed failures are also final; never selectively rerun.
        if not plan["config"].get("parallel_single_gpu"):
            raise ValueError("held-out GPU queue requires parallel_single_gpu plans")
        pending.append({"run_id": job["run_id"], "output": str(output), "state": "pending"})
    return pending


def export_heldout(campaign):
    """Refresh the campaign view from committed per-job results only."""
    manifest = benchmark.read_json(campaign / "launch_manifest.json")
    rows = []
    for job in manifest["jobs"]:
        path = Path(job["output"]) / "heldout_results.json"
        if path.exists():
            rows.extend(benchmark.read_json(path))
    benchmark.write_json(campaign / "heldout_results.json", rows)
    with (campaign / "heldout_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "seed", "status", "eer", "min_dcf", "error", "evaluation_gpu_uuid"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


def run_queue(campaign, pool, *, min_free_mib=4096, max_utilization=10,
              max_wait_seconds=1800, job_timeout_seconds=1800, poll_seconds=10):
    # Hold the same lock as the original campaign supervisor to prevent overlap.
    with ExitStack() as stack:
        stack.enter_context(benchmark.exclusive_run(campaign / "supervisor"))
        jobs = pending_jobs(campaign)
        active = {}
        state = {"stage": "evaluating", "gpu_pool": sorted(pool), "jobs": jobs}
        status_path = campaign / "heldout_queue_status.json"
        waiting_since = None
        try:
            while any(job["state"] in {"pending", "running"} for job in jobs):
                now = time.monotonic()
                for job in jobs:
                    item = active.get(job["run_id"])
                    if item is None:
                        continue
                    child, started = item
                    if child.poll() is None and now - started > job_timeout_seconds:
                        stop_child(child)
                        job["error"] = "held-out worker deadline exceeded"
                    if child.poll() is not None:
                        job["exitcode"] = child.returncode
                        committed = Path(job["output"]) / "heldout_results.json"
                        job["state"] = "finished" if child.returncode == 0 and committed.exists() else "failed"
                        del active[job["run_id"]]
                observations = sample_gpus()
                if not pool.issubset({row["index"] for row in observations}):
                    raise ValueError("selected GPU pool contains unavailable device indices")
                available = eligible_gpus(observations, pool,
                    {job["evaluation_gpu_uuid"] for job in jobs if job["state"] == "running"},
                    min_free_mib, max_utilization)
                for job in jobs:
                    if job["state"] != "pending" or not available:
                        continue
                    gpu = available.pop(0)
                    logs = campaign / "heldout_queue_logs"
                    logs.mkdir(exist_ok=True)
                    with (logs / f'{job["run_id"]}.log').open("ab") as log:
                        child = subprocess.Popen(
                            [sys.executable, str(ROOT / "scripts/experiments/run_hpo_benchmark.py"),
                             "--evaluate-test", "--output-dir", job["output"], "--evaluation-gpu", gpu["uuid"]],
                            cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu["uuid"]},
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    active[job["run_id"]] = (child, time.monotonic())
                    job.update(state="running", pid=child.pid, evaluation_gpu_uuid=gpu["uuid"], physical_gpu=gpu["index"])
                blocked = any(job["state"] == "pending" for job in jobs) and not active
                waiting_since = (waiting_since if waiting_since is not None else now) if blocked else None
                state.update(updated_at=benchmark.now(), observations=observations,
                             waiting_reason="no eligible GPU in selected pool" if blocked else None,
                             wait_seconds=now - waiting_since if waiting_since is not None else 0)
                if blocked and state["wait_seconds"] >= max_wait_seconds:
                    state["stage"] = "resource_wait_timeout"
                    benchmark.write_json(status_path, state)
                    return 2
                benchmark.write_json(status_path, state)
                if any(job["state"] in {"pending", "running"} for job in jobs):
                    time.sleep(poll_seconds)
            state["stage"] = "completed_with_issues" if any(job["state"] == "failed" for job in jobs) else "completed"
            state["updated_at"] = benchmark.now()
            if jobs:
                export_heldout(campaign)
            benchmark.write_json(status_path, state)
            return int(state["stage"] != "completed")
        except BaseException as exc:
            state.update(stage="interrupted", error=type(exc).__name__, updated_at=benchmark.now())
            benchmark.write_json(status_path, state)
            raise
        finally:
            for child, _ in active.values():
                stop_child(child)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="Explicit pool, e.g. 4,5,6,7")
    parser.add_argument("--min-free-mib", type=int, default=4096)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--max-wait-seconds", type=float, default=1800)
    parser.add_argument("--job-timeout-seconds", type=float, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    pool = {int(x) for x in args.gpus.split(",")}
    if (not pool or min(pool) < 0 or args.min_free_mib <= 0
            or not 0 <= args.max_utilization <= 100
            or not 0 < args.max_wait_seconds < float("inf")
            or not 0 < args.job_timeout_seconds < float("inf")):
        parser.error("invalid GPU pool or resource limits")
    campaign = args.campaign_dir.resolve()
    if args.dry_run:
        import json
        print(json.dumps({"gpu_pool": sorted(pool), "pending_jobs": pending_jobs(campaign)}, indent=2))
        return 0
    return run_queue(campaign, pool, min_free_mib=args.min_free_mib,
        max_utilization=args.max_utilization, max_wait_seconds=args.max_wait_seconds,
        job_timeout_seconds=args.job_timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
