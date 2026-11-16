"""Single-writer process scheduler with durable, replayable stage results."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time

from agent.hpo.records import _write_json_atomic
from agent.hpo.resources import ResourceBudgetExhausted
from .stages import execute_stage


def identity(pid):
    import psutil
    process = psutil.Process(pid)
    return {"pid": pid, "created_at": process.create_time()}


def terminate_owned(owner):
    """Never signal a reused PID; include descendants that inherited the stage token."""
    import psutil
    owned = {}
    try:
        root = psutil.Process(owner["pid"])
        if root.create_time() == owner["created_at"]:
            owned[root.pid] = root
            owned.update({p.pid: p for p in root.children(recursive=True)})
    except psutil.NoSuchProcess:
        pass
    children_path = owner.get("children_path")
    if children_path and Path(children_path).is_file():
        for child in json.loads(Path(children_path).read_text(encoding="utf-8")):
            try:
                process = psutil.Process(child["pid"])
                if process.create_time() == child["created_at"]:
                    owned[process.pid] = process
            except psutil.NoSuchProcess:
                pass
    token = owner.get("token")
    if token:
        for process in psutil.process_iter(["pid", "username"]):
            try:
                if process.environ().get("SR_AGENT_STAGE_TOKEN") == token:
                    owned[process.pid] = process
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    for process in reversed(list(owned.values())):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(owned.values()), timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    if alive:
        raise RuntimeError("cannot confirm owned worker tree has exited")


def _worker(request, result_path, gate, parent, token, devices):
    import psutil
    os.environ["SR_AGENT_STAGE_TOKEN"] = token
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    if os.name != "nt":
        os.setsid()
    def watch_parent():
        while True:
            try:
                process = psutil.Process(parent["pid"])
                alive = process.is_running() and process.create_time() == parent["created_at"]
            except psutil.NoSuchProcess:
                alive = False
            if not alive:
                for child in psutil.Process().children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                os._exit(73)
            children = []
            for child in psutil.Process().children(recursive=True):
                try:
                    children.append(identity(child.pid))
                except psutil.NoSuchProcess:
                    pass
            _write_json_atomic(Path(str(result_path) + ".children.json"), children)
            time.sleep(.25)
    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        if not gate.poll(30) or gate.recv() != "start":
            return
        started = time.time()
        try:
            raw = execute_stage(request)
        except BaseException as exc:
            raw = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        _write_json_atomic(Path(result_path), {"raw": raw, "seconds": time.time()-started, "finished_at": time.time()})
    finally:
        gate.close()


def validate_options(options):
    result = dict(options)
    for key in ("max_concurrent_trials", "gpus_per_trial", "confirmation_top_k", "confirmation_seeds"):
        value = result.get(key, 2 if key == "confirmation_top_k" else 3 if key == "confirmation_seeds" else 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if result.get("ddp_devices") and result.get("max_concurrent_trials", 1) > 1:
        raise ValueError("explicit parallel execution conflicts with ddp_devices")
    if result.get("gpus_per_trial", 1) > 1 and not result.get("gpu_pool"):
        raise ValueError("gpus_per_trial > 1 requires an explicit gpu_pool")
    pool = result.get("gpu_pool")
    if isinstance(pool, str):
        pool = [int(v.strip()) for v in pool.split(",")]
    if pool is not None:
        if not isinstance(pool, list) or not pool or any(type(v) is not int or v < 0 for v in pool) or len(pool) != len(set(pool)):
            raise ValueError("gpu_pool must contain distinct non-negative device IDs")
        if result.get("ddp_devices"):
            raise ValueError("gpu_pool conflicts with ddp_devices")
        if str(result.get("device", "")).startswith("cpu"):
            raise ValueError("gpu_pool conflicts with CPU execution")
        if len(pool) < result.get("gpus_per_trial", 1):
            raise ValueError("gpu_pool cannot satisfy gpus_per_trial")
        result["gpu_pool"] = pool
    if result.get("max_concurrent_trials", 1) > 1 and not pool and not str(result.get("device", "")).startswith("cpu"):
        raise ValueError("parallel GPU execution requires an explicit gpu_pool")
    if result.get("promotion_training", "restart") not in {"restart", "resume_if_compatible"}:
        raise ValueError("unsupported promotion_training")
    if "confirmation_enabled" in result and type(result["confirmation_enabled"]) is not bool:
        raise ValueError("confirmation_enabled must be boolean")
    return result


class StagePool:
    def __init__(self, service, resources, options):
        self.service, self.resources = service, resources
        self.options = validate_options(options)
        self.limit = self.options.get("max_concurrent_trials", 1)
        self.devices = self.options.get("gpu_pool") or []

    def run(self, executor, jobs):
        """Interleave Trial coroutines, executing at most one stage per Trial."""
        if self.devices and executor.runner == "speechbrain":
            import torch
            if any(device >= torch.cuda.device_count() for device in self.devices):
                raise ValueError("gpu_pool references unavailable visible GPU indices")
        root = Path(executor.experiments_dir) / executor.experiment_id / "stage_results"
        root.mkdir(parents=True, exist_ok=True)
        pending, active, results = {}, {}, {}
        generators = {trial.trial_id: executor.steps(trial, attempt) for trial, attempt in jobs}
        def advance(trial_id, response=None):
            try:
                pending[trial_id] = generators[trial_id].send(response)
            except StopIteration as done:
                results[trial_id] = done.value
            except ResourceBudgetExhausted:
                raise
            except Exception as exc:
                results[trial_id] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        for trial, _ in jobs:
            advance(trial.trial_id)
        try:
            while pending or active:
                if self.resources:
                    self.resources.check()
                for trial_id, (request, _, attempt) in list(pending.items()):
                    if len(active) >= self.limit:
                        break
                    payload = json.loads(request.payload_json)
                    count = self.options.get("gpus_per_trial", 1) if request.stage == "training" else 1
                    used = {d for item in active.values() for d in item["devices"]}
                    free = [d for d in self.devices if d not in used]
                    if self.devices and len(free) < count:
                        continue
                    assigned = free[:count] if self.devices else []
                    stage_id = sha256(json.dumps([trial_id, attempt, asdict(request)], sort_keys=True).encode()).hexdigest()
                    path = root / f"{stage_id}.json"
                    if not path.exists() and self.resources and any(r.get("stage_id") == stage_id for r in self.resources.data["stages"]):
                        _write_json_atomic(path, {"raw": {"status": "failed", "error": "RuntimeError: worker interrupted before durable result"}, "seconds": 0})
                    if path.exists():
                        del pending[trial_id]
                        advance(trial_id, json.loads(path.read_text(encoding="utf-8"))["raw"])
                        continue
                    gpu_count = len(assigned)
                    if not self.devices:
                        from agent.hpo.resources import stage_gpu_count
                        gpu_count = stage_gpu_count(self.options, executor.runner, evaluation=request.stage != "training")
                    overrides = payload["overrides"]
                    deadline = None
                    if self.resources:
                        deadline = self.resources.start_stage(experiment_id=executor.experiment_id,
                            trial_id=trial_id, attempt=attempt, stage=request.stage, gpu_count=gpu_count,
                            stage_id=stage_id, available_slots=max(1, min(self.limit-len(active), len(pending))),
                            max_seconds=overrides.get("_hpo_max_duration_seconds"))
                    if deadline:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            raise ResourceBudgetExhausted("stage reservation expired before dispatch")
                        overrides["_hpo_max_duration_seconds"] = min(remaining, overrides.get("_hpo_max_duration_seconds", remaining))
                    if assigned:
                        run = overrides.setdefault("_run_opts", {})
                        run["device"] = "cuda:0"
                        if request.stage == "training" and count > 1:
                            run.update(ddp_devices=list(range(count)), batch_size_semantics=self.options.get("batch_size_semantics", "global"))
                        else:
                            run.pop("ddp_devices", None)
                    request = replace(request, payload_json=json.dumps(payload))
                    ctx = mp.get_context("spawn")
                    parent_gate, child_gate = ctx.Pipe()
                    process = ctx.Process(target=_worker, args=(request, str(path), child_gate,
                        identity(os.getpid()), stage_id, assigned))
                    process.start(); child_gate.close()
                    owner = {**identity(process.pid), "token": stage_id, "children_path": str(path) + ".children.json", "result_path": str(path)}
                    active[trial_id] = dict(process=process, owner=owner, stage_id=stage_id,
                        path=path, devices=assigned, started=time.time(), deadline=deadline)
                    _write_json_atomic(root / f"{stage_id}.owner.json", owner)
                    if self.resources:
                        self.resources.attach_worker(stage_id, owner)
                    parent_gate.send("start"); parent_gate.close()
                    del pending[trial_id]
                for trial_id, item in list(active.items()):
                    timed_out = item["deadline"] is not None and time.time() >= item["deadline"]
                    if item["process"].is_alive() and not timed_out:
                        continue
                    terminate_owned(item["owner"])
                    item["process"].join(timeout=1)
                    if item["path"].exists():
                        outcome = json.loads(item["path"].read_text(encoding="utf-8"))
                    else:
                        outcome = {"raw": {"status": "failed", "error": "TimeoutError: stage deadline" if timed_out else "RuntimeError: worker exited without a result"},
                                   "seconds": time.time()-item["started"]}
                        _write_json_atomic(item["path"], outcome)
                    if self.resources:
                        self.resources.finish_stage(stage_id=item["stage_id"], seconds=time.time()-item["started"])
                    del active[trial_id]
                    from agent.hpo.policies import FailurePolicy
                    error = outcome["raw"].get("error")
                    if outcome["raw"].get("status") == "failed" and FailurePolicy().classify(error).scope == "study":
                        raise RuntimeError(f"system failure in managed stage: {error}")
                    advance(trial_id, outcome["raw"])
                time.sleep(.02)
        finally:
            for item in active.values():
                terminate_owned(item["owner"])
                item["process"].join(timeout=1)
                if self.resources:
                    self.resources.finish_stage(stage_id=item["stage_id"])
            for generator in generators.values():
                generator.close()
        return results
