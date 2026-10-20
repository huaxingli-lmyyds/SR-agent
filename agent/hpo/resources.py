"""Campaign-wide, persistent resource budgets (allocated GPU time, not utilization)."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional


class ResourceBudgetExhausted(RuntimeError):
    """A hard Campaign resource ceiling prevents further work."""


def positive_limit(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


class CampaignResourceBudget:
    """Single-controller ledger. Wall deadline survives pauses and restarts.

    Stages are journaled before execution and settled in finally, including failed
    attempts. A crash leaves a reservation, charged conservatively on recovery.
    The OS lock prevents two resumed controllers spending the same allowance.
    """

    def __init__(self, path: Path, *, limits: dict, resume: bool = False,
                 clock: Optional[Callable[[], float]] = None) -> None:
        self.path = Path(path)
        self.clock = clock or time.time
        requested = {key: positive_limit(limits.get(key), key)
                     for key in ("max_gpu_hours", "max_wall_seconds")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.path.with_suffix(".lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self._lock.seek(0)
                if not self._lock.read(1):
                    self._lock.write(b"0")
                    self._lock.flush()
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock.close()
            raise RuntimeError("Campaign resource ledger is in use by another controller") from exc
        try:
            if resume:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                for key, value in requested.items():
                    if value is not None and value != self.data["limits"].get(key):
                        raise ValueError(f"cannot change persisted resource limit on resume: {key}")
                active = self.data.get("active_stage")
                if active:
                    # Do not silently refund an uncommitted GPU attempt.
                    elapsed = max(0.0, self.clock() - active["started_at"])
                    reserved = active.get("timeout_seconds")
                    seconds = max(elapsed, reserved or 0.0)
                    self.finish_stage(seconds=seconds, estimated=True)
            else:
                if self.path.exists():
                    raise ValueError(f"resource ledger already exists: {self.path}")
                now = self.clock()
                self.data = {"schema_version": 1, "limits": requested,
                             "started_at": now, "gpu_seconds": 0.0,
                             "stages": [], "active_stage": None,
                             "deadline_epoch": now + requested["max_wall_seconds"]
                             if requested["max_wall_seconds"] is not None else None}
            self._save()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self._lock.closed:
            if os.name == "nt":
                import msvcrt
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()

    def _save(self) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.path)

    def snapshot(self) -> dict:
        elapsed = max(0.0, self.clock() - self.data["started_at"])
        gpu = self.data["gpu_seconds"]
        active = self.data.get("active_stage")
        if active:
            gpu += max(0.0, self.clock() - active["started_at"]) * active["gpu_count"]
        ceiling = self.data["limits"]["max_gpu_hours"]
        deadline = self.data["deadline_epoch"]
        remaining_gpu = max(0.0, ceiling * 3600 - gpu) if ceiling is not None else None
        remaining_wall = max(0.0, deadline - self.clock()) if deadline is not None else None
        reason = ("max_wall_seconds_reached" if remaining_wall == 0 else
                  "max_gpu_hours_reached" if remaining_gpu == 0 else None)
        return {"ledger_path": str(self.path), **self.data["limits"],
                "wall_seconds": elapsed, "gpu_seconds": gpu, "gpu_hours": gpu / 3600,
                "wall_seconds_remaining": remaining_wall,
                "gpu_hours_remaining": remaining_gpu / 3600 if remaining_gpu is not None else None,
                "deadline_epoch": deadline, "stop_reason": reason,
                "cost_may_be_incomplete": any(s.get("estimated") for s in self.data["stages"])}

    def check(self) -> dict:
        snapshot = self.snapshot()
        if snapshot["stop_reason"]:
            raise ResourceBudgetExhausted(f"resource_budget_exhausted: {snapshot['stop_reason']}")
        return snapshot

    def start_stage(self, *, experiment_id: str, trial_id: str, attempt: int,
                    stage: str, gpu_count: int, max_seconds: Optional[float] = None) -> Optional[float]:
        snapshot = self.check()
        if self.data.get("active_stage"):
            raise RuntimeError("a resource-budget stage is already active")
        ceilings = [x for x in (positive_limit(max_seconds, "max_seconds"),
                               snapshot["wall_seconds_remaining"]) if x is not None]
        if gpu_count and snapshot["gpu_hours_remaining"] is not None:
            ceilings.append(snapshot["gpu_hours_remaining"] * 3600 / gpu_count)
        timeout = min(ceilings) if ceilings else None
        now = self.clock()
        self.data["active_stage"] = {"experiment_id": experiment_id, "trial_id": trial_id,
                                     "attempt": attempt, "stage": stage, "gpu_count": gpu_count,
                                     "started_at": now, "timeout_seconds": timeout}
        self._save()
        return now + timeout if timeout is not None else None

    def finish_stage(self, *, seconds: Optional[float] = None, estimated: bool = False) -> dict:
        active = self.data.get("active_stage")
        if active is None:
            raise RuntimeError("no resource-budget stage is active")
        duration = max(0.0, self.clock() - active["started_at"]) if seconds is None else max(0.0, seconds)
        record = {**active, "seconds": duration, "gpu_seconds": duration * active["gpu_count"], "estimated": estimated}
        self.data["gpu_seconds"] += record["gpu_seconds"]
        self.data["stages"].append(record)
        self.data["active_stage"] = None
        self._save()
        return record


def stage_gpu_count(runtime_options: dict, runner: str, *, evaluation: bool = False) -> int:
    """Training uses all DDP devices; verification uses one visible device."""
    device = str(runtime_options.get("device") or "").lower()
    if device.startswith("cpu"):
        return 0
    if not evaluation and runtime_options.get("ddp_devices"):
        devices = runtime_options["ddp_devices"]
        return len(devices.split(",")) if isinstance(devices, str) else len(devices)
    if device.startswith("cuda"):
        return 1
    if runner == "speechbrain":
        import torch
        return int(torch.cuda.is_available())
    return 0
