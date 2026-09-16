"""Optional one-Trial SpeechBrain DDP launcher.

The HPO process remains the sole owner of Study/Trial state.  Only the child
torchrun workers join the distributed process group.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


_CUSTOM_RUN_OPT_KEYS = {
    "ddp_devices",
    "distributed_world_size",
    "distributed_backend",
    "batch_size_semantics",
}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_ddp_devices(value: Any) -> Optional[list[int] | str]:
    """Normalize an optional visible-device selection."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        stripped = value.strip().lower()
        if not stripped:
            return None
        if stripped == "auto":
            return "auto"
        parts = [item.strip() for item in stripped.split(",")]
        if not all(parts):
            raise ValueError("ddp_devices must be 'auto' or comma-separated GPU indices")
        try:
            devices = [int(item) for item in parts]
        except ValueError as exc:
            raise ValueError(
                "ddp_devices must be 'auto' or comma-separated GPU indices"
            ) from exc
    elif isinstance(value, (list, tuple)):
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("ddp_devices entries must be integer GPU indices")
        devices = list(value)
    else:
        raise ValueError("ddp_devices must be 'auto' or a list of GPU indices")
    if len(devices) < 2:
        raise ValueError("explicit DDP requires at least two GPU indices")
    if any(item < 0 for item in devices) or len(set(devices)) != len(devices):
        raise ValueError("ddp_devices must contain unique non-negative indices")
    return devices


def resolve_ddp_plan(torch_module: Any, run_opts: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve DDP once so auto device selection is frozen before a Study."""
    requested = parse_ddp_devices(run_opts.get("ddp_devices"))
    backend = str(run_opts.get("distributed_backend") or "nccl").strip().lower()
    semantics = str(run_opts.get("batch_size_semantics") or "global").strip().lower()
    if backend not in {"nccl", "gloo"}:
        raise ValueError("distributed_backend must be 'nccl' or 'gloo'")
    if semantics not in {"global", "per_device"}:
        raise ValueError("batch_size_semantics must be 'global' or 'per_device'")
    if requested is None:
        return {
            "enabled": False,
            "requested": None,
            "devices": [],
            "world_size": 1,
            "backend": None,
            "batch_size_semantics": "per_device",
        }
    cuda_available = bool(torch_module.cuda.is_available())
    device_count = int(torch_module.cuda.device_count()) if cuda_available else 0
    if requested == "auto":
        devices = list(range(device_count))
        if len(devices) < 2:
            return {
                "enabled": False,
                "requested": "auto",
                "devices": [],
                "world_size": 1,
                "backend": None,
                "batch_size_semantics": "per_device",
                "fallback_reason": "fewer_than_two_visible_cuda_devices",
            }
    else:
        devices = list(requested)
        if not cuda_available:
            raise RuntimeError("explicit DDP devices requested but CUDA is unavailable")
        invalid = [item for item in devices if item >= device_count]
        if invalid:
            raise ValueError(
                "ddp_devices reference unavailable visible GPU indices: "
                + ", ".join(str(item) for item in invalid)
            )
    if backend == "nccl" and os.name == "nt":
        raise RuntimeError("NCCL DDP is not supported on native Windows; use Linux")
    distributed = getattr(torch_module, "distributed", None)
    if backend == "nccl" and distributed is not None:
        available = getattr(distributed, "is_nccl_available", None)
        if callable(available) and not available():
            raise RuntimeError("the installed PyTorch build does not provide NCCL")
    requested_world_size = run_opts.get("distributed_world_size")
    if requested_world_size is not None and int(requested_world_size) != len(devices):
        raise ValueError("distributed_world_size must equal the number of DDP devices")
    device = str(run_opts.get("device") or "auto").strip().lower()
    if device not in {"", "auto", "cuda"}:
        try:
            selected_device = int(device.split(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "DDP device must be auto, cuda, or the first selected cuda:N"
            ) from exc
        if not device.startswith("cuda:") or selected_device != devices[0]:
            raise ValueError(
                "DDP device must match the first entry in ddp_devices"
            )
    return {
        "enabled": True,
        "requested": requested,
        "devices": devices,
        "world_size": len(devices),
        "backend": backend,
        "batch_size_semantics": semantics,
    }


def freeze_ddp_runtime_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an explicitly requested DDP configuration for audit/signatures."""
    value = dict(options or {})
    if not value.get("ddp_devices"):
        return value
    import torch

    plan = resolve_ddp_plan(torch, value)
    value["ddp_plan"] = plan
    if plan["enabled"]:
        value.update({
            "device": f"cuda:{plan['devices'][0]}",
            "ddp_devices": list(plan["devices"]),
            "distributed_world_size": plan["world_size"],
            "distributed_backend": plan["backend"],
            "batch_size_semantics": plan["batch_size_semantics"],
        })
    else:
        value.pop("ddp_devices", None)
        value.pop("distributed_world_size", None)
        value.pop("distributed_backend", None)
        value.pop("batch_size_semantics", None)
    return value


def training_runtime_signature(options: Dict[str, Any]) -> Dict[str, Any]:
    """Return training-semantics fields that affect cross-Study comparison."""
    options = dict(options or {})
    plan = dict(options.get("ddp_plan") or {})
    # ``distributed_world_size`` is retained in persisted execution records.
    # Fall back to it so records written by direct callers (without the frozen
    # plan helper) cannot be mislabelled as single-device training.
    world_size = int(
        plan.get("world_size")
        or options.get("distributed_world_size")
        or 1
    )
    enabled = bool(plan.get("enabled", world_size > 1)) and world_size > 1
    return {
        "distributed": enabled,
        "world_size": world_size if enabled else 1,
        "backend": (
            plan.get("backend") or options.get("distributed_backend")
        ) if enabled else None,
        "batch_size_semantics": (
            plan.get("batch_size_semantics")
            or options.get("batch_size_semantics")
            or "global"
        ) if enabled else "per_device",
        "precision": options.get("precision") or "fp32",
        "eval_precision": options.get("eval_precision") or "fp32",
    }


def run_distributed_training(
    config_path: str,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Launch exactly one Trial under torchrun and return rank-zero output."""
    import torch

    run_opts = dict(overrides.get("_run_opts") or {})
    plan = resolve_ddp_plan(torch, run_opts)
    if not plan["enabled"]:
        from .speechbrain import _run_speechbrain_training_with_timeout

        normalized = dict(overrides)
        normalized_opts = dict(run_opts)
        for key in _CUSTOM_RUN_OPT_KEYS:
            normalized_opts.pop(key, None)
        normalized["_run_opts"] = normalized_opts
        result = _run_speechbrain_training_with_timeout(config_path, normalized)
        result.setdefault("runtime", {})["ddp"] = plan
        return result

    output = Path(str(overrides.get("output_folder") or "")).resolve()
    if not str(overrides.get("output_folder") or "").strip():
        raise ValueError("DDP training requires an explicit output_folder")
    control_dir = output / ".ddp"
    control_dir.mkdir(parents=True, exist_ok=True)
    launch_id = uuid4().hex
    request_path = control_dir / f"request_{launch_id}.json"
    result_path = control_dir / f"result_{launch_id}.json"
    log_path = control_dir / f"launcher_{launch_id}.log"
    request = {
        "config_path": str(config_path),
        "overrides": dict(overrides),
        "plan": plan,
        "result_path": str(result_path),
    }
    _write_json_atomic(request_path, request)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={plan['world_size']}",
        "--max_restarts=0",
        "--module",
        "agent.runners.speechbrain_ddp_worker",
        "--request-file",
        str(request_path),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = _selected_visible_devices(plan["devices"])
    creationflags = (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    )
    timeout = overrides.get("_hpo_max_duration_seconds")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not _is_finite_positive(float(timeout))
    ):
        raise ValueError("_hpo_max_duration_seconds must be a finite positive number")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(_PROJECT_ROOT),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=creationflags,
        )
        try:
            process.wait(timeout=float(timeout) if timeout is not None else None)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            return {
                "status": "failed",
                "valid_error_rate": None,
                "error": f"TimeoutError: DDP training exceeded {timeout} seconds",
                "timeout_seconds": float(timeout),
                "terminated_by_budget": True,
                "process_exitcode": process.returncode,
                "runtime": {"ddp": plan, "launcher_log": str(log_path)},
            }
        except BaseException:
            _terminate_process_group(process)
            raise
    result = _read_result(result_path)
    if process.returncode != 0:
        error = (
            result.get("error")
            if result else f"torchrun exited with code {process.returncode}"
        )
        result = {
            **result,
            "status": "failed",
            "valid_error_rate": None,
            "error": error,
        }
    elif not result:
        result = {
            "status": "failed",
            "valid_error_rate": None,
            "error": "rank 0 did not write the DDP result",
        }
    result.setdefault("runtime", {}).update({
        "ddp": plan,
        "launcher_log": str(log_path),
    })
    result["process_exitcode"] = process.returncode
    return result


def _selected_visible_devices(devices: list[int]) -> str:
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not existing:
        return ",".join(str(item) for item in devices)
    visible = [item.strip() for item in existing.split(",") if item.strip()]
    if any(item >= len(visible) for item in devices):
        raise ValueError("ddp_devices exceed CUDA_VISIBLE_DEVICES")
    return ",".join(visible[item] for item in devices)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # The result is already a deterministic timeout failure. Avoid
            # letting a pathological child cleanup mask that result.
            pass


def _write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _read_result(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _is_finite_positive(value: float) -> bool:
    from math import isfinite

    return isfinite(value) and value > 0


__all__ = [
    "freeze_ddp_runtime_options",
    "parse_ddp_devices",
    "resolve_ddp_plan",
    "run_distributed_training",
    "training_runtime_signature",
]
