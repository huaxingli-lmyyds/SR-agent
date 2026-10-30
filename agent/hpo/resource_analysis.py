"""Bounded hardware observations and planned search cost estimates."""

from __future__ import annotations

import os
from typing import Any

from .contracts import SearchSpace, TrialBudget
from .policies import HPOPlanningPolicy


def resource_snapshot(
    runtime_options: Any = None,
    declared_profile: Any = None,
) -> dict[str, Any]:
    """Collect bounded planning evidence without making resources mandatory."""
    runtime = (
        dict(runtime_options or {}) if isinstance(runtime_options, dict) else {}
    )
    snapshot: dict[str, Any] = {
        "requested_runtime": {
            key: runtime.get(key)
            for key in (
                "device",
                "ddp_devices",
                "distributed_world_size",
                "distributed_backend",
                "batch_size_semantics",
                "precision",
                "eval_precision",
            )
            if runtime.get(key) is not None
        },
        "cpu_count": os.cpu_count(),
    }
    requested_devices = runtime.get("ddp_devices")
    if isinstance(requested_devices, list | tuple):
        snapshot["requested_runtime"]["training_gpu_count"] = len(
            requested_devices
        )
    if isinstance(declared_profile, dict) and declared_profile:
        snapshot["declared_limits"] = dict(declared_profile)
    try:
        import psutil

        memory = psutil.virtual_memory()
        snapshot["system_memory_gb"] = {
            "total": round(float(memory.total) / (1024**3), 2),
            "available": round(float(memory.available) / (1024**3), 2),
        }
    except Exception:
        snapshot["system_memory_gb"] = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        snapshot["cuda"] = {
            "available": cuda_available,
            "device_count": int(torch.cuda.device_count())
            if cuda_available
            else 0,
            "devices": [],
        }
        if cuda_available:
            for index in range(int(torch.cuda.device_count())):
                properties = torch.cuda.get_device_properties(index)
                device = {
                    "index": index,
                    "name": str(properties.name),
                    "total_memory_gb": round(
                        float(properties.total_memory) / (1024**3),
                        2,
                    ),
                }
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                    device["free_memory_gb"] = round(
                        float(free_bytes) / (1024**3),
                        2,
                    )
                    device["observed_total_memory_gb"] = round(
                        float(total_bytes) / (1024**3),
                        2,
                    )
                except Exception:
                    pass
                snapshot["cuda"]["devices"].append(device)
    except Exception as exc:
        snapshot["cuda"] = {
            "available": None,
            "probe_error": type(exc).__name__,
        }
    return snapshot


def search_budget_analysis(
    search_space: SearchSpace,
    budgets: list[TrialBudget],
    hard_max_training_runs: int,
    initial_trial_count: int,
    promotion_limits: list[int],
    reduction_factor: int,
    campaign: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts = [
        int(initial_trial_count),
        *[int(item) for item in promotion_limits],
    ]
    counts = counts[: len(budgets)]
    counts.extend([0] * max(len(budgets) - len(counts), 0))
    rung_plan = []
    estimated_work_units = 0.0
    work_is_complete = True
    for budget, count in zip(budgets, counts, strict=False):
        unit_work = None
        if budget.epochs is not None and budget.data_fraction is not None:
            unit_work = float(budget.epochs) * float(budget.data_fraction)
            estimated_work_units += count * unit_work
        else:
            work_is_complete = False
        rung_plan.append(
            {
                "stage": budget.stage,
                "planned_runs": count,
                "epochs": budget.epochs,
                "data_fraction": budget.data_fraction,
                "max_duration_seconds": budget.max_duration_seconds,
                "relative_work_per_run": unit_work,
            }
        )
    campaign = campaign or {}
    summaries = list(campaign.get("study_summaries") or [])
    used_runs = sum(int(item.get("training_runs") or 0) for item in summaries)
    total_limit = campaign.get("max_total_training_runs")
    remaining_runs = (
        max(int(total_limit) - used_runs, 0)
        if total_limit is not None
        else None
    )
    resource_sensitive = {
        "batch_size",
        "sentence_len",
        "sample_rate",
        "embedding_dim",
        "channels",
    }
    return {
        "search_dimension_count": len(search_space.parameters),
        "search_parameter_names": [
            parameter.name for parameter in search_space.parameters
        ],
        "resource_sensitive_parameters": [
            parameter.name
            for parameter in search_space.parameters
            if parameter.name in resource_sensitive
        ],
        "finite_grid_cardinality": HPOPlanningPolicy.grid_cardinality(
            search_space
        ),
        "hard_max_training_runs": int(hard_max_training_runs),
        "planned_training_runs": sum(counts),
        "unused_training_run_capacity": max(
            int(hard_max_training_runs) - sum(counts),
            0,
        ),
        "reduction_factor": int(reduction_factor),
        "rung_plan": rung_plan,
        "estimated_relative_work_units": (
            round(estimated_work_units, 4) if work_is_complete else None
        ),
        "campaign_used_training_runs": used_runs,
        "campaign_remaining_training_runs": remaining_runs,
        "recent_resource_summaries": [
            item.get("resource_summary")
            for item in summaries[-3:]
            if item.get("resource_summary")
        ],
    }
