"""Runtime options and deadlines shared by training and evaluation tools."""

from time import time
from typing import Any

from agent.hpo.resources import ResourceBudgetExhausted, positive_limit


def build_run_options(
    *,
    device=None,
    precision=None,
    eval_precision=None,
    ddp_devices=None,
    distributed_backend=None,
    batch_size_semantics=None,
) -> dict[str, Any]:
    options = {
        key: value
        for key, value in (
            ("device", device),
            ("precision", precision),
            ("eval_precision", eval_precision),
        )
        if value
    }
    if ddp_devices:
        options.update(
            ddp_devices=ddp_devices,
            distributed_backend=distributed_backend or "nccl",
            batch_size_semantics=batch_size_semantics or "global",
        )
    return options


def apply_runtime_deadline(
    overrides: dict, deadline: float | None, stage: str
) -> None:
    """Include preparation time and preserve the tighter Trial timeout."""
    if deadline is None:
        return
    positive_limit(deadline, "runtime_deadline_epoch")
    remaining = deadline - time()
    if remaining <= 0:
        raise ResourceBudgetExhausted(
            f"resource budget expired during {stage} preparation"
        )
    limit = overrides.get("_hpo_max_duration_seconds")
    overrides["_hpo_max_duration_seconds"] = min(
        remaining, float(limit) if limit is not None else remaining
    )
