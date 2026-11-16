"""Small, observed training metadata; never infer CLR phase from epoch counts."""

from math import isfinite
from pathlib import Path


def scheduler_snapshot(scheduler):
    result = {"scheduler_class": type(scheduler).__name__}
    for source, target in (
        ("base_lr", "base_lr"),
        ("max_lr", "max_lr"),
        ("step_size", "half_cycle_steps"),
        ("clr_iterations", "scheduler_iterations"),
        ("current_lr", "lr_after_last_batch"),
    ):
        value = getattr(scheduler, source, None)
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and isfinite(value)
        ):
            result[target] = value
    mode = getattr(scheduler, "mode", None)
    if isinstance(mode, str):
        result["mode"] = mode
    # Other schedulers can have a step_size with entirely different semantics.
    if type(scheduler).__name__ == "CyclicLRScheduler":
        steps, half = (
            result.get("scheduler_iterations"),
            result.get("half_cycle_steps"),
        )
        if steps is not None and steps >= 0 and half is not None and half > 0:
            result["cycle_phase"] = (steps % (2 * half)) / (2 * half)
            result["cycle_phase_definition"] = (
                "0=cycle start, 0.5=peak, approaches 1=cycle end"
            )
    return result


def selected_checkpoint_observation(checkpoint):
    import yaml

    path = Path(checkpoint) / "CKPT.yaml"
    if not path.is_file():
        return {}
    try:
        metadata = yaml.safe_load(path.read_text())
        observation = (
            metadata.get("training_observation")
            if isinstance(metadata, dict)
            else None
        )
        return observation if isinstance(observation, dict) else {}
    except (OSError, ValueError, yaml.YAMLError):
        return {}
