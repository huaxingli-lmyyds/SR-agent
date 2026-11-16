"""Bounded checkpoint selection, independent of the SpeechBrain dependency."""
from pathlib import Path
import yaml
from agent.core.metrics import is_finite_metric


def candidates(save_folder):
    rows = []
    for path in sorted(Path(save_folder).glob("CKPT+*")):
        try:
            meta = yaml.safe_load((path / "CKPT.yaml").read_text(encoding="utf-8"))
            observation = meta.get("training_observation") or {}
            epoch = observation.get("checkpoint_epoch")
            if not is_finite_metric(meta.get("ErrorRate")) or not is_finite_metric(epoch):
                continue
            rows.append({"path": str(path.resolve()), "epoch": epoch,
                         "error_rate": meta["ErrorRate"], "observation": observation})
        except (OSError, ValueError, AttributeError, yaml.YAMLError):
            continue
    if not rows:
        return []
    best = sorted(rows, key=lambda r: (r["error_rate"], r["epoch"], r["path"]))[:2]
    latest = max(rows, key=lambda r: (r["epoch"], r["path"]))
    unique = {r["path"]: r for r in [*best, latest]}
    return sorted(unique.values(), key=lambda r: (r["epoch"], r["path"]))


def choose(rows, metric, mode):
    if not rows or any(r.get("status") != "success" or not is_finite_metric(r.get("metrics", {}).get(metric)) for r in rows):
        raise ValueError("checkpoint selection requires all candidate evaluations")
    sign = 1 if mode == "min" else -1
    return min(rows, key=lambda r: (sign*r["metrics"][metric], r["epoch"], r["path"]))
