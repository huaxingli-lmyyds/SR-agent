"""Read-only ledger projections shared by feedback, decisions and analysis tools."""
import json
from pathlib import Path
import time


FIELDS = ("training_seconds", "evaluation_seconds", "training_gpu_seconds", "evaluation_gpu_seconds")


def attach_ledger_costs(tracker, experiment_id, trials):
    record = tracker.get_experiment(experiment_id) or {}
    optimization = record.get("extensions", {}).get("optimization", {})
    resources = optimization.get("study", {}).get("resource_budget") or optimization.get("resource_budget") or optimization.get("campaign", {}).get("resource_budget") or {}
    path = resources.get("ledger_path")
    if not path:
        return trials
    try:
        ledger = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = list(ledger["stages"])
        active = list((ledger.get("active_stages") or {}).values())
        if not active and ledger.get("active_stage"):
            active = [ledger["active_stage"]]
        for row in active:
            seconds = max(0, time.time() - row["started_at"])
            rows.append({**row, "seconds": seconds, "gpu_seconds": seconds * row["gpu_count"], "estimated": True, "active": True})
        grouped = {}
        for row in rows:
            if row["experiment_id"] == experiment_id:
                grouped.setdefault(row["trial_id"], []).append(row)
        for trial in trials:
            selected = grouped.get(trial.trial_id, [])
            values = dict.fromkeys(FIELDS, 0.0)
            for row in selected:
                stage = "training" if row["stage"] == "training" else "evaluation"
                values[f"{stage}_seconds"] += row["seconds"]
                values[f"{stage}_gpu_seconds"] += row["gpu_seconds"]
            trial.cost["ledger_cost"] = {**values, "source": "campaign_ledger", "ledger_path": str(path),
                "stage_count": len(selected), "estimated_stage_count": sum(bool(r.get("estimated")) for r in selected),
                "active_stage_count": sum(bool(r.get("active")) for r in selected),
                "cost_may_be_incomplete": any(r.get("estimated") for r in selected) or (not selected and trial.status != "suggested")}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        for trial in trials:
            trial.cost["ledger_cost"] = {"source": "ledger_unavailable", "cost_may_be_incomplete": True,
                                         "error": f"{type(exc).__name__}: {exc}"}
    return trials


def observed_cost(trials):
    totals = dict.fromkeys(FIELDS, 0.0)
    sources, incomplete = set(), False
    count = estimated = active = 0
    for trial in trials:
        ledger = trial.cost.get("ledger_cost")
        if ledger is not None:
            sources.add(ledger["source"])
            incomplete |= ledger["cost_may_be_incomplete"]
            count += ledger.get("stage_count", 0)
            estimated += ledger.get("estimated_stage_count", 0)
            active += ledger.get("active_stage_count", 0)
            for key in FIELDS:
                totals[key] += ledger.get(key, 0.0)
        else:
            sources.add("legacy_trial_fields")
            # No attempt journal means completeness cannot be established.
            incomplete = True
            for key in FIELDS:
                value = trial.cost.get(f"attempt_{key}")
                if value is None and key.endswith("_seconds") and "gpu" not in key:
                    value = (trial.cost.get(key[:-8]) or {}).get("duration_seconds")
                from agent.core.metrics import is_finite_metric
                if is_finite_metric(value) and value >= 0:
                    totals[key] += value
    return {**totals, "gpu_hours": (totals["training_gpu_seconds"] + totals["evaluation_gpu_seconds"]) / 3600,
            "cost_may_be_incomplete": incomplete, "sources": sorted(sources), "stage_count": count,
            "estimated_stage_count": estimated, "active_stage_count": active,
            "scope": "selected Trials, all ledger stages; stage seconds are not parallel wall time"}
