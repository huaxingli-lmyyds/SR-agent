"""Promotion eligibility; actual prepared data are checked again by the runner."""
from pathlib import Path
import json
from .protocol import file_sha256


def prepare_continuation(service, study, trial, options):
    result = {"capture": True, "enabled": False, "reason": "restart_policy"}
    if options.get("promotion_training") != "resume_if_compatible":
        return result
    if study.history_context.get("runner") != "speechbrain":
        return {**result, "capture": False, "reason": "runner_unsupported"}
    if not trial.parent_trial_id:
        return {**result, "reason": "initial_trial"}
    parent = service.load_trial(study.experiment_id, trial.parent_trial_id)
    if parent.parameters != trial.parameters or (parent.budget.data_fraction or 1.0) != (trial.budget.data_fraction or 1.0):
        return {**result, "reason": "parameters_or_data_fraction_changed"}
    if not parent.budget.epochs or not trial.budget.epochs or trial.budget.epochs <= parent.budget.epochs:
        return {**result, "reason": "not_an_epoch_extension"}
    raw = parent.cost.get("managed_training_result", {})
    observation = (raw.get("extensions") or {}).get("speechbrain") or {}
    manifest_path = (observation.get("runtime") or {}).get("resume_manifest")
    if not manifest_path or not Path(manifest_path).is_file():
        return {**result, "reason": "complete_state_missing"}
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest["epoch"] != parent.budget.epochs:
            return {**result, "reason": "parent_did_not_complete_budget"}
        for name, digest in manifest["files"].items():
            if file_sha256(Path(manifest_path).parent / name) != digest:
                return {**result, "reason": "state_hash_mismatch"}
    except (ValueError, KeyError, OSError):
        return {**result, "reason": "state_manifest_invalid"}
    return {**result, "enabled": True, "reason": "eligible_pending_data_check",
            "manifest": manifest_path, "parent_trial_id": parent.trial_id,
            "parent_candidates": parent.cost.get("checkpoint_evaluations", []),
            "inherited_training_seconds": parent.cost.get("training", {}).get("duration_seconds")}
