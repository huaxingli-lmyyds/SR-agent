"""Model-aware manifests and catalog pointers for experiment records."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict


def _slug(value: Any, fallback: str = "unknown") -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or fallback).strip()).strip("-.")
    return normalized.lower() or fallback


def _campaign(record: Dict[str, Any]) -> str:
    version = record.get("version") or {}
    if version.get("campaign_id"):
        return str(version["campaign_id"])
    optimization = (record.get("extensions") or {}).get("optimization") or {}
    campaign = optimization.get("campaign") or {}
    return str(campaign.get("campaign_id") or "standalone")


def build_version_manifest(record: Dict[str, Any], experiment_dir: Path, root: Path) -> Dict[str, Any]:
    task = record.get("task") or {}
    model = record.get("model") or {}
    execution = record.get("execution") or {}
    version = record.get("version") or {}
    scope = {
        "task_type": str(task.get("type") or "unknown"),
        "model_family": str(model.get("family") or "unknown"),
        "implementation": str(model.get("implementation") or "unknown"),
        "runner": str(execution.get("runner") or "unknown"),
        "dataset": str(task.get("dataset") or ""),
    }
    return {
        "schema_version": "1.0",
        "experiment_id": record.get("experiment_id"),
        "experiment_type": record.get("experiment_type"),
        "record_schema_version": record.get("schema_version"),
        "campaign_id": _campaign(record),
        "study_index": version.get("study_index"),
        "study_id": version.get("study_id") or ((record.get("extensions") or {}).get("optimization") or {}).get("study_id"),
        "scope": scope,
        "status": record.get("status"),
        "created_at": record.get("timestamp"),
        "updated_at": record.get("updated_at"),
        "paths": {
            "experiment": experiment_dir.relative_to(root).as_posix(),
            "record": (experiment_dir / "experiment_record.json").relative_to(root).as_posix(),
            "trials": (experiment_dir / "trials").relative_to(root).as_posix(),
            "evaluation": (experiment_dir / "evaluation").relative_to(root).as_posix(),
        },
    }


def sync_experiment_catalog(root: Path, experiment_dir: Path, record: Dict[str, Any]) -> None:
    """Write one local manifest and one small model-aware catalog pointer."""
    root = root.resolve()
    experiment_dir = experiment_dir.resolve()
    manifest = build_version_manifest(record, experiment_dir, root)
    manifest_path = experiment_dir / "version_manifest.json"
    _atomic_json(manifest_path, manifest)

    scope = manifest["scope"]
    pointer_dir = (
        root
        / "_catalog"
        / _slug(scope["task_type"])
        / _slug(scope["model_family"])
        / _slug(scope["implementation"])
        / _slug(manifest["campaign_id"], "standalone")
    )
    pointer = {
        "experiment_id": manifest["experiment_id"],
        "campaign_id": manifest["campaign_id"],
        "study_index": manifest["study_index"],
        "status": manifest["status"],
        "primary_metric": (record.get("task") or {}).get("primary_metric"),
        "metrics": record.get("metrics") or {},
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "updated_at": manifest["updated_at"],
    }
    pointer_path = pointer_dir / f"{_slug(manifest['experiment_id'])}.json"
    for stale in (root / "_catalog").rglob(pointer_path.name):
        if stale.resolve() != pointer_path.resolve():
            stale.unlink()
    _atomic_json(pointer_path, pointer)


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


__all__ = ["build_version_manifest", "sync_experiment_catalog"]
