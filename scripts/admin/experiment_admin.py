#!/usr/bin/env python3
"""Archive or clear experiment records safely on Linux."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import List, Optional, Set, Tuple
import zipfile

DEFAULT_TARGETS = ("agent/experiments", "agent/results", "agent/logs", "results")
CACHE_TARGETS = ("agent/prep_cache", "save")
MEMORY_TARGET = "agent/memory/episodes.jsonl"


def inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"refusing to access path outside workspace: {resolved}")
    return resolved


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def campaign_id(record: dict) -> Optional[str]:
    version = record.get("version") or {}
    if version.get("campaign_id"):
        return str(version["campaign_id"])
    campaign = (((record.get("extensions") or {}).get("optimization") or {}).get("campaign") or {})
    return str(campaign["campaign_id"]) if campaign.get("campaign_id") else None


def matches(record: dict, args: argparse.Namespace) -> bool:
    model = str((record.get("model") or {}).get("family") or "")
    return (
        (not args.model_family or model == args.model_family)
        and (not args.campaign_id or campaign_id(record) == args.campaign_id)
        and (not args.experiment_id or record.get("experiment_id") in args.experiment_id)
    )


def selected_experiments(root: Path, args: argparse.Namespace) -> List[Tuple[Path, dict]]:
    experiments_root = root / "agent" / "experiments"
    selected = []
    if experiments_root.exists():
        for record_path in experiments_root.rglob("experiment_record.json"):
            record = load_json(record_path)
            if isinstance(record, dict) and matches(record, args):
                selected.append((record_path.parent, record))
    return selected


def has_filter(args: argparse.Namespace) -> bool:
    return bool(args.model_family or args.campaign_id or args.experiment_id)


def relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def file_stats(paths: List[Path]) -> Tuple[int, int]:
    files = []
    for path in paths:
        files.extend([path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()])
    return len(files), sum(item.stat().st_size for item in files)


def archive_experiments(args: argparse.Namespace) -> int:
    root = Path(args.workspace_root).resolve()
    archive_dir = inside(root, root / args.archive_directory)
    archive_dir.mkdir(parents=True, exist_ok=True)
    selected = selected_experiments(root, args)

    if has_filter(args):
        paths = [path for path, _ in selected]
    else:
        targets = list(DEFAULT_TARGETS)
        if not args.exclude_memory:
            targets.append(MEMORY_TARGET)
        if args.include_caches:
            targets.extend(CACHE_TARGETS)
        paths = [inside(root, root / item) for item in targets if (root / item).exists()]
    if not paths:
        raise RuntimeError("no matching experiment outputs or records were found")

    suffix = args.model_family or args.campaign_id or "all"
    name = args.name or f"experiments-{suffix}-{datetime.now():%Y%m%d-%H%M%S}.zip"
    if not name.endswith(".zip"):
        name += ".zip"
    archive_path = inside(root, archive_dir / Path(name).name)
    if archive_path.exists():
        raise FileExistsError(f"archive already exists: {archive_path}")

    count, size = file_stats(paths)
    manifest = {
        "schema_version": "2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "model_family": args.model_family,
            "campaign_id": args.campaign_id,
            "experiment_ids": args.experiment_id or [],
        },
        "experiment_ids": [record.get("experiment_id") for _, record in selected],
        "targets": [relative(root, path) for path in paths],
        "file_count": count,
        "total_bytes": size,
    }
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for path in paths:
            files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
            for item in files:
                if archive_dir not in item.resolve().parents:
                    output.write(item, relative(root, item))
        output.writestr("archive_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    print(json.dumps({"archive_path": str(archive_path), **manifest}, ensure_ascii=False, indent=2))
    return 0


def rebuild_indexes(root: Path, removed_ids: Set[str]) -> None:
    experiments_root = root / "agent" / "experiments"
    for history_path in experiments_root.rglob("experiments_history.json"):
        history = load_json(history_path)
        if isinstance(history, list):
            remaining = [item for item in history if item.get("experiment_id") not in removed_ids]
            history_path.write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")
    catalogs = [item for item in experiments_root.rglob("_catalog") if item.is_dir()]
    for catalog in catalogs:
        for pointer in catalog.rglob("*.json"):
            value = load_json(pointer)
            if isinstance(value, dict) and value.get("experiment_id") in removed_ids:
                pointer.unlink()
        directories = sorted((item for item in catalog.rglob("*") if item.is_dir()), reverse=True)
        for directory in directories:
            if not any(directory.iterdir()):
                directory.rmdir()


def clear_experiments(args: argparse.Namespace) -> int:
    root = Path(args.workspace_root).resolve()
    selected = selected_experiments(root, args)
    if has_filter(args):
        paths = [path for path, _ in selected]
    else:
        targets = list(DEFAULT_TARGETS)
        if not args.keep_memory:
            targets.append(MEMORY_TARGET)
        if args.include_caches:
            targets.extend(CACHE_TARGETS)
        paths = [inside(root, root / item) for item in targets if (root / item).exists()]
    if not paths:
        print("No matching experiment outputs or records were found.")
        return 0

    count, size = file_stats(paths)
    removed_ids = {str(record.get("experiment_id")) for _, record in selected}
    print(json.dumps({
        "execute": args.execute,
        "targets": [relative(root, path) for path in paths],
        "experiment_ids": sorted(removed_ids),
        "file_count": count,
        "total_bytes": size,
    }, ensure_ascii=False, indent=2))
    if not args.execute:
        print("Preview only. Run again with --execute to clear these targets.")
        return 0

    for path in paths:
        safe_path = inside(root, path)
        if safe_path == root:
            raise ValueError("refusing to clear workspace root")
        if safe_path.is_dir():
            if has_filter(args):
                shutil.rmtree(safe_path)
            else:
                for child in safe_path.iterdir():
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
        elif safe_path.exists():
            safe_path.unlink()
    if has_filter(args):
        rebuild_indexes(root, removed_ids)
    print("Experiment outputs and records were cleared.")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace-root", required=True)
    common.add_argument("--model-family")
    common.add_argument("--campaign-id")
    common.add_argument("--experiment-id", action="append")

    archive = commands.add_parser("archive", parents=[common])
    archive.add_argument("--archive-directory", default="experiment_archives")
    archive.add_argument("--name")
    archive.add_argument("--exclude-memory", action="store_true")
    archive.add_argument("--include-caches", action="store_true")
    archive.set_defaults(handler=archive_experiments)

    clear = commands.add_parser("clear", parents=[common])
    clear.add_argument("--execute", action="store_true")
    clear.add_argument("--keep-memory", action="store_true")
    clear.add_argument("--include-caches", action="store_true")
    clear.set_defaults(handler=clear_experiments)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return args.handler(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
