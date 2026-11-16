"""Space-efficient materialization for derived dataset versions."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict

MATERIALIZATION_MODES = {"hardlink", "copy"}


def validate_materialization_mode(mode: str) -> str:
    normalized = str(mode or "hardlink").strip().lower()
    if normalized not in MATERIALIZATION_MODES:
        raise ValueError(
            "materialization_mode must be one of: copy, hardlink"
        )
    return normalized


def materialize_file(source: Path, destination: Path, mode: str) -> str:
    """Link a file when possible and fall back to a physical copy."""

    selected_mode = validate_materialization_mode(mode)
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if source.samefile(destination):
                return "hardlink"
        except OSError:
            pass
        destination.unlink()
    if selected_mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def materialize_tree(
    source: Path, destination: Path, mode: str
) -> Dict[str, int]:
    """Materialize a directory while preserving its relative structure."""

    selected_mode = validate_materialization_mode(mode)
    counts = {"hardlink": 0, "copy": 0}
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            method = materialize_file(path, target, selected_mode)
            counts[method] += 1
    return counts


__all__ = [
    "MATERIALIZATION_MODES",
    "materialize_file",
    "materialize_tree",
    "validate_materialization_mode",
]