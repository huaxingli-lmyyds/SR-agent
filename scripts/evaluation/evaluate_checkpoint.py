#!/usr/bin/env python3
"""Evaluate one SpeechBrain checkpoint on an explicit verification set.

Example:
    python scripts/evaluation/evaluate_checkpoint.py \
        --checkpoint /path/to/CKPT+2026-07-20+11-26-17+00 \
        --data-folder /hy-tmp/voxceleb1 \
        --verification-file /hy-tmp/lists/veri_test2.txt \
        --device cuda \
        --batch-size 8

The verification file must use the VoxCeleb pair format:
``label enrol_audio_path test_audio_path``. Audio paths are resolved against
``--data-folder`` by the existing SpeechBrain preparation pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "verification_ecapa.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "checkpoint_evaluation"

EvaluationRunner = Callable[..., dict[str, Any]]


def _local_path(value: str, label: str, *, directory: bool = False) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    if directory and not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    if not directory and not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def _checkpoint_paths(value: str) -> tuple[Path, Path]:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"checkpoint does not exist: {path}")
    if path.is_dir():
        checkpoint_dir = path
        checkpoint_file = path / "embedding_model.ckpt"
    else:
        checkpoint_dir = path.parent
        checkpoint_file = path
        if checkpoint_file.name != "embedding_model.ckpt":
            raise ValueError(
                "a checkpoint file must be named embedding_model.ckpt; "
                "otherwise pass its CKPT+... directory"
            )
    if not checkpoint_file.is_file():
        raise ValueError(
            f"checkpoint directory has no embedding_model.ckpt: {checkpoint_dir}"
        )
    return checkpoint_dir, checkpoint_file


def _verification_source(value: str) -> str:
    if urlparse(value).scheme.lower() in {"http", "https"}:
        return value
    return str(_local_path(value, "verification file"))


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="CKPT+... directory or its embedding_model.ckpt file",
    )
    parser.add_argument(
        "--data-folder",
        required=True,
        help="Dataset root; audio paths in the verification file are relative to it",
    )
    parser.add_argument(
        "--verification-file",
        required=True,
        help="Local pair-list path or HTTP(S) URL",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="SpeechBrain verification YAML matching the checkpoint architecture",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Result directory; defaults to results/checkpoint_evaluation/<timestamp>",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or cuda:N"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--score-norm",
        choices=("none", "z-norm", "t-norm", "s-norm"),
        default="none",
    )
    parser.add_argument("--cohort-size", type=int, default=20000)
    parser.add_argument("--n-train-snts", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def evaluate_checkpoint(
    args: argparse.Namespace,
    *,
    runner: EvaluationRunner | None = None,
) -> dict[str, Any]:
    checkpoint_dir, checkpoint_file = _checkpoint_paths(args.checkpoint)
    data_folder = _local_path(args.data_folder, "data folder", directory=True)
    verification_file = _verification_source(args.verification_file)
    config_path = _local_path(args.config, "verification config")

    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.num_workers < 0:
        raise ValueError("num workers must be non-negative")
    if args.score_norm != "none" and args.cohort_size <= 0:
        raise ValueError(
            "cohort size must be positive when score normalization is enabled"
        )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides: dict[str, Any] = {
        "verification_file": verification_file,
        "output_folder": str(output_dir),
        "save_folder": str(output_dir / "save"),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "score_norm": args.score_norm,
        "seed": args.seed,
        "_run_opts": {"device": args.device},
    }
    if args.score_norm != "none":
        overrides.update(
            {
                "cohort_size": args.cohort_size,
                "n_train_snts": args.n_train_snts,
            }
        )

    if runner is None:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from agent.runners.speechbrain_backend import run_evaluation

        runner = run_evaluation

    started_at = datetime.now()
    raw = runner(
        config_path=str(config_path),
        model_path=str(checkpoint_dir),
        data_folder=str(data_folder),
        overrides=overrides,
    )
    completed_at = datetime.now()
    report = {
        "schema_version": "1.0",
        "status": raw.get("status", "failed"),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "file": str(checkpoint_file),
        },
        "dataset": {
            "data_folder": str(data_folder),
            "verification_file": verification_file,
        },
        "configuration": {
            "verification_config": str(config_path),
            "device": args.device,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "score_norm": args.score_norm,
            "seed": args.seed,
        },
        "metrics": {
            "eer_percent": raw.get("eer"),
            "min_dcf": raw.get("min_dcf"),
        },
        "artifacts": {
            "output_folder": raw.get("output_folder") or str(output_dir),
            "scores_path": raw.get("scores_path"),
        },
        "error": raw.get("error"),
    }
    report_path = output_dir / "evaluation_result.json"
    report["artifacts"]["result_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_checkpoint(args)
    except Exception as exc:
        report = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
