"""SpeechBrain runner implementation and SpeechBrain-specific result discovery."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
from pathlib import Path
from queue import Empty
import re
from typing import Any, Dict, List, Optional, Tuple

from agent.core.contracts import Artifact, OperationResult


@dataclass
class SpeechBrainRunnerAdapter:
    runner: str = "speechbrain"
    default_evaluation_config: Optional[str] = None
    supported_implementations = {"speechbrain"}
    supported_model_families = {"*"}

    def run_training(self, config_path: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        from .speechbrain_backend import run_training

        timeout = overrides.get("_hpo_max_duration_seconds")
        if timeout is None:
            return run_training(config_path, overrides)
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        process = context.Process(
            target=_run_speechbrain_training_process,
            args=(queue, config_path, overrides),
        )
        process.start()
        process.join(float(timeout))
        if process.is_alive():
            process.terminate()
            process.join()
            return {
                "status": "failed",
                "valid_error_rate": None,
                "error": f"TimeoutError: training exceeded {timeout} seconds",
            }
        try:
            return queue.get(timeout=1.0)
        except Empty:
            return {
                "status": "failed",
                "valid_error_rate": None,
                "error": f"training process exited with code {process.exitcode}",
            }

    def run_evaluation(
        self,
        config_path: str,
        model_path: Optional[str],
        data_path: Optional[str],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        from .speechbrain_backend import run_evaluation

        return run_evaluation(
            config_path=config_path,
            model_path=model_path,
            data_folder=data_path,
            overrides=overrides,
        )

    def collect_training_result(
        self,
        raw: Dict[str, Any],
        output_folder: Optional[Path],
        experiment_dir: Path,
    ) -> Dict[str, Any]:
        result = dict(raw)
        resolved_output = _find_output_folder(experiment_dir, output_folder)
        train_log = resolved_output / "train_log.txt" if resolved_output else None
        epoch_data, final_metrics = _parse_training_log(train_log) if train_log else ([], {})
        metrics = {"valid_error_rate": result.get("valid_error_rate")}
        metrics.update(result.get("metrics") or {})
        metrics.update(final_metrics)
        result.update({
            "output_folder": str(resolved_output) if resolved_output else None,
            "metrics": metrics,
            "model_paths": _find_model_paths(resolved_output, experiment_dir),
            "train_log_path": str(train_log) if train_log else None,
            "epoch_data": epoch_data,
            "final_metrics": final_metrics,
        })
        return result

    def normalize_training_result(self, raw: Dict[str, Any]) -> OperationResult:
        artifacts = [
            Artifact("checkpoint", Path(path).name, str(path))
            for path in raw.get("model_paths") or []
        ]
        if raw.get("train_log_path"):
            artifacts.append(Artifact("log", "training_log", str(raw["train_log_path"])))
        return OperationResult(
            status=raw.get("status", "failed"),
            stage="training",
            metrics={"validation": raw.get("metrics") or {}},
            artifacts=artifacts,
            extensions={"speechbrain": {
                "output_folder": raw.get("output_folder"),
                "epoch_data": raw.get("epoch_data") or [],
                "final_metrics": raw.get("final_metrics") or {},
            }},
            error=raw.get("error"),
        )

    def normalize_evaluation_result(self, raw: Dict[str, Any]) -> OperationResult:
        artifacts = []
        for artifact_type, key, name in (
            ("log", "evaluation_log_path", "evaluation_log"),
            ("predictions", "scores_path", "scores"),
        ):
            if raw.get(key):
                artifacts.append(Artifact(artifact_type, name, str(raw[key])))
        return OperationResult(
            status=raw.get("status", "failed"),
            stage="evaluation",
            metrics={"test": raw.get("metrics") or {}},
            artifacts=artifacts,
            extensions={"speechbrain": {"output_folder": raw.get("output_folder")}},
            error=raw.get("error"),
        )


def _run_speechbrain_training_process(queue: Any, config_path: str, overrides: Dict[str, Any]) -> None:
    from .speechbrain_backend import run_training

    queue.put(run_training(config_path, overrides))


def _find_output_folder(experiment_dir: Path, requested: Optional[Path]) -> Optional[Path]:
    candidates = [item for item in (requested, experiment_dir / "output", experiment_dir / "results") if item]
    return next(
        (item for item in candidates if (item / "train_log.txt").exists() or (item / "save").exists()),
        requested,
    )


def _find_model_paths(output_folder: Optional[Path], experiment_dir: Path) -> List[str]:
    candidates: List[Path] = []
    checkpoint_scores: Dict[Path, float] = {}
    checkpoint_dirs: List[Path] = []
    save_dirs = [output_folder / "save"] if output_folder else []
    save_dirs.extend([experiment_dir / "output" / "save", experiment_dir / "results" / "save"])
    for save_dir in save_dirs:
        if not save_dir.exists():
            continue
        candidates.extend(save_dir.glob("*.ckpt"))
        candidates.extend(save_dir.glob("*.pt"))
        for checkpoint_dir in save_dir.glob("CKPT+*"):
            checkpoint_dirs.append(checkpoint_dir)
            metadata = checkpoint_dir / "CKPT.yaml"
            if not metadata.exists():
                continue
            try:
                for line in metadata.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("ErrorRate:"):
                        checkpoint_scores[checkpoint_dir] = float(line.split(":", 1)[1].strip())
                        break
            except (OSError, ValueError):
                continue
    for directory in (output_folder, experiment_dir):
        if directory and directory.exists():
            candidates.extend(directory.glob("*.ckpt"))
            candidates.extend(directory.glob("*.pt"))
    if checkpoint_scores:
        return [str(min(checkpoint_scores, key=checkpoint_scores.get))]
    if checkpoint_dirs:
        return [str(max(checkpoint_dirs, key=lambda item: item.stat().st_mtime))]
    unique = {item.resolve(): item for item in candidates}
    return [str(max(unique.values(), key=lambda item: item.stat().st_mtime))] if unique else []


def _parse_training_log(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not path.exists():
        return [], {}
    pattern = re.compile(
        r"epoch:\s*(\d+),\s*lr:\s*([\d.e+-]+)\s*-\s*"
        r"train loss:\s*([\d.e+-]+)\s*-\s*"
        r"valid loss:\s*([\d.e+-]+),\s*valid ErrorRate:\s*([\d.e+-]+)"
    )
    epochs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            epochs.append({
                "epoch": int(match.group(1)),
                "lr": float(match.group(2)),
                "train_loss": float(match.group(3)),
                "valid_loss": float(match.group(4)),
                "valid_error_rate": float(match.group(5)),
            })
    if not epochs:
        return [], {}
    final = epochs[-1]
    best = min(epochs, key=lambda item: item["valid_error_rate"])
    return epochs, {
        "final_epoch": final["epoch"],
        "final_lr": final["lr"],
        "final_train_loss": final["train_loss"],
        "final_valid_loss": final["valid_loss"],
        "final_valid_error_rate": final["valid_error_rate"],
        "total_epochs": len(epochs),
        "best_epoch": best["epoch"],
        "best_valid_loss": best["valid_loss"],
        "best_error_rate": best["valid_error_rate"],
    }


__all__ = ["SpeechBrainRunnerAdapter"]
