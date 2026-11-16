from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "demo" / "config" / "ecapa_smoke.json"


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _command(config: dict, data_folder: Path) -> list[str]:
    main_py = ROOT / "main.py"
    command = [
        sys.executable,
        str(main_py),
        "--config-path",
        str((ROOT / config["config_path"]).resolve()),
        "--data-folder",
        str(data_folder.resolve()),
        "--runner",
        config["runner"],
        "--implementation",
        config["implementation"],
        "--model-family",
        config["model_family"],
        "--task-type",
        config["task_type"],
        "--strategy",
        config["strategy"],
        "--primary-metric",
        config["primary_metric"],
        "--metric-mode",
        config["metric_mode"],
        "--epochs",
        str(config["epochs"]),
        "--data-fraction",
        str(config["data_fraction"]),
        "--max-training-runs",
        str(config["max_training_runs"]),
        "--max-studies",
        str(config["max_studies"]),
        "--device",
        config.get("device", "auto"),
        "--search-space-json",
        json.dumps(config["search_space"], separators=(",", ":")),
    ]
    if config.get("precision"):
        command.extend(["--precision", config["precision"]])
    if config.get("eval_precision"):
        command.extend(["--eval-precision", config["eval_precision"]])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real ECAPA-TDNN HPO smoke demo.")
    parser.add_argument("--data-folder", required=True, help="Path to VoxCeleb data.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Demo JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command only.")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    command = _command(config, Path(args.data_folder))
    printable = " ".join(shlex.quote(part) for part in command)
    if args.dry_run:
        print(printable)
        return
    subprocess.run(command, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
