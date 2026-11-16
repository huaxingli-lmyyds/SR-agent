from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agent.runners import RUNNER_ADAPTERS


def test_demo_config_uses_real_speechbrain_runner() -> None:
    config = json.loads(Path("demo/config/ecapa_smoke.json").read_text(encoding="utf-8"))

    assert config["runner"] == "speechbrain"
    assert config["implementation"] == "speechbrain"
    assert config["model_family"] == "ecapa_tdnn"
    assert config["search_space"]["parameters"]
    assert "demo" not in RUNNER_ADAPTERS


def test_real_demo_dry_run_builds_main_command(tmp_path) -> None:
    dataset = tmp_path / "voxceleb1"
    dataset.mkdir()

    completed = subprocess.run(
        [
            sys.executable,
            "demo/run_ecapa_hpo.py",
            "--data-folder",
            str(dataset),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "main.py" in completed.stdout
    assert "--runner speechbrain" in completed.stdout
    assert "--implementation speechbrain" in completed.stdout
    assert "--search-space-json" in completed.stdout
    assert str(dataset.resolve()) in completed.stdout


def test_demo_readme_is_allowed_by_gitignore() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    readme = Path("demo/README.md")

    assert readme.exists()
    assert "!/demo/README.md" in gitignore