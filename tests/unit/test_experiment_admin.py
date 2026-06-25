import json
from pathlib import Path
import subprocess
import sys
import zipfile


SCRIPT = Path(__file__).parents[2] / "scripts" / "experiment_admin.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_filtered_archive_and_clear_keep_indexes_consistent(tmp_path) -> None:
    experiment_id = "20260621_120000_0"
    experiment_dir = tmp_path / "agent" / "experiments" / "hpo" / experiment_id
    experiment_dir.mkdir(parents=True)
    record = {
        "experiment_id": experiment_id,
        "experiment_type": "hpo",
        "model": {"family": "ecapa_tdnn"},
        "version": {"campaign_id": "campaign_001"},
    }
    (experiment_dir / "experiment_record.json").write_text(json.dumps(record), encoding="utf-8")
    history = experiment_dir.parent / "experiments_history.json"
    history.write_text(json.dumps([record]), encoding="utf-8")
    pointer = (
        experiment_dir.parent
        / "_catalog"
        / "speaker_verification"
        / "ecapa_tdnn"
        / "speechbrain"
        / "campaign_001"
        / f"{experiment_id}.json"
    )
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"experiment_id": experiment_id}), encoding="utf-8")

    _run(
        "archive",
        "--workspace-root",
        tmp_path,
        "--model-family",
        "ecapa_tdnn",
        "--name",
        "ecapa.zip",
    )
    archive_path = tmp_path / "experiment_archives" / "ecapa.zip"
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("archive_manifest.json"))
        assert manifest["experiment_ids"] == [experiment_id]
        assert any(name.endswith("experiment_record.json") for name in archive.namelist())

    preview = _run(
        "clear",
        "--workspace-root",
        tmp_path,
        "--model-family",
        "ecapa_tdnn",
    )
    assert "Preview only" in preview.stdout
    assert experiment_dir.exists()

    _run(
        "clear",
        "--workspace-root",
        tmp_path,
        "--model-family",
        "ecapa_tdnn",
        "--execute",
    )
    assert not experiment_dir.exists()
    assert json.loads(history.read_text(encoding="utf-8")) == []
    assert not pointer.exists()
