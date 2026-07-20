from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.evaluate_checkpoint import (
    build_parser,
    evaluate_checkpoint,
)


def _arguments(tmp_path: Path, checkpoint: Path):
    data_folder = tmp_path / "data"
    data_folder.mkdir(exist_ok=True)
    verification_file = tmp_path / "pairs.txt"
    verification_file.write_text("1 a.wav b.wav\n", encoding="utf-8")
    config = tmp_path / "verification.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    return build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-folder",
            str(data_folder),
            "--verification-file",
            str(verification_file),
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--device",
            "cuda:0",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
        ]
    )


def test_evaluate_checkpoint_passes_explicit_inputs_and_writes_report(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "CKPT+test"
    checkpoint_dir.mkdir()
    checkpoint_file = checkpoint_dir / "embedding_model.ckpt"
    checkpoint_file.write_bytes(b"checkpoint")
    args = _arguments(tmp_path, checkpoint_dir)
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        output = Path(kwargs["overrides"]["output_folder"])
        scores = output / "scores.txt"
        scores.write_text("a b 1 0.9\n", encoding="utf-8")
        return {
            "status": "success",
            "eer": 2.5,
            "min_dcf": 0.003,
            "output_folder": str(output),
            "scores_path": str(scores),
            "error": None,
        }

    report = evaluate_checkpoint(args, runner=fake_runner)

    assert captured["model_path"] == str(checkpoint_dir.resolve())
    assert captured["overrides"]["verification_file"].endswith("pairs.txt")
    assert captured["overrides"]["batch_size"] == 4
    assert captured["overrides"]["_run_opts"] == {"device": "cuda:0"}
    assert report["metrics"] == {"eer_percent": 2.5, "min_dcf": 0.003}
    saved = json.loads(
        (tmp_path / "evaluation" / "evaluation_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["checkpoint"]["file"] == str(checkpoint_file.resolve())


def test_evaluate_checkpoint_accepts_embedding_model_file(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "CKPT+test"
    checkpoint_dir.mkdir()
    checkpoint_file = checkpoint_dir / "embedding_model.ckpt"
    checkpoint_file.write_bytes(b"checkpoint")
    args = _arguments(tmp_path, checkpoint_file)

    report = evaluate_checkpoint(
        args,
        runner=lambda **kwargs: {
            "status": "success",
            "eer": 1.0,
            "min_dcf": 0.001,
            "output_folder": kwargs["overrides"]["output_folder"],
            "scores_path": None,
            "error": None,
        },
    )

    assert report["checkpoint"]["directory"] == str(checkpoint_dir.resolve())


def test_evaluate_checkpoint_rejects_directory_without_model_file(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "CKPT+empty"
    checkpoint_dir.mkdir()
    args = _arguments(tmp_path, checkpoint_dir)

    with pytest.raises(ValueError, match="embedding_model.ckpt"):
        evaluate_checkpoint(args, runner=lambda **kwargs: {})
