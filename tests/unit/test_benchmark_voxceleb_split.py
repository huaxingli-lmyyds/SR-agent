"""Exercise the pure split function without importing the GPU/audio recipe stack."""

import ast
import glob
import logging
import os
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from recipes.voxceleb.verification_compat import parse_verification_pair


@pytest.mark.parametrize("split_speaker", [True, False])
def test_split_excludes_both_pair_sides_with_native_paths(
    tmp_path, split_speaker
):
    source = Path("recipes/voxceleb/voxceleb_prepare.py")
    module = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef) and n.name == "_get_utt_split_lists"
    )
    namespace = {
        "os": os,
        "glob": glob,
        "random": random,
        "logger": logging.getLogger(__name__),
        "parse_verification_pair": parse_verification_pair,
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )
    for speaker in ("left", "right_only", "training"):
        folder = tmp_path / "wav" / speaker / "video"
        folder.mkdir(parents=True)
        (folder / "audio.wav").write_bytes(b"fake")
    pairs = tmp_path / "pairs.txt"
    pairs.write_text(
        "0 left/video/audio.wav right_only/video/audio.wav\n", encoding="utf-8"
    )
    train, dev = namespace["_get_utt_split_lists"](
        [str(tmp_path)], [90, 10], str(pairs), split_speaker
    )
    assert len(train + dev) == 1
    assert "training" in str(Path((train + dev)[0]))


def test_preparation_seed_does_not_consume_training_random_state():
    from agent.runners.speechbrain_backend import _data_prep_random_state

    before = random.getstate()
    with _data_prep_random_state(17):
        first = [random.random() for _ in range(4)]
    assert random.getstate() == before
    with pytest.raises(RuntimeError):
        with _data_prep_random_state(17):
            assert [random.random() for _ in range(4)] == first
            raise RuntimeError("interrupted prep")
    assert random.getstate() == before


def test_random_segment_csv_reads_metadata_without_decoding(tmp_path):
    import csv

    source = Path("recipes/voxceleb/voxceleb_prepare.py")
    module = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_csv"
    )

    class MetadataOnlyAudioIO:
        @staticmethod
        def info(_path):
            return SimpleNamespace(sample_rate=16000, num_frames=32000)

        @staticmethod
        def load(_path):
            raise AssertionError("random-segment preparation decoded audio")

    namespace = {
        "csv": csv,
        "logger": logging.getLogger(__name__),
        "tqdm": lambda values, **_kwargs: values,
        "audio_io": MetadataOnlyAudioIO(),
        "SAMPLERATE": 16000,
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )
    destination = tmp_path / "train.csv"
    namespace["prepare_csv"](
        3.0,
        ["/dataset/wav/id10001/video/00001.wav"],
        destination,
        random_segment=True,
    )
    rows = list(csv.DictReader(destination.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["duration"] == "2.0"
    assert rows[0]["start"] == "0"
    assert rows[0]["stop"] == "32000"


def test_preparation_cache_is_scoped_and_fingerprinted(tmp_path, monkeypatch):
    from agent.runners import speechbrain_backend as backend

    calls = []
    monkeypatch.setattr(
        backend,
        "run_data_prep",
        lambda **kw: calls.append(kw) or {"status": "success"},
    )
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("pairs v1", encoding="utf-8")
    cfg = {
        "data_folder": str(tmp_path / "dataset"),
        "verification_file": str(pairs),
        "split_ratio": [90, 10],
        "data_prep_seed": 0,
        "prep_cache_root": str(tmp_path / "group_a"),
    }
    backend._prepare_cached_data("train", cfg, ["train", "dev"], 3.0)
    backend._prepare_cached_data("train", cfg, ["train", "dev"], 3.0)
    assert calls[0] == calls[1]
    cfg["prep_cache_root"] = str(tmp_path / "group_b")
    backend._prepare_cached_data("train", cfg, ["train", "dev"], 3.0)
    assert Path(calls[-1]["save_folder"]).parent == tmp_path / "group_b"
    assert (
        Path(calls[-1]["save_folder"]).name
        == Path(calls[0]["save_folder"]).name
    )
    pairs.write_text("pairs v2", encoding="utf-8")
    backend._prepare_cached_data("train", cfg, ["train", "dev"], 3.0)
    assert calls[-1]["save_folder"] != calls[-2]["save_folder"]
    cfg["data_prep_seed"] = 1
    backend._prepare_cached_data("train", cfg, ["train", "dev"], 3.0)
    assert calls[-1]["save_folder"] != calls[-2]["save_folder"]
    monkeypatch.setattr(
        backend,
        "run_data_prep",
        lambda **kw: {"status": "failed", "error": "broken"},
    )
    with pytest.raises(RuntimeError, match="isolated data preparation failed"):
        backend._prepare_cached_data("train", cfg, ["train", "dev"], 3.0)
