"""Exercise the real CPU recipe, prepared manifests and epoch stop hook."""

from functools import partial
import json
from pathlib import Path

import pytest

from agent.data_processing.preparation import (
    prepare_audio,
    prepared_training_overrides,
)
from tests.unit.test_audio_preparation import raw_audio
from tests.unit.test_training_control import specification


def test_speechbrain_uses_prepared_splits_and_stops_before_budget(
    tmp_path, minimal_config, monkeypatch
):
    torch = pytest.importorskip("torch")
    from agent.runners.speechbrain_dependency import require_speechbrain

    require_speechbrain()
    from speechbrain.utils.checkpoints import Checkpointer
    from speechbrain.utils.epoch_loop import EpochCounter
    from speechbrain.utils.train_logger import FileTrainLogger
    from speechbrain.nnet.schedulers import CyclicLRScheduler
    from speechbrain.utils.metric_stats import MetricStats
    from speechbrain.nnet.losses import classification_error
    from agent.runners import speechbrain_backend as backend

    class Normalize(torch.nn.Module):
        def forward(self, features, lengths):
            return features

    class Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 4)

        def forward(self, features):
            return self.linear(
                torch.stack((features.mean(1), features.std(1)), dim=1)
            ).unsqueeze(1)

    def cost(predictions, labels, lengths):
        return torch.nn.functional.cross_entropy(
            predictions.squeeze(1), labels.squeeze(1)
        )

    prepared = prepare_audio(raw_audio(tmp_path / "raw"), tmp_path / "prepared")
    folder = tmp_path / "training"
    folder.mkdir()
    modules = {
        "compute_features": torch.nn.Identity(),
        "mean_var_norm": Normalize(),
        "embedding_model": Embed(),
        "classifier": torch.nn.Linear(4, prepared["training_speaker_count"]),
    }
    counter = EpochCounter(8)
    scheduler = CyclicLRScheduler(base_lr=0.01, max_lr=0.01, step_size=10)
    hparams = {
        **prepared_training_overrides(prepared),
        "output_folder": str(folder),
        "save_folder": str(folder / "save"),
        "train_log": str(folder / "train_log.txt"),
        "train_logger": FileTrainLogger(str(folder / "train_log.txt")),
        "modules": modules,
        "opt_class": partial(torch.optim.SGD, lr=0.01),
        "compute_cost": cost,
        "lr_annealing": scheduler,
        "error_stats": partial(
            MetricStats, metric=partial(classification_error, reduction="batch")
        ),
        "epoch_counter": counter,
        "batch_size": 4,
        "dataloader_options": {"batch_size": 4, "num_workers": 0},
        "sentence_len": 0.2,
        "random_chunk": False,
        "validation_crop": "bounds",
    }
    hparams["checkpointer"] = Checkpointer(
        folder / "save",
        {
            "embedding": modules["embedding_model"],
            "classifier": modules["classifier"],
            "epoch_counter": counter,
        },
    )
    monkeypatch.setattr(
        backend, "_load_hyperpyyaml_config", lambda *args: (hparams, None)
    )
    monkeypatch.setattr(
        backend,
        "_prepare_cached_data",
        lambda *args: pytest.fail("prepared data must never be split again"),
    )
    result = backend.run_training(
        str(minimal_config),
        {
            "_prepared_data": prepared,
            "_run_opts": {"device": "cpu"},
            "_hpo_training_control": specification(
                min_epochs=2, patience=1, min_delta=1000
            ),
        },
    )
    assert result["status"] == "success", result.get("error")
    control = result["runtime"]["training_control"]
    assert control["should_stop"] and control["epoch"] == 2
    assert counter.current == 2 < 8
    assert (
        len(
            json.loads((folder / "training_control.json").read_text())[
                "history"
            ]
        )
        == 2
    )
    assert result["runtime"]["training_manifests"]["train_annotation"][
        "path"
    ] == str(Path(prepared["root"]) / "train.csv")
    assert list((folder / "save").glob("CKPT*"))
