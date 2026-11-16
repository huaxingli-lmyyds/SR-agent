"""Cheap checks of verification contracts before an expensive training run."""

from __future__ import annotations

import pickle

from agent.utils.config_parser import ConfigParser


def validate_verification_config(config_path: str) -> None:
    config = ConfigParser(config_path).load_config()
    if (
        "compute_plda" in config
        or config.get("scoring_backend", "cosine") != "cosine"
    ):
        raise ValueError(
            "unsupported verification backend: SpeechBrainRunner uses cosine scoring; "
            "PLDA configurations cannot be evaluated by this runner. "
            "For x-vector use configs/verification_cosine_xvector.yaml."
        )
    workers = any(
        int((config.get(key) or {}).get("num_workers", 0)) > 0
        for key in (
            "train_dataloader_opts",
            "enrol_dataloader_opts",
            "test_dataloader_opts",
        )
    )
    if workers:
        from recipes.voxceleb.speaker_verification_cosine import audio_pipeline
        from speechbrain.utils.data_pipeline import DynamicItem

        # Test the wrapped object too: a module-level decorated function can
        # still be unpickleable because the decorator replaced its global name.
        pickle.dumps(
            DynamicItem(
                func=audio_pipeline,
                takes=["wav", "start", "stop"],
                provides=["sig"],
            )
        )
