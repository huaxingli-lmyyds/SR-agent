"""
SpeechBrain ECAPA-TDNN training pipeline wrapper.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from pathlib import Path
import importlib.util
import os

import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


_RECIPE_PATH = (
    Path(__file__).parent.parent.parent
    / "recipes"
    / "voxceleb"
    / "train_speaker_embeddings.py"
)

_PREPARE_PATH = (
    Path(__file__).parent.parent.parent
    / "recipes"
    / "voxceleb"
    / "voxceleb_prepare.py"
)


def _load_recipe_module():
    """Load the SpeechBrain recipe module from a file path."""
    spec = importlib.util.spec_from_file_location("voxceleb_train_recipe", _RECIPE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load recipe module: {_RECIPE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prepare_module():
    """Load the VoxCeleb prepare module from a file path."""
    spec = importlib.util.spec_from_file_location("voxceleb_prepare", _PREPARE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load prepare module: {_PREPARE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_pipeline(
    config_path: str,
    overrides: Optional[List[str]] = None,
    run_opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    """Run training without subprocess and return the validation EER."""
    recipe = _load_recipe_module()
    prepare_module = _load_prepare_module()
    overrides = overrides or []
    run_opts = run_opts or {}

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Enable CUDNN autotuning for better performance when shapes are stable.
    torch.backends.cudnn.benchmark = True

    # Initialize DDP when requested in run options.
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters with override support.
    with open(config_path, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    data_folder = hparams.get("data_folder")
    if not data_folder or data_folder == "!PLACEHOLDER":
        data_folder = "../datasets/voxceleb1"
        hparams["data_folder"] = data_folder

    # Download verification list used during prep.
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    recipe.download_file(hparams["verification_file"], veri_file_path)

    # Prepare VoxCeleb data (train/dev split).
    prepare_module.prepare_voxceleb(
        data_folder=hparams["data_folder"],
        save_folder=hparams["save_folder"],
        verification_pairs_file=veri_file_path,
        splits=["train", "dev"],
        split_ratio=hparams["split_ratio"],
        seg_dur=hparams["sentence_len"],
        skip_prep=hparams["skip_prep"],
    )

    # Prepare augmentation data if configured.
    if "prepare_noise_data" in hparams:
        sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
    if "prepare_rir_data" in hparams:
        sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

    # Prepare training and validation datasets via recipe pipeline.
    train_data, valid_data, _ = recipe.dataio_prep(hparams)

    # Create experiment directory and save hyperparameters.
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=config_path,
        overrides=overrides,
    )

    # Initialize the Brain with modules, optimizer, and checkpointer.
    speaker_brain = recipe.SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Run the training loop.
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Extract validation ErrorRate (EER) from the last validation stage.
    eer = None
    if hasattr(speaker_brain, "error_metrics"):
        try:
            eer = speaker_brain.error_metrics.summarize("average")
        except Exception:
            eer = None

    return {"eer": eer}
