"""
SpeechBrain ECAPA-TDNN evaluation pipeline wrapper.
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
    / "speaker_verification_cosine.py"
)

_PREPARE_PATH = (
    Path(__file__).parent.parent.parent
    / "recipes"
    / "voxceleb"
    / "voxceleb_prepare.py"
)


def _load_verification_module():
    """Load the SpeechBrain verification recipe module from a file path."""
    spec = importlib.util.spec_from_file_location("voxceleb_verification", _RECIPE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load verification recipe: {_RECIPE_PATH}")
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


def eval_pipeline(
    config_path: str,
    model_path: Optional[str] = None,
    data_folder: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    run_opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run evaluation without subprocess and return metrics and outputs."""
    verification_module = _load_verification_module()
    prepare_module = _load_prepare_module()

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    overrides = overrides or []
    run_opts = run_opts or {}

    if data_folder:
        overrides.append(f"data_folder: {data_folder}")
    if model_path:
        overrides.append(f"pretrainer.paths.embedding_model: {model_path}")

    if "device" not in run_opts:
        run_opts["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path, encoding="utf-8") as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Download verification list
    veri_file_path = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file"])
    )
    verification_module.download_file(params["verification_file"], veri_file_path)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=config_path,
        overrides=overrides,
    )

    # Prepare VoxCeleb data
    prepare_module.prepare_voxceleb(
        data_folder=params["data_folder"],
        save_folder=params["save_folder"],
        verification_pairs_file=veri_file_path,
        splits=["train", "dev", "test"],
        split_ratio=params["split_ratio"],
        seg_dur=3.0,
        skip_prep=params["skip_prep"],
        source=(params["voxceleb_source"] if "voxceleb_source" in params else None),
    )

    # Prepare dataloaders
    verification_module.params = params
    verification_module.run_opts = run_opts
    train_dataloader, enrol_dataloader, test_dataloader = verification_module.dataio_prep(params)

    # Load pretrained model
    params["pretrainer"].collect_files()
    params["pretrainer"].load_collected()
    params["embedding_model"].eval()
    params["embedding_model"].to(run_opts["device"])

    # Compute embeddings
    verification_module.enrol_dict = verification_module.compute_embedding_loop(enrol_dataloader)
    verification_module.test_dict = verification_module.compute_embedding_loop(test_dataloader)
    if "score_norm" in params:
        verification_module.train_dict = verification_module.compute_embedding_loop(train_dataloader)

    # Compute scores
    with open(veri_file_path, encoding="utf-8") as f:
        veri_test = [line.rstrip() for line in f]

    positive_scores, negative_scores = verification_module.get_verification_scores(veri_test)

    eer, _ = verification_module.EER(
        torch.tensor(positive_scores), torch.tensor(negative_scores)
    )
    min_dcf, _ = verification_module.minDCF(
        torch.tensor(positive_scores), torch.tensor(negative_scores)
    )

    scores_path = Path(params["output_folder"]) / "scores.txt"

    return {
        "eer": float(eer) * 100,
        "min_dcf": float(min_dcf) * 100,
        "output_folder": str(params["output_folder"]),
        "scores_path": str(scores_path),
    }
