"""
Central SpeechBrain runner utilities.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import os

try:
    from recipes.voxceleb import train_speaker_embeddings as _TRAIN_RECIPE_MODULE
    _TRAIN_RECIPE_ERROR = None
except Exception as exc:
    _TRAIN_RECIPE_MODULE = None
    _TRAIN_RECIPE_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from recipes.voxceleb import speaker_verification_cosine as _VERIFICATION_MODULE
    _VERIFICATION_ERROR = None
except Exception as exc:
    _VERIFICATION_MODULE = None
    _VERIFICATION_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from recipes.voxceleb import voxceleb_prepare as _PREPARE_MODULE
    _PREPARE_ERROR = None
except Exception as exc:
    _PREPARE_MODULE = None
    _PREPARE_ERROR = f"{type(exc).__name__}: {exc}"


def _load_hyperpyyaml_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        from hyperpyyaml import load_hyperpyyaml

        with open(config_path, encoding="utf-8") as fin:
            params = load_hyperpyyaml(fin, overrides or [])
        return params, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def run_data_prep(
    data_folder: str,
    save_folder: str,
    verification_file: str,
    split_ratio: List[Any],
    sentence_len: float,
    splits: List[str],
    skip_prep: bool,
    source: Optional[str],
    random_segment: bool,
    amp_th: float,
    split_speaker: bool,
) -> Dict[str, Any]:
    try:
        from speechbrain.utils.data_utils import download_file

        if _PREPARE_MODULE is None:
            return {
                "status": "failed",
                "error": _PREPARE_ERROR or "Prepare module failed to load",
                "verification_local": None,
                "save_folder": str(save_folder),
            }

        prepare_module = _PREPARE_MODULE

        save_path = Path(save_folder)
        save_path.mkdir(parents=True, exist_ok=True)
        verification_local = save_path / os.path.basename(str(verification_file))
        download_file(str(verification_file), str(verification_local))

        prepare_module.prepare_voxceleb(
            data_folder=data_folder,
            save_folder=str(save_path),
            verification_pairs_file=str(verification_local),
            splits=splits,
            split_ratio=split_ratio,
            seg_dur=sentence_len,
            amp_th=amp_th,
            source=source,
            split_speaker=split_speaker,
            random_segment=random_segment,
            skip_prep=skip_prep,
        )

        return {
            "status": "success",
            "error": None,
            "verification_local": str(verification_local),
            "save_folder": str(save_path),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "verification_local": None,
            "save_folder": str(save_folder),
        }


def run_evaluation(
    config_path: str,
    model_path: Optional[str] = None,
    data_folder: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    run_opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        import torch
        import speechbrain as sb

        if _VERIFICATION_MODULE is None:
            return {
                "status": "failed",
                "error": _VERIFICATION_ERROR or "Verification module failed to load",
                "eer": None,
                "min_dcf": None,
                "output_folder": None,
                "scores_path": None,
            }

        if _PREPARE_MODULE is None:
            return {
                "status": "failed",
                "error": _PREPARE_ERROR or "Prepare module failed to load",
                "eer": None,
                "min_dcf": None,
                "output_folder": None,
                "scores_path": None,
            }

        verification_module = _VERIFICATION_MODULE
        prepare_module = _PREPARE_MODULE

        if not Path(config_path).exists():
            return {
                "status": "failed",
                "error": f"Config not found: {config_path}",
                "eer": None,
                "min_dcf": None,
                "output_folder": None,
                "scores_path": None,
            }

        overrides = list(overrides or [])
        run_opts = dict(run_opts or {})

        if data_folder:
            overrides.append(f"data_folder: {data_folder}")
        if model_path:
            overrides.append(f"pretrainer.paths.embedding_model: {model_path}")

        if "device" not in run_opts:
            run_opts["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        params, err = _load_hyperpyyaml_config(config_path, overrides)
        if err:
            return {
                "status": "failed",
                "error": err,
                "eer": None,
                "min_dcf": None,
                "output_folder": None,
                "scores_path": None,
            }

        veri_file_path = os.path.join(
            params["save_folder"], os.path.basename(params["verification_file"])
        )
        verification_module.download_file(params["verification_file"], veri_file_path)

        sb.core.create_experiment_directory(
            experiment_directory=params["output_folder"],
            hyperparams_to_save=config_path,
            overrides=overrides,
        )

        prepare_module.prepare_voxceleb(
            data_folder=params["data_folder"],
            save_folder=params["save_folder"],
            verification_pairs_file=veri_file_path,
            splits=["train", "dev", "test"],
            split_ratio=params["split_ratio"],
            seg_dur=3.0,
            skip_prep=params["skip_prep"],
            source=(params.get("voxceleb_source")),
        )

        verification_module.params = params
        verification_module.run_opts = run_opts
        train_dataloader, enrol_dataloader, test_dataloader = (
            verification_module.dataio_prep(params)
        )

        params["pretrainer"].collect_files()
        params["pretrainer"].load_collected()
        params["embedding_model"].eval()
        params["embedding_model"].to(run_opts["device"])

        verification_module.enrol_dict = verification_module.compute_embedding_loop(
            enrol_dataloader
        )
        verification_module.test_dict = verification_module.compute_embedding_loop(
            test_dataloader
        )
        if "score_norm" in params:
            verification_module.train_dict = verification_module.compute_embedding_loop(
                train_dataloader
            )

        with open(veri_file_path, encoding="utf-8") as f:
            veri_test = [line.rstrip() for line in f]

        positive_scores, negative_scores = verification_module.get_verification_scores(
            veri_test
        )

        eer, _ = verification_module.EER(
            torch.tensor(positive_scores), torch.tensor(negative_scores)
        )
        min_dcf, _ = verification_module.minDCF(
            torch.tensor(positive_scores), torch.tensor(negative_scores)
        )

        scores_path = Path(params["output_folder"]) / "scores.txt"

        return {
            "status": "success",
            "error": None,
            "eer": float(eer) * 100,
            "min_dcf": float(min_dcf) * 100,
            "output_folder": str(params["output_folder"]),
            "scores_path": str(scores_path),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "eer": None,
            "min_dcf": None,
            "output_folder": None,
            "scores_path": None,
        }


def run_training(config_path: str, overrides: List[str]) -> Dict[str, Any]:
    """
    Run SpeechBrain training pipeline.

    Returns:
        {"status": "success"|"failed", "eer": float|None, "error": str|None}
    """
    try:
        import torch
        import speechbrain as sb

        if _TRAIN_RECIPE_MODULE is None:
            return {
                "status": "failed",
                "eer": None,
                "error": _TRAIN_RECIPE_ERROR or "Training recipe failed to load",
            }

        if _PREPARE_MODULE is None:
            return {
                "status": "failed",
                "eer": None,
                "error": _PREPARE_ERROR or "Prepare module failed to load",
            }

        recipe = _TRAIN_RECIPE_MODULE
        prepare_module = _PREPARE_MODULE

        if not Path(config_path).exists():
            return {
                "status": "failed",
                "eer": None,
                "error": f"Config not found: {config_path}",
            }

        overrides = list(overrides or [])
        run_opts: Dict[str, Any] = {}

        torch.backends.cudnn.benchmark = True
        sb.utils.distributed.ddp_init_group(run_opts)

        hparams, err = _load_hyperpyyaml_config(config_path, overrides)
        if err:
            return {"status": "failed", "eer": None, "error": err}

        data_folder = hparams.get("data_folder")
        if not data_folder or data_folder == "!PLACEHOLDER":
            data_folder = "../datasets/voxceleb1"
            hparams["data_folder"] = data_folder

        veri_file_path = os.path.join(
            hparams["save_folder"], os.path.basename(hparams["verification_file"])
        )
        recipe.download_file(hparams["verification_file"], veri_file_path)

        prepare_module.prepare_voxceleb(
            data_folder=hparams["data_folder"],
            save_folder=hparams["save_folder"],
            verification_pairs_file=veri_file_path,
            splits=["train", "dev"],
            split_ratio=hparams["split_ratio"],
            seg_dur=hparams["sentence_len"],
            skip_prep=hparams["skip_prep"],
        )

        if "prepare_noise_data" in hparams:
            sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
        if "prepare_rir_data" in hparams:
            sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

        train_data, valid_data, _ = recipe.dataio_prep(hparams)

        sb.core.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=config_path,
            overrides=overrides,
        )

        speaker_brain = recipe.SpeakerBrain(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )

        speaker_brain.fit(
            speaker_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )

        eer = None
        if hasattr(speaker_brain, "error_metrics"):
            try:
                eer = speaker_brain.error_metrics.summarize("average")
            except Exception:
                eer = None

        return {"status": "success", "eer": eer, "error": None}
    except Exception as exc:
        return {
            "status": "failed",
            "eer": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


__all__ = ["run_training", "run_data_prep", "run_evaluation"]
