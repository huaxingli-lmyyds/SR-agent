"""Low-level SpeechBrain execution functions used by SpeechBrainRunnerAdapter."""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import hashlib
import json
import re
import os
import shutil
import traceback

from agent.utils.path_tool import (
    get_prep_cache_dir,
    is_remote_path,
    resolve_config_path,
    resolve_config_value_path,
    resolve_data_path,
    resolve_project_path,
)
from agent.runners.speechbrain_dependency import patch_torchaudio_compatibility, require_speechbrain

patch_torchaudio_compatibility()

def _format_exception(exc: Exception) -> str:
    """Return an actionable exception summary with traceback context."""
    return f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

try:
    from recipes.voxceleb import train_speaker_embeddings as _TRAIN_RECIPE_MODULE
    _TRAIN_RECIPE_ERROR = None
except Exception as exc:
    _TRAIN_RECIPE_MODULE = None
    _TRAIN_RECIPE_ERROR = _format_exception(exc)

try:
    from recipes.voxceleb import speaker_verification_cosine as _VERIFICATION_MODULE
    _VERIFICATION_ERROR = None
except Exception as exc:
    _VERIFICATION_MODULE = None
    _VERIFICATION_ERROR = _format_exception(exc)

try:
    from recipes.voxceleb import voxceleb_prepare as _PREPARE_MODULE
    _PREPARE_ERROR = None
except Exception as exc:
    _PREPARE_MODULE = None
    _PREPARE_ERROR = _format_exception(exc)


def _load_hyperpyyaml_config(
    config_path: str,
    overrides: Optional[Union[List[str], Dict[str, Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        from hyperpyyaml import load_hyperpyyaml

        normalized_overrides: Union[List[str], Dict[str, Any]]
        if isinstance(overrides, dict):
            normalized_overrides = overrides
        elif isinstance(overrides, list):
            normalized_overrides = overrides
        else:
            normalized_overrides = []

        config_path = str(resolve_config_path(config_path))
        with open(config_path, encoding="utf-8") as fin:
            params = load_hyperpyyaml(fin, normalized_overrides)
        return params, None
    except Exception as exc:
        return None, _format_exception(exc)



def _runtime_options_from_overrides(
    overrides: Union[List[str], Dict[str, Any]],
) -> Tuple[Union[List[str], Dict[str, Any]], Dict[str, Any]]:
    if isinstance(overrides, dict):
        normalized = dict(overrides)
        raw = normalized.pop("_run_opts", {}) or {}
        return normalized, dict(raw) if isinstance(raw, dict) else {}
    return overrides, {}


def _resolve_run_opts(torch_module: Any, requested: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    run_opts = dict(requested or {})
    device = str(run_opts.get("device") or "auto").strip().lower()
    if device in {"", "auto"}:
        device = "cuda" if torch_module.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch_module.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for SpeechBrain training/evaluation, but "
                "torch.cuda.is_available() is False. Check the CUDA-matched "
                "torch installation and CUDA_VISIBLE_DEVICES."
            )
        if ":" in device:
            try:
                torch_module.cuda.set_device(int(device.split(":", 1)[1]))
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(f"invalid CUDA device requested: {device}") from exc
    run_opts["device"] = device
    return run_opts

def _extract_best_valid_error_rate(train_log_path: Path) -> Optional[float]:
    if not train_log_path.exists():
        return None

    pattern = re.compile(
        r"valid ErrorRate:\s*([\d.eE+-]+)"
    )
    best = None
    with train_log_path.open("r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            match = pattern.search(line)
            if not match:
                continue
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if best is None or value < best:
                best = value
    return best


def _get_prep_cache_dir(tag: str) -> Path:
    cache_dir = get_prep_cache_dir(tag)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _prep_cache_tag(stage: str, **parameters: Any) -> str:
    payload = json.dumps(parameters, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{stage}_{digest}"


def _required_prep_files(splits: List[str], verification_file: str) -> List[str]:
    files = ["opt_voxceleb_prepare.pkl"]
    if "train" in splits:
        files.append("train.csv")
    if "dev" in splits:
        files.append("dev.csv")
    if "test" in splits:
        files.extend(["test.csv", "enrol.csv"])
    files.append(os.path.basename(verification_file))
    return files


def _link_from_cache(cache_dir: Path, save_dir: Path, files: List[str]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        src = cache_dir / name
        dst = save_dir / name
        if dst.exists() or not src.exists():
            continue
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


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
    signal:str = None
) -> Dict[str, Any]:
    try:
        require_speechbrain()
        from speechbrain.utils.data_utils import download_file

        data_folder = str(resolve_data_path(data_folder))
        save_folder = str(resolve_project_path(save_folder))
        if source and not is_remote_path(source):
            source = str(resolve_project_path(source))

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
            save_folder=str(save_folder),
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
            "save_folder": str(save_folder),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": _format_exception(exc),
            "verification_local": None,
            "save_folder": str(save_folder),
        }


def run_evaluation(
    config_path: str,
    model_path: Optional[str] = None,
    data_folder: Optional[str] = None,
    overrides: Optional[Union[List[str], Dict[str, Any]]] = None,
    run_opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        require_speechbrain()
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

        config_path = str(resolve_config_path(config_path))
        if not Path(config_path).exists():
            return {
                "status": "failed",
                "error": f"Config not found: {config_path}",
                "eer": None,
                "min_dcf": None,
                "output_folder": None,
                "scores_path": None,
            }

        if isinstance(overrides, dict):
            overrides, override_run_opts = _runtime_options_from_overrides(overrides)
        else:
            overrides = list(overrides or [])
            override_run_opts = {}
        run_opts = {**override_run_opts, **dict(run_opts or {})}

        if data_folder:
            if isinstance(overrides, dict):
                overrides["data_folder"] = data_folder
            else:
                overrides.append(f"data_folder: {data_folder}")
        if model_path:
            if isinstance(overrides, dict):
                overrides["pretrain_path"] = model_path
            else:
                overrides.append(f"pretrain_path: {model_path}")
                
        run_opts = _resolve_run_opts(torch, run_opts)
        print(f"SR-agent SpeechBrain evaluation run_opts: {run_opts}", flush=True)

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

        params["data_folder"] = str(resolve_data_path(params.get("data_folder")))
        for key in ("output_folder", "save_folder"):
            resolved = resolve_config_value_path(params.get(key))
            if resolved is not None:
                params[key] = str(resolved)

        splits = ["train", "dev", "test"]
        cache_dir = _get_prep_cache_dir(_prep_cache_tag(
            "eval",
            data_folder=params["data_folder"],
            verification_file=params["verification_file"],
            split_ratio=params["split_ratio"],
            splits=splits,
        ))
        cache_result = run_data_prep(
            data_folder=params["data_folder"],
            save_folder=str(cache_dir),
            verification_file=params["verification_file"],
            split_ratio=params["split_ratio"],
            sentence_len=3.0,
            splits=splits,
            skip_prep=False,
            source=params.get("voxceleb_source"),
            random_segment=False,
            amp_th=params.get("amp_th", 5e-4),
            split_speaker=params.get("split_speaker", False),
        )

        if cache_result.get("status") == "success":
            required_files = _required_prep_files(splits, params["verification_file"])
            _link_from_cache(Path(cache_result["save_folder"]), Path(params["save_folder"]), required_files)
            params["skip_prep"] = True

        veri_file_path = os.path.join(
            params["save_folder"], os.path.basename(params["verification_file"])
        )
        if not Path(veri_file_path).exists():
            verification_module.download_file(params["verification_file"], veri_file_path)

        sb.core.create_experiment_directory(
            experiment_directory=params["output_folder"],
            hyperparams_to_save=config_path,
            overrides=overrides,
        )

        if not params.get("skip_prep", False):
            prepare_module.prepare_voxceleb(
                data_folder=params["data_folder"],
                save_folder=params["save_folder"],
                verification_pairs_file=veri_file_path,
                splits=splits,
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
        if verification_module.score_norm_mode():
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
            "error": _format_exception(exc),
            "eer": None,
            "min_dcf": None,
            "output_folder": None,
            "scores_path": None,
        }


def run_training(config_path: str, overrides: Union[List[str], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run SpeechBrain training pipeline.

    Returns:
        {"status": "success"|"failed", "valid_error_rate": float|None, "error": str|None}
    """
    try:
        require_speechbrain()
        import torch
        import speechbrain as sb

        if _TRAIN_RECIPE_MODULE is None:
            return {
                "status": "failed",
                "valid_error_rate": None,
                "error": _TRAIN_RECIPE_ERROR or "Training recipe failed to load",
            }

        if _PREPARE_MODULE is None:
            return {
                "status": "failed",
                "valid_error_rate": None,
                "error": _PREPARE_ERROR or "Prepare module failed to load",
            }

        recipe = _TRAIN_RECIPE_MODULE
        prepare_module = _PREPARE_MODULE

        config_path = str(resolve_config_path(config_path))
        if not Path(config_path).exists():
            return {
                "status": "failed",
                "valid_error_rate": None,
                "error": f"Config not found: {config_path}",
            }

        data_fraction = None
        if isinstance(overrides, dict):
            normalized_overrides, run_opts = _runtime_options_from_overrides(overrides)
            data_fraction = normalized_overrides.pop("_hpo_data_fraction", None)
            normalized_overrides.pop("_hpo_max_duration_seconds", None)
        elif isinstance(overrides, list):
            normalized_overrides = overrides
            run_opts = {}
        else:
            normalized_overrides = []
            run_opts = {}
        run_opts = _resolve_run_opts(torch, run_opts)
        print(f"SR-agent SpeechBrain training run_opts: {run_opts}", flush=True)

        torch.backends.cudnn.benchmark = True
        sb.utils.distributed.ddp_init_group(run_opts)

        hparams, err = _load_hyperpyyaml_config(config_path, normalized_overrides)
        if err:
            return {"status": "failed", "valid_error_rate": None, "error": err}

        data_folder = hparams.get("data_folder")
        if not data_folder or data_folder == "!PLACEHOLDER":
            data_folder = str(resolve_data_path())
            hparams["data_folder"] = data_folder
        else:
            hparams["data_folder"] = str(resolve_data_path(data_folder))

        for key in ("output_folder", "save_folder", "train_log"):
            resolved = resolve_config_value_path(hparams.get(key))
            if resolved is not None:
                hparams[key] = str(resolved)

        splits = ["train", "dev"]
        cache_dir = _get_prep_cache_dir(_prep_cache_tag(
            "train",
            data_folder=hparams["data_folder"],
            verification_file=hparams["verification_file"],
            split_ratio=hparams["split_ratio"],
            sentence_len=hparams["sentence_len"],
            splits=splits,
            random_chunk=hparams.get("random_chunk", False),
            split_speaker=hparams.get("split_speaker", False),
        ))
        cache_result = run_data_prep(
            data_folder=hparams["data_folder"],
            save_folder=str(cache_dir),
            verification_file=hparams["verification_file"],
            split_ratio=hparams["split_ratio"],
            sentence_len=hparams["sentence_len"],
            splits=splits,
            skip_prep=False,
            source=hparams.get("voxceleb_source"),
            random_segment=hparams.get("random_chunk", False),
            amp_th=hparams.get("amp_th", 5e-4),
            split_speaker=hparams.get("split_speaker", False),
        )

        if cache_result.get("status") == "success":
            required_files = _required_prep_files(splits, hparams["verification_file"])
            _link_from_cache(Path(cache_result["save_folder"]), Path(hparams["save_folder"]), required_files)
            hparams["skip_prep"] = True

        veri_file_path = os.path.join(
            hparams["save_folder"], os.path.basename(hparams["verification_file"])
        )
        if not Path(veri_file_path).exists():
            recipe.download_file(hparams["verification_file"], veri_file_path)

        if not hparams.get("skip_prep", False):
            prepare_module.prepare_voxceleb(
                data_folder=hparams["data_folder"],
                save_folder=hparams["save_folder"],
                verification_pairs_file=veri_file_path,
                splits=splits,
                split_ratio=hparams["split_ratio"],
                seg_dur=hparams["sentence_len"],
                skip_prep=hparams["skip_prep"],
            )

        if "prepare_noise_data" in hparams:
            sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
        if "prepare_rir_data" in hparams:
            sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

        train_data, valid_data, _ = recipe.dataio_prep(hparams)
        if data_fraction is not None and float(data_fraction) < 1.0:
            from torch.utils.data import Subset

            sample_count = max(1, int(len(train_data) * float(data_fraction)))
            train_data = Subset(train_data, range(sample_count))

        sb.core.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=config_path,
            overrides=normalized_overrides,
        )

        speaker_brain = recipe.SpeakerBrain(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )

        dataloader_options = recipe.with_padded_batch(hparams["dataloader_options"])
        speaker_brain.fit(
            speaker_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=dataloader_options,
            valid_loader_kwargs=dataloader_options,
        )

        train_log_path = hparams.get("train_log")
        if train_log_path:
            log_path = Path(str(train_log_path))
        else:
            log_path = Path(hparams["output_folder"]) / "train_log.txt"

        valid_error_rate = _extract_best_valid_error_rate(log_path)

        return {"status": "success", "valid_error_rate": valid_error_rate, "error": None, "runtime": {"run_opts": dict(run_opts)}}
    except Exception as exc:
        return {
            "status": "failed",
            "valid_error_rate": None,
            "error": _format_exception(exc),
        }


__all__ = ["run_training", "run_data_prep", "run_evaluation"]
