"""VoxCeleb preparation tool using the same options as the training runner."""

import csv
import json
import math
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from agent.core.contracts import Artifact, OperationResult
from agent.runners.contracts import validate_runner_result
from agent.utils import ConfigParser, ExperimentTracker
from agent.utils.path_tool import (
    get_experiment_artifact_dir,
    is_remote_path,
    resolve_config_path,
    resolve_config_value_path,
    resolve_data_path,
)


def _parse_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _split_ratio(value: Any) -> list[float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = value.split(",")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) for item in value)
    ):
        raise ValueError(
            "split_ratio must contain two positive percentages summing to 100"
        )
    ratio = [float(item) for item in value]
    if any(
        not math.isfinite(item) or item <= 0 for item in ratio
    ) or not math.isclose(sum(ratio), 100):
        raise ValueError(
            "split_ratio must contain two positive percentages summing to 100"
        )
    return ratio


def _count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


@tool
def PrepareVoxCelebData(
    config_path: str | None = None,
    experiment_id: str | None = None,
    split_ratio: str | None = None,
    sentence_len: str | None = None,
    skip_prep: str | None = None,
    random_segment: str | None = None,
    split_speaker: str | None = None,
    split_unit: str | None = None,
    data_prep_seed: int | None = None,
) -> str:
    """Prepare VoxCeleb manifests; explicit options override training YAML defaults.

    split_ratio accepts a JSON array or comma-separated percentages. split_unit
    is utterance or session; data_prep_seed controls preparation independently
    of model randomness. random_segment defaults to the YAML random_chunk value.
    """
    tracker = None
    try:
        tracker = ExperimentTracker()
        config = ConfigParser(
            str(resolve_config_path(config_path))
        ).load_config(resolve_references=True)
        save_folder = config.get("save_folder")
        if not save_folder or not config.get("verification_file"):
            raise ValueError(
                "save_folder and verification_file must be configured"
            )
        save_folder = (
            get_experiment_artifact_dir(
                experiment_id, "data", "data_processing", create=True
            )
            if experiment_id
            else resolve_config_value_path(save_folder)
        )
        source = config.get("voxceleb_source")
        if source and not is_remote_path(source):
            source = str(resolve_config_value_path(source))
        length = float(
            sentence_len
            if sentence_len is not None
            else config.get("sentence_len", 3.0)
        )
        if not math.isfinite(length) or length <= 0:
            raise ValueError("sentence_len must be finite and positive")
        options = {
            "data_folder": str(resolve_data_path(config.get("data_folder"))),
            "save_folder": str(save_folder),
            "verification_file": str(config["verification_file"]),
            "split_ratio": _split_ratio(
                split_ratio
                if split_ratio is not None
                else config.get("split_ratio", [90, 10])
            ),
            "sentence_len": length,
            "splits": config.get("splits") or ["train", "dev"],
            "source": source,
            "amp_th": config.get("amp_th", 5e-4),
            "split_unit": split_unit
            if split_unit is not None
            else config.get("split_unit", "utterance"),
            "data_prep_seed": data_prep_seed
            if data_prep_seed is not None
            else config.get("data_prep_seed", 0),
        }
        for name, explicit, default in (
            ("skip_prep", skip_prep, config.get("skip_prep", False)),
            (
                "random_segment",
                random_segment,
                config.get("random_chunk", False),
            ),
            (
                "split_speaker",
                split_speaker,
                config.get("split_speaker", False),
            ),
        ):
            options[name] = _parse_bool(
                explicit if explicit is not None else default
            )
        if options["split_unit"] not in {"utterance", "session"}:
            raise ValueError("split_unit must be 'utterance' or 'session'")
        if options["split_unit"] == "session" and options["split_speaker"]:
            raise ValueError(
                "session splitting requires shared train/dev speaker identities"
            )

        from agent.runners.speechbrain_backend import run_data_prep

        prepared = validate_runner_result(
            run_data_prep(**options), "data_preparation"
        )
        result = OperationResult(
            status=prepared["status"],
            stage="data_preparation",
            task={
                "type": "speaker_verification",
                "dataset": options["data_folder"],
            },
            execution={
                "runner": "speechbrain",
                "output_folder": str(save_folder),
            },
            parameters=dict(options),
            error=prepared.get("error"),
            experiment_id=experiment_id,
        )
        if result.status == "success":
            save_folder = Path(prepared.get("save_folder") or save_folder)
            result.execution["output_folder"] = str(save_folder)
            result.parameters.update(
                save_folder=str(save_folder),
                verification_file=prepared.get("verification_local")
                or options["verification_file"],
            )
            names = list(
                dict.fromkeys(
                    [
                        *options["splits"],
                        *(["enrol"] if "test" in options["splits"] else []),
                    ]
                )
            )
            manifests = {name: save_folder / f"{name}.csv" for name in names}
            manifests = {
                name: path for name, path in manifests.items() if path.is_file()
            }
            result.metrics = {
                "summary": {
                    name: _count_csv_rows(path)
                    for name, path in manifests.items()
                }
            }
            result.artifacts = [
                Artifact("manifest", name, str(path))
                for name, path in manifests.items()
            ]
        result.extensions = {
            "data_preparation": {"prepare_status": result.status}
        }
    except Exception as exc:
        result = OperationResult(
            status="failed",
            stage="data_preparation",
            error=f"{type(exc).__name__}: {exc}",
            experiment_id=experiment_id,
        )
    if experiment_id and tracker is not None:
        tracker.update_experiment(
            experiment_id=experiment_id,
            experiment_type="data_processing",
            stage=result.stage,
            status=result.status,
            error=result.error,
            parameters=result.parameters,
            metrics=result.metrics,
            artifacts=[artifact.to_dict() for artifact in result.artifacts],
            extensions=result.extensions,
        )
    return result.to_json()


__all__ = ["PrepareVoxCelebData"]
