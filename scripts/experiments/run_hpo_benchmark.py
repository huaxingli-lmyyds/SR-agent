#!/usr/bin/env python3
"""Five-group, validation-only HPO benchmark; held-out evaluation is a separate command."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = (
    "default_parameters",
    "random_search",
    "tpe",
    "tpe_successive_halving",
    "system",
)
DEFAULT_CONFIG = ROOT / "configs/experiments/five_group_ecapa.json"
SCHEMA_VERSION = 2


def now():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(
            json.dumps(
                safe_json(value), ensure_ascii=False, indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_event(run_dir, name, event):
    path = Path(run_dir) / f"{name}.jsonl"
    if path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, 2)
            unterminated = stream.read(1) != b"\n"
        if unterminated:
            raw = path.read_bytes()
            prefix, separator, tail = raw.rpartition(b"\n")
            try:
                json.loads(tail)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Preserve the torn append for audit before repairing our own journal.
                path.with_name(
                    f"{path.name}.{uuid4().hex}.partial"
                ).write_bytes(tail)
                path.write_bytes(prefix + separator)
                event = {**event, "repaired_truncated_event": True}
            else:
                with path.open("ab") as stream:
                    stream.write(b"\n")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                safe_json({"timestamp": now(), **event}),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()


def read_events(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    events = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            # A hard kill may interrupt the final append; never discard earlier events.
    return events


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hash(path):
    path = Path(path)
    if path.is_file():
        return file_hash(path)
    if not path.is_dir():
        raise ValueError(f"checkpoint does not exist: {path}")
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"empty checkpoint directory: {path}")
    return hashlib.sha256(
        json.dumps(
            [(p.relative_to(path).as_posix(), file_hash(p)) for p in files]
        ).encode()
    ).hexdigest()


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_config(config):
    prep_seed = config.setdefault("data_prep_seed", 0)
    if type(prep_seed) is not int or not 0 <= prep_seed < 2**32 - 1:
        raise ValueError(
            "data_prep_seed must be a nonnegative integer below 2**32-1"
        )
    if (
        not finite(config.get("temperature", 0.0))
        or not 0 <= config.get("temperature", 0.0) <= 2
    ):
        raise ValueError("temperature must be finite and in [0,2]")
    for name in (
        "full_epochs",
        "full_trials",
        "sh_initial_trials",
        "candidate_batch_size",
        "n_startup_trials",
    ):
        positive_int(config[name], name)
    if (
        type(config.get("max_retries", 0)) is not int
        or config.get("max_retries", 0) < 0
    ):
        raise ValueError("max_retries must be a nonnegative integer")
    seeds = config["seeds"]
    if (
        not seeds
        or any(type(s) is not int or not 0 <= s < 2**32 - 1 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "seeds must be unique nonnegative integers below 2**32-1"
        )
    budgets, limits = config["budgets"], config["promotion_limits"]
    if len(budgets) < 2 or len(limits) != len(budgets) - 1:
        raise ValueError("promotion_limits must have one entry per transition")
    previous_count = config["sh_initial_trials"]
    for count in limits:
        positive_int(count, "promotion limit")
        if count > math.ceil(previous_count / 3):
            raise ValueError(
                "promotion limit exceeds reduction_factor=3 capacity"
            )
        previous_count = count
    previous = (0, 0.0)
    stages = set()
    for budget in budgets:
        if set(budget) - {
            "stage",
            "epochs",
            "data_fraction",
            "max_duration_seconds",
        }:
            raise ValueError("unknown TrialBudget field")
        positive_int(budget["epochs"], "epochs")
        fraction = budget["data_fraction"]
        if not finite(fraction) or not 0 < fraction <= 1:
            raise ValueError("data_fraction must be finite and in (0,1]")
        current = (budget["epochs"], fraction)
        if current[0] < previous[0] or current[1] < previous[1]:
            raise ValueError("budgets must be nondecreasing")
        previous = current
        if not budget.get("stage") or budget["stage"] in stages:
            raise ValueError("budget stages must be unique and nonempty")
        stages.add(budget["stage"])
        timeout = budget.get("max_duration_seconds")
        if timeout is not None and (not finite(timeout) or timeout <= 0):
            raise ValueError("max_duration_seconds must be positive and finite")
    if previous != (config["full_epochs"], 1.0):
        raise ValueError(
            "SH confirmation must equal full_epochs and data_fraction=1.0"
        )
    allowed = {
        "lr",
        "batch_size",
        "margin",
        "weight_decay",
        "sentence_len",
        "lr_final",
    }
    space = config["search_space"]
    params = space["parameters"]
    names = [p["name"] for p in params]
    if not names or len(set(names)) != len(names) or set(names) - allowed:
        raise ValueError(
            "search parameters must be unique supported training hyperparameters"
        )
    for p in params:
        if p.get("condition"):
            raise ValueError(
                "conditional search spaces are not supported by this benchmark"
            )
        kind = p["parameter_type"]
        if kind == "categorical":
            if not p.get("choices") or any(not finite(v) for v in p["choices"]):
                raise ValueError(
                    "categorical choices must be nonempty finite numbers"
                )
        elif kind in {"float", "int"}:
            if (
                not finite(p.get("low"))
                or not finite(p.get("high"))
                or p["low"] >= p["high"]
            ):
                raise ValueError(
                    "numeric search bounds must be finite and increasing"
                )
            if kind == "int" and any(
                type(p[k]) is not int for k in ("low", "high")
            ):
                raise ValueError("integer search bounds must be integers")
            if p.get("scale", "linear") not in {"linear", "log"} or (
                p.get("scale") == "log" and p["low"] <= 0
            ):
                raise ValueError("invalid search scale")
        else:
            raise ValueError(f"unsupported parameter_type: {kind}")


def read_pairs(path):
    rows, speakers, labels = [], set(), set()
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3 or parts[0] not in {"0", "1"}:
            raise ValueError(f"invalid verification pair at {path}:{number}")
        for index in (1, 2):
            item = parts[index].replace("\\", "/")
            if (
                item.startswith("/")
                or ":" in item
                or ".." in item.split("/")
                or len(item.split("/")) < 3
            ):
                raise ValueError(
                    f"unsafe/non-VoxCeleb pair path at {path}:{number}"
                )
            parts[index] = item
            speakers.add(item.split("/")[0])
        rows.append(" ".join(parts))
        labels.add(parts[0])
    if labels != {"0", "1"}:
        raise ValueError(
            f"pairs must contain positive and negative examples: {path}"
        )
    return rows, speakers


def variant_spec(config, name):
    full = {**config["budgets"][-1], "stage": "full"}
    is_sh = name in {"tpe_successive_halving", "system"}
    count = 1 if name == "default_parameters" else config["full_trials"]
    if is_sh:
        count = config["sh_initial_trials"]
    limits = config["promotion_limits"] if is_sh else []
    runs = count + sum(limits)
    budgets = config["budgets"] if is_sh else [full]
    work = sum(
        n * b["epochs"] * b["data_fraction"]
        for n, b in zip([count, *limits], budgets)
    )
    return {
        "name": name,
        "sampler": "grid_search"
        if name == "default_parameters"
        else ("random_search" if name == "random_search" else "tpe"),
        "pruner": "successive_halving" if is_sh else "none",
        "budgets": deepcopy(budgets),
        "initial_trial_count": count,
        "promotion_limits": list(limits),
        "max_training_runs": runs,
        "nominal_full_data_epochs": work,
        "candidate_batch_size": config["candidate_batch_size"] if is_sh else 1,
    }


def build_plan(args):
    config_path = Path(args.config or DEFAULT_CONFIG).resolve()
    config = read_json(config_path)
    if args.seeds:
        config["seeds"] = [int(s) for s in args.seeds.split(",")]
    validate_config(config)
    names = args.only.split(",") if args.only else list(VARIANTS)
    if len(set(names)) != len(names) or not names or set(names) - set(VARIANTS):
        raise ValueError(f"--only must select unique names from {VARIANTS}")
    inputs = {
        "train_config": str(project_path(config["train_config"])),
        "validation_config": str(project_path(config["validation_config"])),
    }
    for name in ("validation_pairs", "test_pairs"):
        value = getattr(args, name)
        if value:
            inputs[name] = str(Path(value).resolve())
        elif not args.dry_run:
            raise ValueError(
                f"--{name.replace('_', '-')} is required; no implicit official test set"
            )
    fingerprints = {name: file_hash(path) for name, path in inputs.items()}
    pair_counts = {}
    if "validation_pairs" in inputs and "test_pairs" in inputs:
        validation, validation_speakers = read_pairs(inputs["validation_pairs"])
        test, test_speakers = read_pairs(inputs["test_pairs"])
        overlap = validation_speakers & test_speakers
        if overlap:
            raise ValueError(
                f"validation/test speakers overlap ({len(overlap)} speakers)"
            )
        pair_counts = {
            "validation": len(validation),
            "test": len(test),
            "validation_speakers": len(validation_speakers),
            "test_speakers": len(test_speakers),
        }
    if not args.data_folder and not args.dry_run:
        raise ValueError("--data-folder is required")
    data_folder = (
        str(Path(args.data_folder).resolve()) if args.data_folder else None
    )
    if data_folder and not Path(data_folder).is_dir():
        raise ValueError(f"data folder not found: {data_folder}")
    order = []
    for seed in config["seeds"]:
        shuffled = list(names)
        random.Random(seed).shuffle(shuffled)
        order.extend(
            {"variant": name, "seed": seed, "run_id": f"{name}_seed_{seed}"}
            for name in shuffled
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now(),
        "config": config,
        "inputs": inputs,
        "input_sha256": fingerprints,
        "data_folder": data_folder,
        "pair_counts": pair_counts,
        "variants": [variant_spec(config, name) for name in names],
        "run_order": order,
        "objective": {
            "metric": "eer",
            "mode": "min",
            "selection_split": "validation",
        },
        "budget_mode": "fixed_training_quotas_not_equal_gpu_time",
        "warnings": [
            "Full-fidelity and SH groups have different work quotas; report actual cost, not an equal-GPU-time claim.",
            "This entry benchmarks the HPO subsystem on identical data, not LLM data cleaning or coordinator routing.",
            "Only the system group uses the LLM; active agent_proposal is disabled.",
        ],
    }


def snapshot_yaml(source, destination, overrides, *, isolate=False):
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import TaggedScalar

    yaml = YAML()
    yaml.preserve_quotes = True
    with Path(source).open(encoding="utf-8") as stream:
        value = yaml.load(stream)
    value.update(overrides)
    if isolate:
        # Keep late !ref binding: the runner supplies a different output per Trial.
        bindings = {
            "save_folder": "<output_folder>/save",
            "train_log": "<output_folder>/train_log.txt",
            "train_annotation": "<save_folder>/train.csv",
            "valid_annotation": "<save_folder>/dev.csv",
            "train_data": "<save_folder>/train.csv",
            "enrol_data": "<save_folder>/enrol.csv",
            "test_data": "<save_folder>/test.csv",
        }
        for key, reference in bindings.items():
            if key in value or key == "save_folder":
                value[key] = TaggedScalar(reference, tag="!ref")
        for key, field, reference in (
            ("checkpointer", "checkpoints_dir", "<save_folder>"),
            ("train_logger", "save_file", "<train_log>"),
            ("pretrainer", "collect_in", "<save_folder>"),
        ):
            if key in value:
                if not isinstance(value[key], dict):
                    raise ValueError(
                        f"cannot isolate opaque {key} configuration"
                    )
                value[key][field] = TaggedScalar(reference, tag="!ref")
        # Augmentation is prepared once, outside per-method timers, then read only.
        for kind in ("noise", "rir"):
            value.pop(f"prepare_{kind}_data", None)
    with Path(destination).open("w", encoding="utf-8") as stream:
        yaml.dump(value, stream)


def package_versions():
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in (
        "torch",
        "speechbrain",
        "optuna",
        "langgraph",
        "langchain-core",
        "hyperpyyaml",
        "ruamel.yaml",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return packages


def freeze_inputs(plan, output):
    import shutil
    import platform

    sources = [
        *sorted((ROOT / "agent").rglob("*.py")),
        *sorted((ROOT / "agent/prompt").glob("*.json")),
        *sorted((ROOT / "recipes").rglob("*.py")),
        Path(__file__),
        Path(__file__).with_name("benchmark_runtime.py"),
    ]
    plan["output_root"] = str(output.resolve())
    plan["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "source_sha256": {
            str(p.relative_to(ROOT)): file_hash(p) for p in sources
        },
    }
    folder = output / "inputs"
    folder.mkdir()
    frozen = {}
    for name, source in plan["inputs"].items():
        path = folder / (name + Path(source).suffix)
        shutil.copyfile(source, path)
        if file_hash(path) != plan["input_sha256"][name]:
            raise ValueError(f"input changed while snapshotting: {name}")
        frozen[name] = str(path)
    validation, _ = read_pairs(frozen["validation_pairs"])
    test, _ = read_pairs(frozen["test_pairs"])
    exclusion = folder / "training_exclusions.txt"
    exclusion.write_text("\n".join(validation + test) + "\n", encoding="utf-8")
    plan["frozen_inputs"] = frozen
    plan["training_exclusions"] = str(exclusion)
    plan["training_exclusions_sha256"] = file_hash(exclusion)


def verify_frozen(plan, output):
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "benchmark protocol version changed; use a new output directory"
        )
    if Path(plan["output_root"]).resolve() != output.resolve():
        raise ValueError(
            "benchmark output was moved/copied; resume only at its original output root"
        )
    if plan["environment"]["packages"] != package_versions():
        raise ValueError(
            "benchmark dependency versions changed; restore the original environment"
        )
    for relative, digest in plan["environment"]["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(
                f"benchmark source changed: {relative}; use original code or a new output directory"
            )
    for name, path in plan["frozen_inputs"].items():
        if file_hash(path) != plan["input_sha256"][name]:
            raise ValueError(f"frozen input was changed: {name}")
    if (
        file_hash(plan["training_exclusions"])
        != plan["training_exclusions_sha256"]
    ):
        raise ValueError("frozen training exclusions were changed")


def prepare_assets(plan, output):
    """Prepare common immutable augmentation inputs.

    Existing ``<data_folder>/noise`` and ``<data_folder>/rir`` audio is reused
    read-only. Only generated annotations and the manifest are written below
    the experiment output. Missing local assets retain the URL-download
    fallback declared by the training YAML.
    """
    from agent.utils import ConfigParser

    manifest = output / "assets.json"
    if manifest.exists():
        assets = read_json(manifest)
        for path, digest in assets["sha256"].items():
            if file_hash(path) != digest:
                raise ValueError(f"augmentation annotation changed: {path}")
        for kind, source in assets.get("sources", {}).items():
            current = augmentation_inventory_hash(
                Path(source["path"]), source["extension"]
            )
            if current != source["inventory_sha256"]:
                raise ValueError(
                    f"{kind} augmentation audio inventory changed: "
                    f"{source['path']}"
                )
        return assets["overrides"]
    if any((output / "runs").glob("*/run_inputs.json")):
        raise ValueError(
            "missing frozen assets manifest; refusing to rebuild inputs of existing runs"
        )
    started = perf_counter()
    raw = ConfigParser(plan["frozen_inputs"]["train_config"]).load_config(
        convert_to_dict=False
    )
    overrides = {}
    sources = {}
    for kind in ("noise", "rir"):
        key = f"prepare_{kind}_data"
        if key not in raw:
            if f"{kind}_annotation" in raw:
                raise ValueError(
                    f"{kind}_annotation requires a supported {key} declaration"
                )
            continue
        spec = raw[key]
        tag = getattr(spec, "tag", "")
        tag_value = getattr(tag, "value", None)
        actual_tag = tag_value if tag_value is not None else str(tag)
        expected_tag = (
            "!name:recipes.voxceleb.augmentation_prepare."
            "prepare_dataset_from_URL"
        )
        if actual_tag != expected_tag:
            raise ValueError(f"unsupported augmentation preparer: {key}")
        from agent.utils.config_parser import parse_yaml_tags
        from recipes.voxceleb.augmentation_prepare import (
            prepare_dataset_from_folder,
            prepare_dataset_from_URL,
        )

        destination = output / "assets" / kind
        annotation = destination / f"{kind}.csv"
        extension = str(spec.get("ext", "wav")).lstrip(".")
        max_length = parse_yaml_tags(spec.get("max_length"), raw)
        local_source = (
            Path(plan["data_folder"]) / kind
            if plan.get("data_folder")
            else None
        )
        if local_source is not None and any(
            local_source.rglob(f"*.{extension}")
        ):
            local_source = local_source.resolve()
            print(
                f"Using local {kind} augmentation audio: {local_source}",
                flush=True,
            )
            prepare_dataset_from_folder(
                source_folder=str(local_source),
                csv_file=str(annotation),
                ext=extension,
                max_length=max_length,
            )
            asset_folder = local_source
            mode = "local"
        else:
            print(
                f"No local {kind} audio found; preparing URL asset in "
                f"{destination}",
                flush=True,
            )
            prepare_dataset_from_URL(
                URL=str(parse_yaml_tags(spec["URL"], raw)),
                dest_folder=str(destination),
                csv_file=str(annotation),
                ext=extension,
                max_length=max_length,
            )
            asset_folder = destination.resolve()
            mode = "download"
        overrides[f"data_folder_{kind}"] = str(asset_folder)
        overrides[f"{kind}_annotation"] = str(annotation)
        sources[kind] = {
            "mode": mode,
            "path": str(asset_folder),
            "extension": extension,
            "inventory_sha256": augmentation_inventory_hash(
                asset_folder, extension
            ),
        }
    write_json(
        manifest,
        {
            "overrides": overrides,
            "sources": sources,
            "sha256": {
                v: file_hash(v)
                for k, v in overrides.items()
                if k.endswith("_annotation")
            },
            "setup_seconds": perf_counter() - started,
        },
    )
    return overrides


def augmentation_inventory_hash(folder, extension):
    """Hash augmentation paths and sizes without rereading all audio bytes."""
    folder = Path(folder)
    files = sorted(folder.rglob(f"*.{str(extension).lstrip('.')}"))
    if not files:
        raise ValueError(f"augmentation audio folder is empty: {folder}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(folder).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_run(plan, item, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    config, seed = plan["config"], item["seed"]
    metadata_path = run_dir / "run_inputs.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        for path, digest in metadata["sha256"].items():
            if file_hash(path) != digest:
                raise ValueError(f"run snapshot changed: {path}")
        return metadata
    if (run_dir / "hpo").exists():
        raise ValueError(
            "existing HPO state has no run_inputs.json; refusing to regenerate its configuration"
        )
    paths = {
        "train": run_dir / "train.yaml",
        "validation": run_dir / "validation.yaml",
        "test": run_dir / "test.yaml",
    }
    common = {
        "seed": seed,
        "data_folder": plan["data_folder"],
        "data_prep_seed": config["data_prep_seed"],
        "prep_cache_root": str(run_dir / "prep_cache"),
        "voxceleb_source": None,
        "skip_prep": False,
    }
    snapshot_yaml(
        plan["frozen_inputs"]["train_config"],
        paths["train"],
        {
            **common,
            **read_json(Path(plan["output_root"]) / "assets.json")["overrides"],
            "verification_file": plan["training_exclusions"],
            "number_of_epochs": config["full_epochs"],
            "output_folder": str(run_dir / "unused_training_output"),
        },
        isolate=True,
    )
    for split, pairs in (
        ("validation", "validation_pairs"),
        ("test", "test_pairs"),
    ):
        snapshot_yaml(
            plan["frozen_inputs"]["validation_config"],
            paths[split],
            {
                **common,
                "prep_cache_root": str(run_dir / "prep_cache" / split),
                "verification_file": plan["frozen_inputs"][pairs],
                "output_folder": str(run_dir / f"unused_{split}_output"),
            },
            isolate=True,
        )
    metadata = {
        "paths": {k: str(v) for k, v in paths.items()},
        "training_seed": seed,
        "hpo_seed": seed,
        "sha256": {str(v): file_hash(v) for v in paths.values()},
    }
    write_json(metadata_path, metadata)
    return metadata


def run_variant(plan, item, output):
    from agent.agents.communication import AgentTaskRequest
    from agent.hpo import HPOService
    from agent.memory import MemoryService
    from agent.utils import ConfigParser, ExperimentTracker
    from scripts.experiments.benchmark_runtime import BenchmarkHPOAgent

    run_dir = output / "runs" / item["run_id"]
    metadata = prepare_run(plan, item, run_dir)
    config = plan["config"]
    spec = next(v for v in plan["variants"] if v["name"] == item["variant"])
    space = deepcopy(config["search_space"])
    if item["variant"] == "default_parameters":
        defaults = ConfigParser(metadata["paths"]["train"]).load_config()
        space = {
            "parameters": [
                {
                    "name": p["name"],
                    "parameter_type": "categorical",
                    "choices": [defaults[p["name"]]],
                }
                for p in space["parameters"]
            ],
            "constraints": [],
        }
    request = AgentTaskRequest(
        action="optimize_hyperparameters",
        objective="Minimize validation EER within the frozen benchmark protocol.",
        context={
            "dataset_uri": plan["data_folder"],
            "data_folder": plan["data_folder"],
            "hpo_seed": item["seed"],
            "strategy": spec["sampler"],
            "sampler": spec["sampler"],
            "pruner": spec["pruner"],
            "search_space": space,
            "budgets": spec["budgets"],
            "primary_metric": "eer",
            "metric_mode": "min",
            "controller_mode": "llm" if item["variant"] == "system" else "fixed",
            "sampler_config": {
                "n_startup_trials": config["n_startup_trials"],
                "multivariate": False,
            }
            if spec["sampler"] == "tpe"
            else {},
            "runtime_options": {
                k: config.get(k)
                for k in (
                    "device",
                    "ddp_devices",
                    "distributed_backend",
                    "batch_size_semantics",
                    "precision",
                    "eval_precision",
                )
                if config.get(k) is not None
            },
        },
        budget={
            "max_training_runs": spec["max_training_runs"],
            "max_total_training_runs": spec["max_training_runs"],
            "max_studies": 1,
            "initial_trial_count": spec["initial_trial_count"],
            "promotion_limits": spec["promotion_limits"],
            "candidate_batch_size": spec["candidate_batch_size"],
            "strategy_review_interval_trials": spec["candidate_batch_size"],
            "reduction_factor": 3,
            "max_retries": config.get("max_retries", 0),
        },
    )
    request.context["runtime_options"]["verification_config"] = metadata[
        "paths"
    ]["validation"]
    request.context["runtime_options"]["validation_pairs"] = plan[
        "frozen_inputs"
    ]["validation_pairs"]
    request.context["runtime_options"]["training_exclusion_pairs"] = plan[
        "training_exclusions"
    ]
    store = run_dir / "hpo"
    records = sorted(store.glob("*/experiment_record.json"))
    if len(records) > 1:
        raise ValueError(f"expected at most one Study in {run_dir}")
    if records:
        request.context["resume_experiment_id"] = read_json(records[0])[
            "experiment_id"
        ]
    agent = BenchmarkHPOAgent(
        variant=item["variant"],
        envelope=config["search_space"],
        event_sink=lambda name, event: append_event(run_dir, name, event),
        model_name=config.get("model_name", "GLM-4.7"),
        temperature=config.get("temperature", 0.0),
        max_iterations=spec["max_training_runs"],
        verbose=False,
        config_path=metadata["paths"]["train"],
        experiments_dir=str(store),
        model_family=config["model_family"],
        runner=config.get("runner", "speechbrain"),
        implementation=config.get("implementation", "speechbrain"),
        memory_service=MemoryService(run_dir / "memory"),
    )
    append_event(run_dir, "sessions", {"event": "started"})
    started = perf_counter()
    try:
        result = agent.execute_task(request).to_dict()
    finally:
        append_event(
            run_dir,
            "sessions",
            {"event": "finished", "duration_seconds": perf_counter() - started},
        )
    write_json(run_dir / "agent_result.json", result)
    records = sorted(store.glob("*/experiment_record.json"))
    summary = {
        **item,
        "status": "failed",
        "error": result.get("error"),
        "validation_eer": None,
        "validation_min_dcf": None,
        "confirmation_satisfied": False,
        "training_runs": 0,
        "spec": spec,
        "run_dir": str(run_dir),
    }
    if records:
        record = read_json(records[0])
        summary["experiment_id"] = record["experiment_id"]
        service = HPOService(ExperimentTracker(store))
        study = service.load_study(record["experiment_id"])
        trials = service.list_trials(record["experiment_id"])
        summary["training_runs"] = service.training_runs_used(study)
        summary["study_status"] = study.status
        summary["strategy_reviews"] = len(study.strategy_reviews)
        summary["search_phases"] = study.search_phases
        if (
            [b.to_dict() for b in study.budgets]
            != [{"max_duration_seconds": None, **b} for b in spec["budgets"]]
            or study.pruner_strategy != spec["pruner"]
            or summary["training_runs"] > spec["max_training_runs"]
        ):
            raise ValueError(
                "benchmark resource protocol was changed or exceeded"
            )
        write_json(run_dir / "trials.json", [t.to_dict() for t in trials])
        highest = len(spec["budgets"]) - 1
        expected = spec["budgets"][-1]
        confirmed = [
            t
            for t in trials
            if t.rung == highest
            and t.status in {"completed", "promoted"}
            and finite(t.metrics.get("eer"))
            and t.budget.epochs == expected["epochs"]
            and t.budget.data_fraction == expected["data_fraction"]
        ]
        if (
            confirmed
            and result["status"] == "success"
            and study.status == "completed"
        ):
            winner = min(
                confirmed, key=lambda t: (t.metrics["eer"], t.trial_id)
            )
            checkpoint = winner.cost.get("evaluated_checkpoint")
            if not checkpoint:
                raise ValueError(
                    "winner has no checkpoint explicitly bound to its evaluation"
                )
            locked = {
                "trial_id": winner.trial_id,
                "parameters": winner.parameters,
                "budget": winner.budget.to_dict(),
                "validation_metrics": winner.metrics,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_hash(checkpoint),
                "locked_at": now(),
            }
            write_json(run_dir / "locked_winner.json", locked)
            summary.update(
                {
                    "status": "success",
                    "error": None,
                    "confirmation_satisfied": True,
                    "validation_eer": winner.metrics["eer"],
                    "validation_min_dcf": winner.metrics.get("min_dcf"),
                }
            )
        elif not summary["error"]:
            summary["error"] = "no valid full-budget confirmation"
        if item["variant"] != "system":
            if (
                any(
                    p["sampler"] != spec["sampler"] for p in study.search_phases
                )
                or len(study.search_phases) != 1
                or study.search_space.to_dict()
                != agent._resolve_search_space(space).to_dict()
            ):
                summary.update(
                    status="failed",
                    error="fixed baseline unexpectedly changed strategy or search space",
                    validation_eer=None,
                    validation_min_dcf=None,
                    confirmation_satisfied=False,
                )
    attempts = [
        e
        for e in read_events(run_dir / "attempts.jsonl")
        if e.get("event") == "finished"
    ]
    summary["attempt_duration_seconds"] = sum(
        e["duration_seconds"] for e in attempts
    )
    for stage in ("training", "evaluation"):
        summary[f"{stage}_seconds"] = sum(
            float(
                e.get("attempt_cost", {}).get(f"attempt_{stage}_seconds") or 0
            )
            for e in attempts
        )
    sessions = read_events(run_dir / "sessions.jsonl")
    summary["wall_seconds"] = sum(
        e.get("duration_seconds", 0) for e in sessions
    )
    summary["cost_may_be_incomplete"] = (
        sum(e.get("event") == "started" for e in sessions)
        != sum(e.get("event") == "finished" for e in sessions)
        or sum(
            e.get("event") == "started"
            for e in read_events(run_dir / "attempts.jsonl")
        )
        != len(attempts)
        or any("attempt_cost" not in e for e in attempts)
        or any(run_dir.glob("*.partial"))
    )
    advisor = read_events(run_dir / "advisor.jsonl")
    summary["advisor_requests"] = len(advisor)
    summary["advisor_seconds"] = sum(
        e.get("duration_seconds", 0) for e in advisor
    )
    summary["advisor_failures"] = sum(
        e.get("status") != "success" for e in advisor
    )
    proposals = read_events(run_dir / "proposals.jsonl")
    summary["llm_valid_proposals"] = sum(
        (e.get("raw") or {}).get("action") in agent.proposal_actions
        for e in proposals
    )
    summary["controller_degraded"] = (
        item["variant"] == "system" and summary["llm_valid_proposals"] == 0
    )
    write_json(run_dir / "result.json", summary)
    return summary


def summarize(results):
    grouped = {}
    for name in VARIANTS:
        rows = [r for r in results if r["variant"] == name]
        if not rows:
            continue
        valid = [
            r
            for r in rows
            if r["status"] == "success"
            and r.get("confirmation_satisfied")
            and finite(r.get("validation_eer"))
        ]
        values = [r["validation_eer"] for r in valid]
        grouped[name] = {
            "runs": len(rows),
            "successes": len(valid),
            "failures": len(rows) - len(valid),
            "validation_eer_values": values,
            "validation_eer_mean": statistics.mean(values) if values else None,
            "validation_eer_std": statistics.stdev(values)
            if len(values) > 1
            else None,
            "wall_seconds": sum(r.get("wall_seconds", 0) for r in rows),
            "degraded_controller_runs": sum(
                bool(r.get("controller_degraded")) for r in rows
            ),
            "incomplete_cost_runs": sum(
                bool(r.get("cost_may_be_incomplete")) for r in rows
            ),
        }
    paired = {}
    ours = {
        r["seed"]: r
        for r in results
        if r["variant"] == "system"
        and r["status"] == "success"
        and r.get("confirmation_satisfied")
        and finite(r.get("validation_eer"))
    }
    for name in grouped:
        if name == "system":
            continue
        differences = [
            {
                "seed": r["seed"],
                "system_minus_baseline": ours[r["seed"]]["validation_eer"]
                - r["validation_eer"],
            }
            for r in results
            if r["variant"] == name
            and r["seed"] in ours
            and r["status"] == "success"
            and r.get("confirmation_satisfied")
            and finite(r.get("validation_eer"))
        ]
        paired[name] = differences
    return {
        "by_variant": grouped,
        "paired_validation_differences": paired,
        "note": "Validation-only, complete-pair analysis; failures and incomplete costs remain in by_variant. Not an equal-GPU-time comparison.",
    }


def export_results(plan, output):
    results = [
        read_json(output / "runs" / item["run_id"] / "result.json")
        for item in plan["run_order"]
        if (output / "runs" / item["run_id"] / "result.json").exists()
    ]
    write_json(output / "results.json", results)
    write_json(output / "summary.json", summarize(results))
    fields = [
        "variant",
        "seed",
        "status",
        "validation_eer",
        "validation_min_dcf",
        "training_runs",
        "training_seconds",
        "evaluation_seconds",
        "wall_seconds",
        "advisor_requests",
        "advisor_seconds",
        "llm_valid_proposals",
        "controller_degraded",
        "confirmation_satisfied",
        "cost_may_be_incomplete",
        "error",
    ]
    with (output / "results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(results)
    return results


def evaluate_test(plan, output):
    from agent.tools.evaluation_tools import RunEvaluation
    from agent.utils import ExperimentTracker

    results = export_results(plan, output)
    if len(results) != len(plan["run_order"]):
        raise ValueError(
            "all planned search runs must finish before held-out evaluation"
        )
    heldout = []
    for row in results:
        if row["status"] != "success":
            heldout.append(
                {
                    "variant": row["variant"],
                    "seed": row["seed"],
                    "status": "search_failed",
                    "eer": None,
                }
            )
            continue
        run_dir = output / "runs" / row["run_id"]
        winner = read_json(run_dir / "locked_winner.json")
        if checkpoint_hash(winner["checkpoint"]) != winner["checkpoint_sha256"]:
            raise ValueError(f"winner checkpoint changed: {row['run_id']}")
        destination = output / "heldout" / row["run_id"]
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "result.json"
        if path.exists():
            heldout.append(read_json(path))
            continue
        inputs = read_json(run_dir / "run_inputs.json")
        for snapshot, digest in inputs["sha256"].items():
            if file_hash(snapshot) != digest:
                raise ValueError(f"run snapshot changed: {snapshot}")
        config = plan["config"]
        tracker = ExperimentTracker(destination / "records")
        eid = tracker.create_hpo_experiment(
            config_path=inputs["paths"]["train"],
            data_folder=plan["data_folder"],
            task={
                "type": "speaker_verification",
                "dataset": plan["data_folder"],
            },
        )
        start = perf_counter()
        payload = json.loads(
            RunEvaluation.invoke(
                {
                    "model_path": winner["checkpoint"],
                    "verification_config": inputs["paths"]["test"],
                    "verification_pairs": plan["frozen_inputs"]["test_pairs"],
                    "evaluation_split": "test",
                    "experiment_id": eid,
                    "experiments_dir": str(tracker.experiments_dir),
                    "data_folder": plan["data_folder"],
                    "task_type": "speaker_verification",
                    "model_family": config["model_family"],
                    "implementation": config.get(
                        "implementation", "speechbrain"
                    ),
                    "runner": config.get("runner", "speechbrain"),
                    **{
                        k: config.get(k)
                        for k in ("device", "precision", "eval_precision")
                    },
                }
            )
        )
        metrics = {}
        for values in (payload.get("metrics") or {}).values():
            metrics.update(values)
        result = {
            "variant": row["variant"],
            "seed": row["seed"],
            "status": payload["status"],
            "eer": metrics.get("eer"),
            "min_dcf": metrics.get("min_dcf"),
            "error": payload.get("error"),
            "duration_seconds": perf_counter() - start,
            "checkpoint_sha256": winner["checkpoint_sha256"],
        }
        if result["status"] == "success" and not finite(result["eer"]):
            result.update(status="failed", error="invalid held-out EER")
        write_json(path, result)
        heldout.append(result)
    write_json(output / "heldout_results.json", heldout)
    fields = [
        "variant",
        "seed",
        "status",
        "eer",
        "min_dcf",
        "duration_seconds",
        "error",
    ]
    with (output / "heldout_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(heldout)
    grouped = {}
    for name in {r["variant"] for r in heldout}:
        rows = [r for r in heldout if r["variant"] == name]
        values = [
            r["eer"]
            for r in rows
            if r["status"] == "success" and finite(r.get("eer"))
        ]
        grouped[name] = {
            "runs": len(rows),
            "successes": len(values),
            "eer_values": values,
            "eer_mean": statistics.mean(values) if values else None,
            "eer_std": statistics.stdev(values) if len(values) > 1 else None,
        }
    write_json(output / "heldout_summary.json", grouped)
    return heldout


@contextmanager
def exclusive_run(output):
    """OS-owned lock is released on a crash; no stale lock-file deletion needed."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".benchmark.lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OSError(
                f"another benchmark holds the execution lock: {output}"
            ) from exc
        try:
            yield
        finally:
            if sys.platform == "win32":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config",
        help="JSON experiment definition; defaults to five_group_ecapa.json",
    )
    result.add_argument("--data-folder")
    result.add_argument("--validation-pairs")
    result.add_argument("--test-pairs")
    result.add_argument("--only", help="Comma-separated group names")
    result.add_argument(
        "--seeds", help="Comma-separated distinct integer seeds"
    )
    result.add_argument("--output-dir")
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan only; no GPU, LLM or output writes",
    )
    modes = result.add_mutually_exclusive_group()
    modes.add_argument(
        "--resume",
        action="store_true",
        help="Use frozen plan; skip committed runs, resume interrupted Study",
    )
    modes.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate locked winners only; never feed test metrics to HPO",
    )
    return result


def preflight(plan):
    """Fail once for missing dependencies/configuration, not once per expensive run."""
    from agent.core.adapters import resolve_adapter_bundle
    from agent.hpo import HPOService
    from agent.models import get_model_adapter
    from agent.utils import ConfigParser
    from scripts.experiments.benchmark_runtime import BenchmarkHPOAgent  # noqa: F401

    cfg = plan["config"]
    resolve_adapter_bundle(
        "speaker_verification",
        cfg["model_family"],
        cfg.get("implementation", "speechbrain"),
        cfg.get("runner", "speechbrain"),
    )
    available = HPOService.available_samplers()
    if any(v["sampler"] not in available for v in plan["variants"]):
        raise ValueError(
            f"required sampler unavailable; available: {available}"
        )
    adapter = get_model_adapter(cfg["model_family"])
    adapter.validate_config(
        ConfigParser(plan["frozen_inputs"]["train_config"]).load_config()
    )
    if cfg.get("runner", "speechbrain") == "speechbrain":
        validate_speechbrain_inputs(plan)
        import torch

        if (
            str(cfg.get("device", "")).startswith("cuda")
            and not torch.cuda.is_available()
        ):
            raise ValueError(
                "CUDA requested but unavailable; select a working training environment"
            )
        if cfg.get("ddp_devices"):
            from agent.runners.speechbrain_distributed import resolve_ddp_plan

            resolve_ddp_plan(torch, {
                key: cfg.get(key)
                for key in (
                    "device",
                    "ddp_devices",
                    "distributed_backend",
                    "batch_size_semantics",
                )
                if cfg.get(key) is not None
            })


def validate_speechbrain_inputs(plan):
    from agent.utils import ConfigParser

    train_parser = ConfigParser(plan["frozen_inputs"]["train_config"])
    eval_parser = ConfigParser(plan["frozen_inputs"]["validation_config"])
    train, evaluation = train_parser.load_config(), eval_parser.load_config()
    if str(evaluation.get("score_norm", "none")).lower() != "none":
        raise ValueError(
            "benchmark requires score_norm=none; a separate frozen cohort protocol is not implemented"
        )
    for key in ("n_mels", "sample_rate"):
        if key in train and key in evaluation and train[key] != evaluation[key]:
            raise ValueError(f"train/evaluation feature mismatch: {key}")
    raw_train = train_parser.load_config(convert_to_dict=False)
    raw_eval = eval_parser.load_config(convert_to_dict=False)
    if not all(
        isinstance(v.get("embedding_model"), dict)
        for v in (raw_train, raw_eval)
    ):
        raise ValueError("benchmark requires explicit embedding_model mappings")
    if str(raw_train["embedding_model"].tag) != str(
        raw_eval["embedding_model"].tag
    ):
        raise ValueError("train/evaluation embedding model classes differ")
    for key in (
        train["embedding_model"].keys() & evaluation["embedding_model"].keys()
    ):
        if train["embedding_model"][key] != evaluation["embedding_model"][key]:
            raise ValueError(
                f"train/evaluation embedding model mismatch: {key}"
            )
    for split in ("validation_pairs", "test_pairs"):
        pairs, _ = read_pairs(plan["frozen_inputs"][split])
        for line in pairs:
            for utterance in line.split()[1:]:
                path = Path(plan["data_folder"]) / "wav" / utterance
                if not path.is_file():
                    raise ValueError(f"missing {split} audio: {path}")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.resume or args.evaluate_test:
        if args.dry_run or any(
            getattr(args, k)
            for k in (
                "config",
                "data_folder",
                "validation_pairs",
                "test_pairs",
                "only",
                "seeds",
            )
        ):
            raise ValueError(
                "resume/test mode accepts only --output-dir and the mode flag; inputs are frozen"
            )
    if args.dry_run:
        print(json.dumps(build_plan(args), ensure_ascii=False, indent=2))
        return 0
    if not args.output_dir:
        raise ValueError("--output-dir is required for execution")
    output = Path(args.output_dir).resolve()
    with ExitStack() as stack:
        stack.enter_context(exclusive_run(output))
        path = output / "benchmark_plan.json"
        if args.resume or args.evaluate_test:
            plan = read_json(path)
            verify_frozen(plan, output)
        else:
            if (
                path.exists()
                or (output / "inputs").exists()
                or (output / "runs").exists()
            ):
                raise ValueError(
                    "output already contains a benchmark; use --resume or a new directory"
                )
            plan = build_plan(args)
            freeze_inputs(plan, output)
            write_json(path, plan)
        if plan["config"].get("runner", "speechbrain") == "speechbrain":
            # Conservative process-wide exclusion, including different output roots.
            # This cannot block unrelated GPU jobs launched outside this entry point.
            stack.enter_context(
                exclusive_run(
                    Path(tempfile.gettempdir())
                    / "sr-agent-speechbrain-benchmark"
                )
            )
        if args.evaluate_test:
            prepare_assets(plan, output)
            rows = evaluate_test(plan, output)
            print(f"Held-out results: {output / 'heldout_results.json'}")
            return int(any(r["status"] != "success" for r in rows))
        preflight(plan)
        prepare_assets(plan, output)
        for item in plan["run_order"]:
            run_dir = output / "runs" / item["run_id"]
            if (run_dir / "result.json").exists():
                if (run_dir / "run_inputs.json").exists():
                    prepare_run(plan, item, run_dir)
                continue
            print(f"Running {item['run_id']}", flush=True)
            try:
                run_variant(plan, item, output)
            except KeyboardInterrupt:
                export_results(plan, output)
                print(
                    "Interrupted. Re-run with --resume --output-dir to continue the current Study.",
                    file=sys.stderr,
                )
                return 130
            except Exception as exc:
                run_dir.mkdir(parents=True, exist_ok=True)
                write_json(
                    run_dir / "result.json",
                    {
                        **item,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "validation_eer": None,
                        "confirmation_satisfied": False,
                        "cost_may_be_incomplete": True,
                    },
                )
            export_results(plan, output)
        results = export_results(plan, output)
        print(f"Results: {output / 'results.csv'}")
        return int(any(r["status"] != "success" for r in results))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        raise SystemExit(2)
