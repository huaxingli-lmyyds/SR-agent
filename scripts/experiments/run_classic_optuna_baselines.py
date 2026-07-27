#!/usr/bin/env python3
"""Run classic Optuna HPO baselines outside the agentic optimization system.

This script is intentionally separate from the SR-agent HPO service and the
CoordinatorAgent workflow.  It reuses only the existing task/model/runner
adapters to execute training, while Optuna owns all candidate generation and
multi-fidelity pruning decisions.

Implemented baselines:

- random_search: RandomSampler + no pruner.
- tpe: TPESampler + no pruner.
- tpe_successive_halving: TPESampler + SuccessiveHalvingPruner.
- bohb: BOHB-style baseline implemented as TPESampler + HyperbandPruner.

The BOHB label is "style" rather than a byte-for-byte hpbandster BOHB
implementation because Optuna does not expose a dedicated BOHB sampler.  The
combination of model-based sampling and Hyperband resource allocation is the
closest Optuna-native counterpart and is sufficient for a fair non-agentic
comparison when documented clearly.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from uuid import uuid4

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from agent.core.adapters import resolve_adapter_bundle
from agent.runners import collect_training_result
from agent.utils import ConfigParser, resolve_config_path, resolve_data_path
from agent.utils.path_tool import get_experiments_dir
from scripts.experiments.run_comparison_experiments import (
    default_search_space,
    model_defaults,
)


SINGLE_FULL_BUDGET = [
    {"stage": "full", "epochs": 24, "data_fraction": 1.0, "max_duration_seconds": None}
]

MULTI_FIDELITY_BUDGETS = [
    {
        "stage": "screen",
        "epochs": 3,
        "data_fraction": 0.25,
        "max_duration_seconds": None,
    },
    {
        "stage": "promote",
        "epochs": 8,
        "data_fraction": 0.5,
        "max_duration_seconds": None,
    },
    {"stage": "full", "epochs": 24, "data_fraction": 1.0, "max_duration_seconds": None},
]

SUPPORTED_MODEL_FAMILIES = ("ecapa_tdnn", "resnet", "xvector")
SUPPORTED_PARAMETER_TYPES = {"categorical", "int", "float"}
SUPPORTED_CONSTRAINT_OPERATORS = {"lte", "gte", "eq", "in"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON without leaving a partially-written result after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass
class ClassicBaselineVariant:
    name: str
    sampler: str
    pruner: str = "none"
    n_trials: int = 8
    budgets: list[dict[str, Any]] = field(
        default_factory=lambda: deepcopy(SINGLE_FULL_BUDGET)
    )


@dataclass
class TrialRunRecord:
    model_family: str
    variant: str
    optuna_trial_number: int
    stage_index: int
    stage: str
    status: str
    value: float | None
    parameters: dict[str, Any]
    budget: dict[str, Any]
    output_folder: str | None
    model_paths: list[str]
    train_log_path: str | None
    objective_metric_name: str | None
    started_at: str
    finished_at: str
    metrics: dict[str, Any]
    duration_seconds: float
    error: str | None = None


def variants_for_suite(suite: str) -> list[ClassicBaselineVariant]:
    if suite == "classic":
        return [
            ClassicBaselineVariant("random_search", "random", "none", n_trials=8),
            ClassicBaselineVariant("tpe", "tpe", "none", n_trials=8),
            ClassicBaselineVariant(
                "tpe_successive_halving",
                "tpe",
                "successive_halving",
                n_trials=8,
                budgets=deepcopy(MULTI_FIDELITY_BUDGETS),
            ),
            ClassicBaselineVariant(
                "bohb",
                "tpe",
                "hyperband",
                n_trials=8,
                budgets=deepcopy(MULTI_FIDELITY_BUDGETS),
            ),
        ]
    if suite == "smoke":
        smoke_budget = [
            {
                "stage": "smoke",
                "epochs": 1,
                "data_fraction": 0.05,
                "max_duration_seconds": 1800.0,
            }
        ]
        return [
            ClassicBaselineVariant(
                "random_search",
                "random",
                "none",
                n_trials=2,
                budgets=deepcopy(smoke_budget),
            ),
            ClassicBaselineVariant(
                "tpe", "tpe", "none", n_trials=2, budgets=deepcopy(smoke_budget)
            ),
            ClassicBaselineVariant(
                "tpe_successive_halving",
                "tpe",
                "successive_halving",
                n_trials=2,
                budgets=deepcopy(smoke_budget),
            ),
            ClassicBaselineVariant(
                "bohb", "tpe", "hyperband", n_trials=2, budgets=deepcopy(smoke_budget)
            ),
        ]
    raise ValueError(f"unsupported classic baseline suite: {suite}")


def load_json_arg(value: str | None, *, expected_type: type, label: str) -> Any:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(parsed, expected_type):
        raise argparse.ArgumentTypeError(
            f"{label} must be a JSON {expected_type.__name__}"
        )
    return parsed


def selected_variants(args: argparse.Namespace) -> list[ClassicBaselineVariant]:
    variants = variants_for_suite(args.suite)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        variants = [item for item in variants if item.name in wanted]
        missing = sorted(wanted - {item.name for item in variants})
        if missing:
            raise ValueError(
                f"unknown variant(s) for suite {args.suite}: {', '.join(missing)}"
            )

    budgets = load_json_arg(args.budgets_json, expected_type=list, label="budgets")
    for variant in variants:
        if args.n_trials is not None:
            variant.n_trials = args.n_trials
        if budgets is not None:
            variant.budgets = deepcopy(budgets)
    return variants


def validate_budgets(budgets: list[dict[str, Any]]) -> None:
    if not budgets:
        raise ValueError("at least one budget stage is required")
    previous_epochs = 0
    previous_fraction = 0.0
    stage_names: set[str] = set()
    for index, budget in enumerate(budgets, start=1):
        stage = str(budget.get("stage") or f"stage_{index}")
        if stage in stage_names:
            raise ValueError(f"duplicate budget stage: {stage}")
        stage_names.add(stage)
        epochs = budget.get("epochs")
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
            raise ValueError(f"budget stage {stage} must have positive integer epochs")
        fraction = budget.get("data_fraction", 1.0)
        if (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not 0 < float(fraction) <= 1
        ):
            raise ValueError(f"budget stage {stage} data_fraction must be in (0, 1]")
        timeout = budget.get("max_duration_seconds")
        if timeout is not None and (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or float(timeout) <= 0
        ):
            raise ValueError(
                f"budget stage {stage} max_duration_seconds must be positive"
            )
        if epochs < previous_epochs or float(fraction) < previous_fraction:
            raise ValueError(
                "budget stages must use non-decreasing epochs and data_fraction"
            )
        previous_epochs = epochs
        previous_fraction = float(fraction)


def validate_search_space(search_space: dict[str, Any]) -> None:
    parameters = search_space.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        raise ValueError("search space must contain a non-empty parameters list")
    names: set[str] = set()
    for parameter in parameters:
        if not isinstance(parameter, dict):
            raise ValueError("each search-space parameter must be an object")
        name = parameter.get("name")
        parameter_type = parameter.get("parameter_type")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(
                f"invalid or duplicate search-space parameter name: {name!r}"
            )
        if name in {
            "number_of_epochs",
            "data_folder",
            "output_folder",
        } or name.startswith("_hpo_"):
            raise ValueError(
                f"search-space parameter is reserved by the experiment runner: {name}"
            )
        if parameter_type not in SUPPORTED_PARAMETER_TYPES:
            raise ValueError(f"unsupported parameter type for {name}: {parameter_type}")
        names.add(name)
        if parameter_type == "categorical":
            choices = parameter.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"categorical parameter {name} must have non-empty choices"
                )
        else:
            if parameter.get("scale") not in (None, "linear", "log"):
                raise ValueError(
                    f"unsupported scale for {name}: {parameter.get('scale')}"
                )
            low, high = parameter.get("low"), parameter.get("high")
            if (
                not isinstance(low, (int, float))
                or isinstance(low, bool)
                or not isinstance(high, (int, float))
                or isinstance(high, bool)
                or low > high
                or (
                    parameter_type == "int"
                    and (not isinstance(low, int) or not isinstance(high, int))
                )
            ):
                raise ValueError(
                    f"numeric parameter {name} must have valid low/high bounds"
                )
            if parameter.get("scale") == "log" and low <= 0:
                raise ValueError(f"log-scaled parameter {name} must have low > 0")
    for constraint in search_space.get("constraints") or []:
        if not isinstance(constraint, dict):
            raise ValueError("each search-space constraint must be an object")
        if constraint.get("parameter") not in names:
            raise ValueError(
                f"constraint references unknown parameter: {constraint.get('parameter')}"
            )
        if constraint.get("operator") not in SUPPORTED_CONSTRAINT_OPERATORS:
            raise ValueError(
                f"unsupported constraint operator: {constraint.get('operator')}"
            )
        if constraint.get("operator") == "in" and not isinstance(
            constraint.get("value"), list
        ):
            raise ValueError("constraint operator 'in' requires a list value")


def validate_experiment_setup(
    args: argparse.Namespace,
    variants: list[ClassicBaselineVariant],
    search_space: dict[str, Any],
    config_data: dict[str, Any],
) -> Any:
    if args.model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"unsupported model family: {args.model_family}")
    validate_search_space(search_space)
    for variant in variants:
        validate_budgets(variant.budgets)
    adapter_bundle = resolve_adapter_bundle(
        args.task_type, args.model_family, args.implementation, args.runner
    )
    adapter_bundle.model.validate_config(config_data)
    for parameter in search_space["parameters"]:
        sample = (
            parameter["choices"][0]
            if parameter["parameter_type"] == "categorical"
            else parameter["low"]
        )
        adapter_bundle.model.validate_parameters({parameter["name"]: sample})
    return adapter_bundle


def make_sampler_and_pruner(
    variant: ClassicBaselineVariant, *, seed: int
) -> tuple[Any, Any]:
    import optuna

    if variant.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif variant.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=min(5, max(1, variant.n_trials // 2)),
            multivariate=True,
            group=True,
        )
    else:
        raise ValueError(f"unsupported sampler: {variant.sampler}")

    if variant.pruner == "none":
        pruner = optuna.pruners.NopPruner()
    elif variant.pruner == "successive_halving":
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate=0,
        )
    elif variant.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=max(1, len(variant.budgets)),
            reduction_factor=3,
        )
    else:
        raise ValueError(f"unsupported pruner: {variant.pruner}")
    return sampler, pruner


def suggest_parameters(trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for parameter in search_space.get("parameters") or []:
        condition = parameter.get("condition") or {}
        if condition and any(
            parameters.get(key) != value for key, value in condition.items()
        ):
            continue
        name = parameter["name"]
        parameter_type = parameter["parameter_type"]
        if parameter_type == "categorical":
            parameters[name] = trial.suggest_categorical(
                name, parameter.get("choices") or []
            )
        elif parameter_type == "int":
            parameters[name] = trial.suggest_int(
                name,
                int(parameter["low"]),
                int(parameter["high"]),
                log=parameter.get("scale") == "log",
            )
        elif parameter_type == "float":
            parameters[name] = trial.suggest_float(
                name,
                float(parameter["low"]),
                float(parameter["high"]),
                log=parameter.get("scale") == "log",
            )
        else:
            raise ValueError(f"unsupported parameter type for {name}: {parameter_type}")
    validate_constraints(parameters, search_space.get("constraints") or [])
    return parameters


def validate_constraints(
    parameters: dict[str, Any], constraints: list[dict[str, Any]]
) -> None:
    for constraint in constraints:
        name = constraint.get("parameter")
        operator = constraint.get("operator")
        value = constraint.get("value")
        current = parameters.get(name)
        if current is None:
            continue
        if operator == "lte" and not current <= value:
            raise ValueError(f"constraint failed: {name} <= {value}")
        if operator == "gte" and not current >= value:
            raise ValueError(f"constraint failed: {name} >= {value}")
        if operator == "eq" and not current == value:
            raise ValueError(f"constraint failed: {name} == {value}")
        if operator == "in" and current not in value:
            raise ValueError(f"constraint failed: {name} in {value}")


def metric_observation(
    metrics: dict[str, Any], metric: str
) -> tuple[float | None, str | None]:
    """Return only semantically compatible aliases for the requested objective."""
    aliases = {
        "best_error_rate": ("best_error_rate", "valid_error_rate"),
        "valid_error_rate": ("valid_error_rate", "final_valid_error_rate"),
        "final_valid_error_rate": ("final_valid_error_rate",),
        "eer": ("eer",),
        "min_dcf": ("min_dcf",),
    }
    for candidate in aliases.get(metric, (metric,)):
        value = metrics.get(candidate)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), candidate
    return None, None


def metric_value(metrics: dict[str, Any], metric: str) -> float | None:
    return metric_observation(metrics, metric)[0]


def flatten_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(raw.get("metrics") or {})
    metrics.update(raw.get("final_metrics") or {})
    for key in (
        "eer",
        "min_dcf",
        "best_error_rate",
        "valid_error_rate",
        "final_valid_error_rate",
        "train_loss",
        "valid_loss",
    ):
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = value

    for namespace in ("validation", "test", "train"):
        nested = metrics.get(namespace)
        if isinstance(nested, dict):
            metrics.update(nested)
    return metrics


def run_training_stage(
    *,
    args: argparse.Namespace,
    adapter_bundle: Any,
    config_data: dict[str, Any],
    output_dir: Path,
    variant: str,
    trial_number: int,
    stage_index: int,
    parameters: dict[str, Any],
    budget: dict[str, Any],
) -> TrialRunRecord:
    started_at = utc_now()
    started = time.perf_counter()
    stage = str(budget.get("stage") or f"stage_{stage_index}")
    trial_root = output_dir / variant / f"trial_{trial_number:04d}"
    trial_dir = trial_root / "stages" / f"{stage_index:02d}_{stage}"
    trial_output = trial_root / "output"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_output.mkdir(parents=True, exist_ok=True)

    data_folder = str(
        resolve_data_path(args.data_folder or config_data.get("data_folder"))
    )
    overrides: dict[str, Any] = {
        "data_folder": data_folder,
        "output_folder": str(trial_output),
        **parameters,
    }
    if budget.get("epochs") is not None:
        overrides["number_of_epochs"] = int(budget["epochs"])
    if budget.get("data_fraction") is not None:
        overrides["_hpo_data_fraction"] = float(budget["data_fraction"])
    if budget.get("max_duration_seconds") is not None:
        overrides["_hpo_max_duration_seconds"] = float(budget["max_duration_seconds"])
    run_opts: dict[str, Any] = {}
    if args.device:
        run_opts["device"] = args.device
    if args.precision:
        run_opts["precision"] = args.precision
    if args.eval_precision:
        run_opts["eval_precision"] = args.eval_precision
    if run_opts:
        overrides["_run_opts"] = run_opts

    error = None
    raw: dict[str, Any] = {}
    try:
        validator = getattr(adapter_bundle.model, "validate_parameters", None)
        if callable(validator):
            validator(parameters)
        raw = adapter_bundle.runner.run_training(str(args.config_path), overrides)
        collected = collect_training_result(
            adapter_bundle.runner, raw, trial_output, trial_dir
        )
        collected["status"] = raw.get("status", collected.get("status", "success"))
        collected["error"] = raw.get("error") or collected.get("error")
    except Exception as exc:
        collected = {
            "status": "failed",
            "metrics": {},
            "output_folder": str(trial_output),
            "error": f"{type(exc).__name__}: {exc}",
        }
    status = str(collected.get("status") or "failed")
    error = collected.get("error")
    metrics = flatten_metrics(collected)
    value, objective_metric_name = metric_observation(metrics, args.primary_metric)
    finished_at = utc_now()
    duration_seconds = time.perf_counter() - started
    record = TrialRunRecord(
        variant=variant,
        model_family=args.model_family,
        optuna_trial_number=trial_number,
        stage_index=stage_index,
        stage=stage,
        status=status,
        value=value,
        parameters=dict(parameters),
        budget=dict(budget),
        output_folder=collected.get("output_folder") or str(trial_output),
        model_paths=[str(path) for path in collected.get("model_paths") or []],
        train_log_path=collected.get("train_log_path"),
        objective_metric_name=objective_metric_name,
        started_at=started_at,
        finished_at=finished_at,
        metrics=metrics,
        duration_seconds=duration_seconds,
        error=error,
    )
    write_json_atomic(trial_dir / "trial_record.json", asdict(record))
    return record


def load_stage_records(variant_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(variant_dir.glob("trial_*/**/trial_record.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def variant_signature(
    args: argparse.Namespace,
    variant: ClassicBaselineVariant,
    search_space: dict[str, Any],
) -> str:
    payload = {
        "model_family": args.model_family,
        "config_path": str(args.config_path),
        "primary_metric": args.primary_metric,
        "metric_mode": args.metric_mode,
        "sampler": variant.sampler,
        "pruner": variant.pruner,
        "budgets": variant.budgets,
        "search_space": search_space,
        "seed": args.seed,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_variant_result(
    *,
    args: argparse.Namespace,
    variant: ClassicBaselineVariant,
    study: Any,
    trial_records: list[dict[str, Any]],
    started_at: str,
    session_duration_seconds: float,
    storage_path: Path,
) -> dict[str, Any]:
    import optuna

    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "intermediate_values": dict(trial.intermediate_values),
            "datetime_start": (
                trial.datetime_start.isoformat() if trial.datetime_start else None
            ),
            "datetime_complete": (
                trial.datetime_complete.isoformat() if trial.datetime_complete else None
            ),
            "duration_seconds": (
                trial.duration.total_seconds() if trial.duration else None
            ),
            "user_attrs": dict(trial.user_attrs),
        }
        for trial in study.trials
    ]
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    best_trial = study.best_trial if complete else None
    best_stage = None
    if best_trial is not None:
        candidates = [
            record
            for record in trial_records
            if record.get("optuna_trial_number") == best_trial.number
            and record.get("status") == "success"
        ]
        if candidates:
            best_stage = max(
                candidates, key=lambda record: int(record.get("stage_index") or 0)
            )
    training_duration = sum(
        float(record.get("duration_seconds") or 0.0) for record in trial_records
    )
    optimization_duration = sum(item["duration_seconds"] or 0.0 for item in trials)
    return {
        "model_family": args.model_family,
        "variant": variant.name,
        "sampler": variant.sampler,
        "pruner": variant.pruner,
        "requested_n_trials": variant.n_trials,
        "n_trials": len(trials),
        "objective_metric": args.primary_metric,
        "budget_execution_mode": "shared_trial_checkpoint_directory",
        "metric_mode": args.metric_mode,
        "budgets": variant.budgets,
        "status": "success" if complete else "failed",
        "started_at": study.user_attrs.get("experiment_started_at", started_at),
        "session_started_at": started_at,
        "finished_at": utc_now(),
        "session_duration_seconds": session_duration_seconds,
        "training_duration_seconds": training_duration,
        "optimization_duration_seconds": optimization_duration,
        "duration_seconds": optimization_duration,
        "study_storage_path": str(storage_path),
        "best_trial_number": best_trial.number if best_trial is not None else None,
        "best_value": best_trial.value if best_trial is not None else None,
        "best_params": dict(best_stage.get("parameters") or {}) if best_stage else {},
        "best_objective_metric_name": (
            best_stage.get("objective_metric_name") if best_stage else None
        ),
        "best_metrics": dict(best_stage.get("metrics") or {}) if best_stage else {},
        "best_output_folder": best_stage.get("output_folder") if best_stage else None,
        "best_model_paths": (
            list(best_stage.get("model_paths") or []) if best_stage else []
        ),
        "trials": trials,
        "stage_runs": trial_records,
    }


def run_variant(
    *,
    args: argparse.Namespace,
    variant: ClassicBaselineVariant,
    search_space: dict[str, Any],
    output_dir: Path,
    config_data: dict[str, Any],
) -> dict[str, Any]:
    import optuna

    adapter_bundle = validate_experiment_setup(
        args, [variant], search_space, config_data
    )
    sampler, pruner = make_sampler_and_pruner(variant, seed=args.seed)
    variant_dir = output_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    storage_path = variant_dir / "optuna_study.sqlite3"
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    study = optuna.create_study(
        direction="minimize" if args.metric_mode == "min" else "maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{args.comparison_id}_{args.model_family}_{variant.name}",
        storage=storage_url,
        load_if_exists=True,
    )
    signature = variant_signature(args, variant, search_space)
    saved_signature = study.user_attrs.get("experiment_signature")
    if saved_signature is not None and saved_signature != signature:
        raise ValueError(
            f"existing study configuration differs for {variant.name}; "
            "use a new --comparison-id or --output-dir"
        )
    if saved_signature is None:
        study.set_user_attr("experiment_signature", signature)
        study.set_user_attr("experiment_started_at", utc_now())
    trial_records = load_stage_records(variant_dir)
    started_at = utc_now()
    started = time.perf_counter()

    def objective(trial: Any) -> float:
        parameters = suggest_parameters(trial, search_space)
        trial.set_user_attr("resolved_parameters", parameters)
        last_value: float | None = None
        for stage_index, budget in enumerate(variant.budgets, start=1):
            record = run_training_stage(
                args=args,
                adapter_bundle=adapter_bundle,
                config_data=config_data,
                output_dir=output_dir,
                variant=variant.name,
                trial_number=trial.number,
                stage_index=stage_index,
                parameters=parameters,
                budget=budget,
            )
            trial_records.append(asdict(record))
            trial.set_user_attr(
                f"stage_{stage_index}",
                {
                    "status": record.status,
                    "value": record.value,
                    "duration_seconds": record.duration_seconds,
                    "output_folder": record.output_folder,
                },
            )
            if record.status != "success":
                raise RuntimeError(record.error or "training failed")
            if record.value is None:
                raise RuntimeError(f"primary metric not found: {args.primary_metric}")
            last_value = float(record.value)
            trial.report(last_value, step=stage_index)
            if stage_index < len(variant.budgets) and trial.should_prune():
                raise optuna.TrialPruned(f"pruned after budget stage {stage_index}")
        if last_value is None:
            raise RuntimeError("no budget stage was executed")
        return last_value

    def persist_progress(current_study: Any, _trial: Any) -> None:
        progress = build_variant_result(
            args=args,
            variant=variant,
            study=current_study,
            trial_records=trial_records,
            started_at=started_at,
            session_duration_seconds=time.perf_counter() - started,
            storage_path=storage_path,
        )
        write_json_atomic(variant_dir / "variant_progress.json", progress)

    remaining_trials = max(0, variant.n_trials - len(study.trials))
    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            catch=(RuntimeError, ValueError),
            callbacks=[persist_progress],
        )
    result = build_variant_result(
        args=args,
        variant=variant,
        study=study,
        trial_records=trial_records,
        started_at=started_at,
        session_duration_seconds=time.perf_counter() - started,
        storage_path=storage_path,
    )
    write_json_atomic(variant_dir / "variant_result.json", result)
    return result


def summarize(
    results: list[dict[str, Any]],
    *,
    metric: str,
    mode: str,
    started_at: str | None = None,
    total_duration_seconds: float | None = None,
) -> dict[str, Any]:
    ranked = [
        (item["variant"], item.get("best_value"))
        for item in results
        if isinstance(item.get("best_value"), (int, float))
    ]
    ranked.sort(key=lambda item: item[1], reverse=mode == "max")
    return {
        "metric": metric,
        "mode": mode,
        "ranking": ranked,
        "started_at": started_at,
        "finished_at": utc_now(),
        "total_duration_seconds": total_duration_seconds,
        "by_variant": {
            item["variant"]: {
                "model_family": item.get("model_family"),
                "status": item.get("status"),
                "best_trial_number": item.get("best_trial_number"),
                "best_value": item.get("best_value"),
                "best_params": item.get("best_params"),
                "best_metrics": item.get("best_metrics"),
                "best_output_folder": item.get("best_output_folder"),
                "best_model_paths": item.get("best_model_paths"),
                "completed_trials": sum(
                    1
                    for trial in item.get("trials", [])
                    if trial.get("state") == "COMPLETE"
                ),
                "pruned_trials": sum(
                    1
                    for trial in item.get("trials", [])
                    if trial.get("state") == "PRUNED"
                ),
                "failed_trials": sum(
                    1
                    for trial in item.get("trials", [])
                    if trial.get("state") == "FAIL"
                ),
                "stage_run_count": len(item.get("stage_runs", [])),
                "duration_seconds": item.get("duration_seconds"),
                "training_duration_seconds": item.get("training_duration_seconds"),
                "session_duration_seconds": item.get("session_duration_seconds"),
            }
            for item in results
        },
    }


def build_plan(
    args: argparse.Namespace,
    variants: list[ClassicBaselineVariant],
    search_space: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "comparison_id": args.comparison_id,
        "created_at": utc_now(),
        "note": "Classic Optuna baselines; no SR-agent HPO service or LLM advisor is used.",
        "suite": args.suite,
        "config_path": str(args.config_path),
        "data_folder": args.data_folder,
        "task_type": args.task_type,
        "model_family": args.model_family,
        "implementation": args.implementation,
        "runner": args.runner,
        "primary_metric": args.primary_metric,
        "metric_mode": args.metric_mode,
        "seed": args.seed,
        "search_space": deepcopy(search_space),
        "variants": [asdict(item) for item in variants],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suite", choices=["smoke", "classic"], default="smoke")
    result.add_argument(
        "--only",
        help="Comma-separated variants: random_search,tpe,tpe_successive_halving,bohb.",
    )
    result.add_argument(
        "--comparison-id",
        default=f"classic_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}",
    )
    result.add_argument("--output-dir")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--n-trials", type=int)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--config-path")
    result.add_argument("--data-folder")
    result.add_argument("--task-type", default="speaker_verification")
    result.add_argument(
        "--model-family", choices=SUPPORTED_MODEL_FAMILIES, default="ecapa_tdnn"
    )
    result.add_argument("--implementation", default="speechbrain")
    result.add_argument("--runner", default="speechbrain")
    result.add_argument("--primary-metric", default="best_error_rate")
    result.add_argument("--metric-mode", choices=["min", "max"], default="min")
    result.add_argument("--budgets-json")
    result.add_argument("--search-space-json")
    result.add_argument("--device")
    result.add_argument("--precision")
    result.add_argument("--eval-precision")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.n_trials is not None and args.n_trials <= 0:
        print("error: --n-trials must be positive", file=sys.stderr)
        return 2
    try:
        if args.config_path is None:
            args.config_path = model_defaults(args.model_family)["config_path"]
        args.config_path = resolve_config_path(args.config_path)
        variants = selected_variants(args)
        search_space = load_json_arg(
            args.search_space_json, expected_type=dict, label="search space"
        ) or default_search_space(args.model_family)
        config_data = ConfigParser(str(args.config_path)).load_config(
            resolve_references=True
        )
        args.data_folder = str(
            resolve_data_path(args.data_folder or config_data.get("data_folder"))
        )
        validate_experiment_setup(args, variants, search_space, config_data)
    except (argparse.ArgumentTypeError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.dry_run and not Path(args.data_folder).is_dir():
        print(
            f"error: data folder does not exist or is not a directory: {args.data_folder}",
            file=sys.stderr,
        )
        return 2

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else get_experiments_dir() / "classic_optuna_baselines" / args.comparison_id
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args, variants, search_space)
    plan_path = output_dir / "classic_baseline_plan.json"
    write_json_atomic(plan_path, plan)
    print(
        json.dumps(
            {"comparison_id": args.comparison_id, "plan_path": str(plan_path)},
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return 0

    optimization_started_at = utc_now()
    optimization_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    results_path = output_dir / "classic_baseline_results.json"
    summary_path = output_dir / "classic_baseline_summary.json"
    for variant in variants:
        result = run_variant(
            args=args,
            variant=variant,
            search_space=search_space,
            output_dir=output_dir,
            config_data=config_data,
        )
        results.append(result)
        write_json_atomic(results_path, results)
        write_json_atomic(
            summary_path,
            summarize(
                results,
                metric=args.primary_metric,
                mode=args.metric_mode,
                started_at=optimization_started_at,
                total_duration_seconds=time.perf_counter() - optimization_started,
            ),
        )
    print(
        json.dumps(
            {"results_path": str(results_path), "summary_path": str(summary_path)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
