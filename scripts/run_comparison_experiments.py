#!/usr/bin/env python3
"""Run reproducible comparison campaigns for HPO strategies.

The script intentionally keeps orchestration in Python instead of shell loops so every
variant writes the same plan/result schema and can be inspected after long GPU runs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any
from uuid import uuid4

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from agent.utils.path_tool import get_experiments_dir


MODEL_EXPERIMENT_DEFAULTS = {
    "ecapa_tdnn": {
        "config_path": "configs/train_ecapa_tdnn.yaml",
        "search_space": {
            "parameters": [
                {"name": "lr", "parameter_type": "float", "low": 1e-5, "high": 3e-3, "scale": "log"},
                {"name": "batch_size", "parameter_type": "categorical", "choices": [8, 16, 24, 32]},
                {"name": "sentence_len", "parameter_type": "categorical", "choices": [2.0, 3.0, 4.0]},
            ],
            "constraints": [],
        },
        "baseline_space": {
            "parameters": [
                {"name": "lr", "parameter_type": "categorical", "choices": [0.001]},
                {"name": "batch_size", "parameter_type": "categorical", "choices": [16]},
                {"name": "sentence_len", "parameter_type": "categorical", "choices": [4.0]},
            ],
            "constraints": [],
        },
    },
    "resnet": {
        "config_path": "recipes/voxceleb/hparams/train_resnet.yaml",
        "search_space": {
            "parameters": [
                {"name": "lr", "parameter_type": "float", "low": 1e-5, "high": 3e-3, "scale": "log"},
                {"name": "batch_size", "parameter_type": "categorical", "choices": [16, 24, 32, 48]},
                {"name": "sentence_len", "parameter_type": "categorical", "choices": [2.0, 3.0, 4.0]},
            ],
            "constraints": [],
        },
        "baseline_space": {
            "parameters": [
                {"name": "lr", "parameter_type": "categorical", "choices": [0.001]},
                {"name": "batch_size", "parameter_type": "categorical", "choices": [32]},
                {"name": "sentence_len", "parameter_type": "categorical", "choices": [3.0]},
            ],
            "constraints": [],
        },
    },
    "xvector": {
        "config_path": "recipes/voxceleb/hparams/train_x_vectors.yaml",
        "search_space": {
            "parameters": [
                {"name": "lr", "parameter_type": "float", "low": 1e-5, "high": 3e-3, "scale": "log"},
                {"name": "lr_final", "parameter_type": "float", "low": 1e-6, "high": 1e-3, "scale": "log"},
                {"name": "batch_size", "parameter_type": "categorical", "choices": [64, 128, 256]},
                {"name": "sentence_len", "parameter_type": "categorical", "choices": [2.0, 3.0, 4.0]},
            ],
            "constraints": [{"parameter": "lr_final", "operator": "lte", "value": 0.001}],
        },
        "baseline_space": {
            "parameters": [
                {"name": "lr", "parameter_type": "categorical", "choices": [0.001]},
                {"name": "lr_final", "parameter_type": "categorical", "choices": [0.0001]},
                {"name": "batch_size", "parameter_type": "categorical", "choices": [256]},
                {"name": "sentence_len", "parameter_type": "categorical", "choices": [3.0]},
            ],
            "constraints": [],
        },
    },
}


def model_defaults(model_family: str) -> dict[str, Any]:
    try:
        return MODEL_EXPERIMENT_DEFAULTS[model_family]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_EXPERIMENT_DEFAULTS))
        raise ValueError(f"unsupported model family for comparison script: {model_family}; available: {available}") from exc


def default_search_space(model_family: str) -> dict[str, Any]:
    return deepcopy(model_defaults(model_family)["search_space"])


def baseline_search_space(model_family: str) -> dict[str, Any]:
    return deepcopy(model_defaults(model_family)["baseline_space"])

SMOKE_BUDGETS = [
    {"stage": "smoke", "epochs": 1, "data_fraction": 0.05, "max_duration_seconds": 1800.0}
]

LOW_FIDELITY_BUDGETS = [
    {"stage": "screen", "epochs": 3, "data_fraction": 0.25, "max_duration_seconds": None}
]

FULL_BUDGETS = [
    {"stage": "full", "epochs": 24, "data_fraction": 1.0, "max_duration_seconds": None}
]

HALVING_BUDGETS = [
    {"stage": "screen", "epochs": 3, "data_fraction": 0.25, "max_duration_seconds": None},
    {"stage": "promote", "epochs": 8, "data_fraction": 0.5, "max_duration_seconds": None},
    {"stage": "full", "epochs": 24, "data_fraction": 1.0, "max_duration_seconds": None},
]


@dataclass
class ComparisonVariant:
    name: str
    strategy: str
    enable_llm_advisor: bool = False
    max_training_runs: int = 4
    max_studies: int = 1
    budgets: list[dict[str, Any]] = field(default_factory=lambda: deepcopy(LOW_FIDELITY_BUDGETS))
    search_space: dict[str, Any] = field(default_factory=lambda: default_search_space("ecapa_tdnn"))
    initial_trial_count: int | None = None
    promotion_limits: list[int] | None = None
    min_completed_per_rung: int = 1
    strategy_review_interval_trials: int = 3
    max_retries: int = 1

    def to_context(self, *, primary_metric: str, metric_mode: str) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "primary_metric": primary_metric,
            "metric_mode": metric_mode,
            "search_space": deepcopy(self.search_space),
            "budgets": deepcopy(self.budgets),
            "comparison_variant": self.name,
        }

    def to_budget(self) -> dict[str, Any]:
        budget: dict[str, Any] = {
            "max_training_runs": self.max_training_runs,
            "max_studies": self.max_studies,
            "min_completed_per_rung": self.min_completed_per_rung,
            "strategy_review_interval_trials": self.strategy_review_interval_trials,
            "max_retries": self.max_retries,
        }
        if self.initial_trial_count is not None:
            budget["initial_trial_count"] = self.initial_trial_count
        if self.promotion_limits is not None:
            budget["promotion_limits"] = list(self.promotion_limits)
        return budget


def variants_for_suite(suite: str, model_family: str = "ecapa_tdnn") -> list[ComparisonVariant]:
    if suite == "smoke":
        return [
            ComparisonVariant(
                "default_baseline",
                "grid_search",
                max_training_runs=1,
                budgets=deepcopy(SMOKE_BUDGETS),
                search_space=baseline_search_space(model_family),
            ),
            ComparisonVariant(
                "random_search",
                "random_search",
                max_training_runs=2,
                budgets=deepcopy(SMOKE_BUDGETS),
            ),
            ComparisonVariant(
                "tpe",
                "tpe",
                max_training_runs=2,
                budgets=deepcopy(SMOKE_BUDGETS),
            ),
        ]
    if suite == "low_fidelity":
        return [
            ComparisonVariant(
                "default_baseline",
                "grid_search",
                max_training_runs=1,
                search_space=baseline_search_space(model_family),
            ),
            ComparisonVariant("random_search", "random_search", max_training_runs=6),
            ComparisonVariant("tpe", "tpe", max_training_runs=6),
            ComparisonVariant("adaptive_search", "adaptive_search", max_training_runs=6),
            ComparisonVariant(
                "successive_halving",
                "successive_halving",
                max_training_runs=8,
                budgets=deepcopy(HALVING_BUDGETS),
                initial_trial_count=6,
                promotion_limits=[2, 1],
                min_completed_per_rung=2,
            ),
        ]
    if suite == "llm_ablation":
        return [
            ComparisonVariant("tpe", "tpe", max_training_runs=6),
            ComparisonVariant(
                "llm_assisted_tpe",
                "tpe",
                enable_llm_advisor=True,
                max_training_runs=6,
            ),
            ComparisonVariant("adaptive_search", "adaptive_search", max_training_runs=6),
            ComparisonVariant(
                "llm_assisted_adaptive",
                "adaptive_search",
                enable_llm_advisor=True,
                max_training_runs=6,
            ),
        ]
    if suite == "full_confirm":
        return [
            ComparisonVariant(
                "default_baseline",
                "grid_search",
                max_training_runs=1,
                budgets=deepcopy(FULL_BUDGETS),
                search_space=baseline_search_space(model_family),
            ),
            ComparisonVariant("random_search", "random_search", max_training_runs=3, budgets=deepcopy(FULL_BUDGETS)),
            ComparisonVariant("tpe", "tpe", max_training_runs=3, budgets=deepcopy(FULL_BUDGETS)),
            ComparisonVariant(
                "llm_assisted_tpe",
                "tpe",
                enable_llm_advisor=True,
                max_training_runs=3,
                budgets=deepcopy(FULL_BUDGETS),
            ),
        ]
    raise ValueError(f"unsupported comparison suite: {suite}")


def load_json_arg(value: str | None, *, expected_type: type, label: str) -> Any:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(parsed, expected_type):
        raise argparse.ArgumentTypeError(f"{label} must be a JSON {expected_type.__name__}")
    return parsed


def selected_variants(args: argparse.Namespace) -> list[ComparisonVariant]:
    variants = variants_for_suite(args.suite, args.model_family)
    for variant in variants:
        if variant.name != "default_baseline":
            variant.search_space = default_search_space(args.model_family)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        variants = [item for item in variants if item.name in wanted]
        missing = sorted(wanted - {item.name for item in variants})
        if missing:
            raise ValueError(f"unknown variant(s) for suite {args.suite}: {', '.join(missing)}")
    override_budgets = load_json_arg(args.budgets_json, expected_type=list, label="budgets")
    override_search_space = load_json_arg(args.search_space_json, expected_type=dict, label="search space")
    for variant in variants:
        if args.max_training_runs is not None:
            variant.max_training_runs = args.max_training_runs
        if args.max_studies is not None:
            variant.max_studies = args.max_studies
        if args.max_retries is not None:
            variant.max_retries = args.max_retries
        if args.strategy_review_interval_trials is not None:
            variant.strategy_review_interval_trials = args.strategy_review_interval_trials
        if override_budgets is not None and variant.strategy != "successive_halving":
            variant.budgets = deepcopy(override_budgets)
        if override_search_space is not None and variant.name != "default_baseline":
            variant.search_space = deepcopy(override_search_space)
    return variants


def build_plan(args: argparse.Namespace, variants: list[ComparisonVariant]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "comparison_id": args.comparison_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "repetitions": args.repetitions,
        "config_path": args.config_path,
        "objective": args.objective,
        "task_type": args.task_type,
        "model_family": args.model_family,
        "implementation": args.implementation,
        "runner": args.runner,
        "primary_metric": args.primary_metric,
        "metric_mode": args.metric_mode,
        "variants": [asdict(item) for item in variants],
    }


def metric_value(result: dict[str, Any], metric: str) -> Any:
    direct = (result.get("metrics") or {}).get(metric)
    if direct is not None:
        return direct
    for task_result in result.get("task_results") or []:
        value = ((task_result.get("result") or {}).get("metrics") or {}).get(metric)
        if value is not None:
            return value
    return None


def run_variant(args: argparse.Namespace, variant: ComparisonVariant, repetition: int) -> dict[str, Any]:
    from agent.agents.orchestrator import CoordinatorAgent

    started = time.time()
    pipeline = CoordinatorAgent(
        model_name=args.model_name,
        temperature=args.temperature,
        max_iterations=variant.max_training_runs,
        data_iterations=args.data_iterations,
        verbose=args.verbose,
        config_path=args.config_path,
        task_type=args.task_type,
        model_family=args.model_family,
        implementation=args.implementation,
        runner=args.runner,
        enable_llm_advisor=variant.enable_llm_advisor,
    )
    context = variant.to_context(primary_metric=args.primary_metric, metric_mode=args.metric_mode)
    context.update({
        "comparison_id": args.comparison_id,
        "comparison_suite": args.suite,
        "comparison_repetition": repetition,
    })
    result = pipeline.run(
        objective=f"{args.objective} [{variant.name} repeat {repetition}]",
        context=context,
        budget=variant.to_budget(),
    ).to_dict()
    return {
        "variant": variant.name,
        "strategy": variant.strategy,
        "enable_llm_advisor": variant.enable_llm_advisor,
        "repetition": repetition,
        "status": result.get("status"),
        "duration_seconds": time.time() - started,
        "primary_metric_value": metric_value(result, args.primary_metric),
        "orchestration_experiment_id": result.get("experiment_id"),
        "result": result,
    }


def summarize(results: list[dict[str, Any]], *, metric: str, mode: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(item["variant"], []).append(item)
    summary = {}
    for variant, items in grouped.items():
        values = [item.get("primary_metric_value") for item in items]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        best = None
        if numeric:
            best = min(numeric) if mode == "min" else max(numeric)
        summary[variant] = {
            "runs": len(items),
            "successes": sum(item.get("status") == "success" for item in items),
            "failures": sum(item.get("status") != "success" for item in items),
            "metric": metric,
            "mode": mode,
            "values": values,
            "best": best,
            "average": sum(numeric) / len(numeric) if numeric else None,
            "total_duration_seconds": sum(float(item.get("duration_seconds") or 0) for item in items),
        }
    ranked = [
        (name, data["best"])
        for name, data in summary.items()
        if data["best"] is not None
    ]
    ranked.sort(key=lambda item: item[1], reverse=mode == "max")
    return {"by_variant": summary, "ranking": ranked}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suite", choices=["smoke", "low_fidelity", "llm_ablation", "full_confirm"], default="smoke")
    result.add_argument("--only", help="Comma-separated variant names to run from the selected suite.")
    result.add_argument("--comparison-id", default=f"cmp_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}")
    result.add_argument("--output-dir", default=None)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--repetitions", type=int, default=1)
    result.add_argument("--config-path")
    result.add_argument("--objective", default="Compare speaker-verification HPO strategies")
    result.add_argument("--task-type", default="speaker_verification")
    result.add_argument("--model-family", default="ecapa_tdnn")
    result.add_argument("--implementation", default="speechbrain")
    result.add_argument("--runner", default="speechbrain")
    result.add_argument("--primary-metric", default="eer")
    result.add_argument("--metric-mode", choices=["min", "max"], default="min")
    result.add_argument("--model-name", default="GLM-4.7")
    result.add_argument("--temperature", type=float, default=0.2)
    result.add_argument("--data-iterations", type=int, default=1)
    result.add_argument("--max-training-runs", type=int)
    result.add_argument("--max-studies", type=int)
    result.add_argument("--max-retries", type=int)
    result.add_argument("--strategy-review-interval-trials", type=int)
    result.add_argument("--budgets-json")
    result.add_argument("--search-space-json")
    result.add_argument("--verbose", action="store_true")
    result.add_argument("--continue-on-error", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.config_path is None:
        args.config_path = model_defaults(args.model_family)["config_path"]
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    try:
        variants = selected_variants(args)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir) if args.output_dir else get_experiments_dir() / "comparisons" / args.comparison_id
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args, variants)
    plan_path = output_dir / "comparison_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"comparison_id": args.comparison_id, "plan_path": str(plan_path)}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    results: list[dict[str, Any]] = []
    results_path = output_dir / "comparison_results.json"
    for repetition in range(1, args.repetitions + 1):
        for variant in variants:
            print(f"[comparison] running {variant.name} repeat {repetition}/{args.repetitions}", flush=True)
            try:
                item = run_variant(args, variant, repetition)
            except Exception as exc:
                item = {
                    "variant": variant.name,
                    "strategy": variant.strategy,
                    "enable_llm_advisor": variant.enable_llm_advisor,
                    "repetition": repetition,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if not args.continue_on_error:
                    results.append(item)
                    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                    raise
            results.append(item)
            results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    summary = summarize(results, metric=args.primary_metric, mode=args.metric_mode)
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"results_path": str(results_path), "summary_path": str(summary_path), "summary": summary}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())