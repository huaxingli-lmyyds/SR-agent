"""Project entry point for the orchestration agent."""

from __future__ import annotations

import argparse
import json
from typing import Any

from agent.utils.path_tool import get_config_file


def _json_object(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc


def _int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers"
        ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SR-agent orchestration entry point")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--data-iterations", type=int, default=6)
    parser.add_argument("--model-name", type=str, default="GLM-4.7")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default="Improve the primary model metric")
    parser.add_argument(
        "--strategy",
        choices=[
            "auto",
            "random_search",
            "grid_search",
            "adaptive_search",
            "tpe",
            "successive_halving",
        ],
        default="auto",
    )
    parser.add_argument(
        "--enable-llm-advisor",
        action="store_true",
        help="Allow LLMs to submit structured strategy/data-processing proposals.",
    )
    parser.add_argument(
        "--max-training-runs",
        type=int,
        default=None,
        help="Maximum training runs per Study. Defaults to --max-iterations.",
    )
    parser.add_argument("--max-studies", type=int, default=1)
    parser.add_argument("--max-total-training-runs", type=int, default=None)
    parser.add_argument("--campaign-patience", type=int, default=None)
    parser.add_argument("--campaign-min-improvement", type=float, default=0.0)
    parser.add_argument("--target-value", type=float, default=None)
    parser.add_argument("--initial-trial-count", type=int, default=None)
    parser.add_argument(
        "--promotion-limits",
        type=_int_list,
        default=None,
        help="Comma-separated promotion limits for successive halving, e.g. 2,1.",
    )
    parser.add_argument("--min-completed-per-rung", type=int, default=1)
    parser.add_argument("--strategy-review-interval-trials", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--budgets-json",
        type=_json_object,
        default=None,
        help=(
            "JSON list of Trial budgets, e.g. "
            '\'[{"stage":"screen","epochs":3,"data_fraction":0.25}]\''
        ),
    )
    parser.add_argument(
        "--budget-stage",
        type=str,
        default=None,
        help="Stage name for a single budget rung when --budgets-json is omitted.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs for a single budget rung.",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=None,
        help="Training data fraction for a single budget rung, in (0, 1].",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=None,
        help="Timeout for each training run in seconds.",
    )
    parser.add_argument(
        "--search-space-json",
        type=_json_object,
        default=None,
        help="Optional JSON search space object with parameters/constraints.",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default=None,
        help="Dataset path; absolute paths such as /tmp/voxceleb1 are supported.",
    )
    parser.add_argument("--primary-metric", type=str, default="eer")
    parser.add_argument("--metric-mode", choices=["min", "max"], default="min")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(get_config_file("train_ecapa_tdnn.yaml")),
    )
    parser.add_argument("--task-type", type=str, default="speaker_verification")
    parser.add_argument("--model-family", type=str, default="ecapa_tdnn")
    parser.add_argument("--implementation", type=str, default="speechbrain")
    parser.add_argument("--runner", type=str, default="speechbrain")
    return parser


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    context: dict[str, Any] = {
        "strategy": args.strategy,
        "primary_metric": args.primary_metric,
        "metric_mode": args.metric_mode,
    }
    if args.data_folder is not None:
        context["data_folder"] = args.data_folder
        context["dataset_uri"] = args.data_folder
    if args.search_space_json is not None:
        if not isinstance(args.search_space_json, dict):
            raise SystemExit("--search-space-json must be a JSON object")
        context["search_space"] = args.search_space_json
    if args.target_value is not None:
        context["target_value"] = args.target_value

    if args.budgets_json is not None:
        if not isinstance(args.budgets_json, list):
            raise SystemExit("--budgets-json must be a JSON list")
        context["budgets"] = args.budgets_json
    elif any(
        value is not None
        for value in (
            args.budget_stage,
            args.epochs,
            args.data_fraction,
            args.max_duration_seconds,
        )
    ):
        budget: dict[str, Any] = {"stage": args.budget_stage or "full"}
        if args.epochs is not None:
            budget["epochs"] = args.epochs
        if args.data_fraction is not None:
            budget["data_fraction"] = args.data_fraction
        if args.max_duration_seconds is not None:
            budget["max_duration_seconds"] = args.max_duration_seconds
        context["budgets"] = [budget]
    return context


def build_budget(args: argparse.Namespace) -> dict[str, Any]:
    budget: dict[str, Any] = {
        "max_training_runs": args.max_training_runs or args.max_iterations,
        "max_studies": args.max_studies,
        "campaign_min_improvement": args.campaign_min_improvement,
        "min_completed_per_rung": args.min_completed_per_rung,
        "strategy_review_interval_trials": args.strategy_review_interval_trials,
        "max_retries": args.max_retries,
    }
    optional_values = {
        "max_total_training_runs": args.max_total_training_runs,
        "campaign_patience": args.campaign_patience,
        "target_value": args.target_value,
        "initial_trial_count": args.initial_trial_count,
        "promotion_limits": args.promotion_limits,
    }
    budget.update({
        key: value for key, value in optional_values.items() if value is not None
    })
    return budget


def main() -> None:
    args = build_arg_parser().parse_args()

    from agent.agents.orchestrator import CoordinatorAgent

    pipeline = CoordinatorAgent(
        model_name=args.model_name,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        data_iterations=args.data_iterations,
        verbose=args.verbose,
        config_path=args.config_path,
        task_type=args.task_type,
        model_family=args.model_family,
        implementation=args.implementation,
        runner=args.runner,
        enable_llm_advisor=args.enable_llm_advisor,
    )
    result = pipeline.run(
        objective=args.objective,
        context=build_context(args),
        budget=build_budget(args),
    )
    print(result.experiment_id)


if __name__ == "__main__":
    main()
