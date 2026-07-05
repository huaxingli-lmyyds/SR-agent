import pytest

from main import build_arg_parser, build_budget, build_context


def parse_args(*args):
    return build_arg_parser().parse_args(list(args))


def test_cli_accepts_tpe_and_llm_advisor_flag() -> None:
    args = parse_args("--strategy", "tpe", "--enable-llm-advisor")

    assert args.strategy == "tpe"
    assert args.enable_llm_advisor is True


def test_cli_builds_single_budget_from_budget_flags() -> None:
    args = parse_args(
        "--strategy",
        "random_search",
        "--epochs",
        "3",
        "--data-fraction",
        "0.25",
        "--max-duration-seconds",
        "1800",
    )

    context = build_context(args)

    assert context["strategy"] == "random_search"
    assert context["budgets"] == [
        {
            "stage": "full",
            "epochs": 3,
            "data_fraction": 0.25,
            "max_duration_seconds": 1800.0,
        }
    ]


def test_cli_builds_budget_controls() -> None:
    args = parse_args(
        "--max-iterations",
        "4",
        "--max-training-runs",
        "3",
        "--max-studies",
        "2",
        "--max-total-training-runs",
        "5",
        "--campaign-patience",
        "2",
        "--campaign-min-improvement",
        "0.01",
        "--target-value",
        "2.5",
        "--initial-trial-count",
        "2",
        "--promotion-limits",
        "2,1",
        "--min-completed-per-rung",
        "2",
        "--strategy-review-interval-trials",
        "4",
        "--max-retries",
        "2",
    )

    budget = build_budget(args)
    context = build_context(args)

    assert budget["max_training_runs"] == 3
    assert budget["max_studies"] == 2
    assert budget["max_total_training_runs"] == 5
    assert budget["campaign_patience"] == 2
    assert budget["campaign_min_improvement"] == 0.01
    assert budget["target_value"] == 2.5
    assert budget["initial_trial_count"] == 2
    assert budget["promotion_limits"] == [2, 1]
    assert budget["min_completed_per_rung"] == 2
    assert budget["strategy_review_interval_trials"] == 4
    assert budget["max_retries"] == 2
    assert context["target_value"] == 2.5


def test_cli_accepts_json_budgets_and_search_space() -> None:
    args = parse_args(
        "--budgets-json",
        '[{"stage":"screen","epochs":3,"data_fraction":0.25}]',
        "--search-space-json",
        '{"parameters":[{"name":"lr","parameter_type":"float","low":1e-5,"high":1e-3,"scale":"log"}]}',
    )

    context = build_context(args)

    assert context["budgets"][0]["stage"] == "screen"
    assert context["budgets"][0]["epochs"] == 3
    assert context["search_space"]["parameters"][0]["name"] == "lr"


def test_cli_rejects_non_list_budgets_json() -> None:
    args = parse_args("--budgets-json", '{"stage":"bad"}')

    with pytest.raises(SystemExit, match="--budgets-json must be a JSON list"):
        build_context(args)