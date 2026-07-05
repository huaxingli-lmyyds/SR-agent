import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.run_comparison_experiments import (
    ComparisonVariant,
    summarize,
    selected_variants,
    variants_for_suite,
)


def test_smoke_suite_contains_baseline_random_and_tpe() -> None:
    names = [item.name for item in variants_for_suite("smoke")]

    assert names == ["default_baseline", "random_search", "tpe"]


def test_variant_context_and_budget_are_hpo_compatible() -> None:
    variant = ComparisonVariant(
        "tpe_test",
        "tpe",
        max_training_runs=3,
        max_studies=2,
    )

    context = variant.to_context(primary_metric="eer", metric_mode="min")
    budget = variant.to_budget()

    assert context["strategy"] == "tpe"
    assert context["primary_metric"] == "eer"
    assert context["search_space"]["parameters"]
    assert context["budgets"][0]["epochs"] == 3
    assert budget["max_training_runs"] == 3
    assert budget["max_studies"] == 2


def test_selected_variants_filters_and_overrides_budget() -> None:
    args = argparse.Namespace(
        suite="low_fidelity",
        only="random_search,tpe",
        max_training_runs=2,
        max_studies=1,
        max_retries=0,
        strategy_review_interval_trials=2,
        budgets_json='[{"stage":"tiny","epochs":1,"data_fraction":0.1}]',
        search_space_json=None,
    )

    variants = selected_variants(args)

    assert [item.name for item in variants] == ["random_search", "tpe"]
    assert all(item.max_training_runs == 2 for item in variants)
    assert all(item.max_retries == 0 for item in variants)
    assert all(item.budgets[0]["stage"] == "tiny" for item in variants)


def test_summarize_ranks_min_metric() -> None:
    summary = summarize(
        [
            {"variant": "random", "status": "success", "primary_metric_value": 3.0, "duration_seconds": 10},
            {"variant": "tpe", "status": "success", "primary_metric_value": 2.0, "duration_seconds": 12},
            {"variant": "tpe", "status": "failed", "primary_metric_value": None, "duration_seconds": 1},
        ],
        metric="eer",
        mode="min",
    )

    assert summary["ranking"][0] == ("tpe", 2.0)
    assert summary["by_variant"]["tpe"]["successes"] == 1
    assert summary["by_variant"]["tpe"]["failures"] == 1


def test_comparison_script_dry_run_writes_plan(tmp_path) -> None:
    script = Path("scripts/run_comparison_experiments.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "smoke",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--comparison-id",
            "cmp_test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    plan = json.loads((tmp_path / "comparison_plan.json").read_text(encoding="utf-8"))
    assert plan["comparison_id"] == "cmp_test"
    assert [item["name"] for item in plan["variants"]] == [
        "default_baseline",
        "random_search",
        "tpe",
    ]
    assert "comparison_plan.json" in completed.stdout

def test_resnet_suite_uses_resnet_baseline_and_search_space() -> None:
    variants = variants_for_suite("smoke", "resnet")
    baseline = variants[0]

    assert baseline.search_space["parameters"][1]["choices"] == [32]

    args = argparse.Namespace(
        suite="smoke",
        only="random_search",
        model_family="resnet",
        max_training_runs=None,
        max_studies=None,
        max_retries=None,
        strategy_review_interval_trials=None,
        budgets_json=None,
        search_space_json=None,
    )
    selected = selected_variants(args)
    assert selected[0].search_space["parameters"][1]["choices"] == [16, 24, 32, 48]


def test_xvector_suite_exposes_lr_final_search_parameter() -> None:
    args = argparse.Namespace(
        suite="smoke",
        only="tpe",
        model_family="xvector",
        max_training_runs=None,
        max_studies=None,
        max_retries=None,
        strategy_review_interval_trials=None,
        budgets_json=None,
        search_space_json=None,
    )

    selected = selected_variants(args)
    names = [item["name"] for item in selected[0].search_space["parameters"]]

    assert "lr_final" in names