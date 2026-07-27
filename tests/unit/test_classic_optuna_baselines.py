import argparse
from types import SimpleNamespace
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent.utils import ConfigParser, resolve_config_path
import scripts.experiments.run_classic_optuna_baselines as classic_baselines
from scripts.experiments.run_comparison_experiments import (
    default_search_space,
    model_defaults,
)
from scripts.experiments.run_classic_optuna_baselines import (
    make_sampler_and_pruner,
    selected_variants,
    variants_for_suite,
    ClassicBaselineVariant,
    metric_observation,
    run_variant,
    validate_experiment_setup,
)


def test_classic_suite_contains_requested_external_baselines() -> None:
    names = [item.name for item in variants_for_suite("classic")]

    assert names == ["random_search", "tpe", "tpe_successive_halving", "bohb"]


def test_selected_variants_filters_and_overrides_trials_and_budgets() -> None:
    args = argparse.Namespace(
        suite="classic",
        only="random_search,bohb",
        n_trials=3,
        budgets_json='[{"stage":"tiny","epochs":1,"data_fraction":0.1}]',
    )

    variants = selected_variants(args)

    assert [item.name for item in variants] == ["random_search", "bohb"]
    assert all(item.n_trials == 3 for item in variants)
    assert all(item.budgets[0]["stage"] == "tiny" for item in variants)


def test_bohb_variant_uses_optuna_hyperband_pruner() -> None:
    optuna = pytest.importorskip("optuna")
    bohb = next(item for item in variants_for_suite("classic") if item.name == "bohb")

    sampler, pruner = make_sampler_and_pruner(bohb, seed=7)

    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert isinstance(pruner, optuna.pruners.HyperbandPruner)


def test_tpe_variant_uses_tpe_sampler_without_pruning() -> None:
    optuna = pytest.importorskip("optuna")
    tpe = next(item for item in variants_for_suite("classic") if item.name == "tpe")

    sampler, pruner = make_sampler_and_pruner(tpe, seed=7)

    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert isinstance(pruner, optuna.pruners.NopPruner)


def test_classic_baseline_script_dry_run_writes_plan(tmp_path: Path) -> None:
    script = Path("scripts/experiments/run_classic_optuna_baselines.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "smoke",
            "--only",
            "random_search,tpe,tpe_successive_halving,bohb",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--comparison-id",
            "classic_test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(
        (tmp_path / "classic_baseline_plan.json").read_text(encoding="utf-8")
    )
    assert plan["comparison_id"] == "classic_test"
    assert plan["note"].startswith("Classic Optuna baselines")
    assert [item["name"] for item in plan["variants"]] == [
        "random_search",
        "tpe",
        "tpe_successive_halving",
        "bohb",
    ]
    assert "classic_baseline_plan.json" in completed.stdout


@pytest.mark.parametrize("model_family", ["ecapa_tdnn", "resnet", "xvector"])
def test_all_supported_models_pass_default_baseline_preflight(
    model_family: str,
) -> None:
    config_path = resolve_config_path(model_defaults(model_family)["config_path"])
    config_data = ConfigParser(str(config_path)).load_config(resolve_references=True)
    args = argparse.Namespace(
        model_family=model_family,
        task_type="speaker_verification",
        implementation="speechbrain",
        runner="speechbrain",
    )

    bundle = validate_experiment_setup(
        args,
        variants_for_suite("smoke"),
        default_search_space(model_family),
        config_data,
    )

    assert bundle.model.model_family == model_family


def test_metric_observation_does_not_mix_unrelated_metrics() -> None:
    metrics = {"eer": 1.2, "min_dcf": 0.2, "valid_error_rate": 0.08}

    assert metric_observation(metrics, "best_error_rate") == (0.08, "valid_error_rate")
    assert metric_observation({"eer": 1.2}, "best_error_rate") == (None, None)
    assert metric_observation(metrics, "eer") == (1.2, "eer")


def test_variant_records_timing_best_metrics_artifacts_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("optuna")

    class FakeModel:
        def validate_parameters(self, parameters: dict) -> None:
            assert parameters["lr"] > 0

    class FakeRunner:
        def run_training(self, _config_path: str, overrides: dict) -> dict:
            return {
                "status": "success",
                "metrics": {"best_error_rate": float(overrides["lr"])},
                "output_folder": overrides["output_folder"],
                "model_paths": [str(Path(overrides["output_folder"]) / "best.ckpt")],
                "train_log_path": str(
                    Path(overrides["output_folder"]) / "train_log.txt"
                ),
            }

    bundle = SimpleNamespace(model=FakeModel(), runner=FakeRunner())
    monkeypatch.setattr(
        classic_baselines,
        "validate_experiment_setup",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(
        classic_baselines,
        "collect_training_result",
        lambda _runner, raw, _output, _experiment: raw,
    )
    args = argparse.Namespace(
        task_type="speaker_verification",
        model_family="ecapa_tdnn",
        implementation="speechbrain",
        runner="speechbrain",
        seed=3,
        metric_mode="min",
        comparison_id="resume_test",
        primary_metric="best_error_rate",
        config_path=Path("configs/train_ecapa_tdnn.yaml"),
        data_folder=str(tmp_path),
        device=None,
        precision=None,
        eval_precision=None,
    )
    variant = ClassicBaselineVariant(
        "random_search",
        "random",
        "none",
        n_trials=1,
        budgets=[
            {"stage": "tiny", "epochs": 1, "data_fraction": 0.5},
            {"stage": "full", "epochs": 2, "data_fraction": 1.0},
        ],
    )
    search_space = {
        "parameters": [
            {"name": "lr", "parameter_type": "float", "low": 0.01, "high": 0.02}
        ],
        "constraints": [],
    }

    first = run_variant(
        args=args,
        variant=variant,
        search_space=search_space,
        output_dir=tmp_path,
        config_data={},
    )
    resumed = run_variant(
        args=args,
        variant=variant,
        search_space=search_space,
        output_dir=tmp_path,
        config_data={},
    )

    assert first["status"] == "success"
    assert first["best_trial_number"] == 0
    assert first["best_metrics"]["best_error_rate"] == first["best_value"]
    assert first["best_model_paths"][0].endswith("best.ckpt")
    assert first["training_duration_seconds"] >= 0
    assert first["started_at"] and first["finished_at"]
    assert len(first["stage_runs"]) == 2
    assert (
        first["stage_runs"][0]["output_folder"]
        == first["stage_runs"][1]["output_folder"]
    )
    assert len(resumed["trials"]) == 1
    assert (tmp_path / "random_search" / "optuna_study.sqlite3").exists()
    assert (tmp_path / "random_search" / "variant_result.json").exists()
