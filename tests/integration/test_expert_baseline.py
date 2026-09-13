"""Expert baseline exercises real stage persistence without agents or GPU work."""

import builtins
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent.runners import RUNNER_ADAPTERS
from agent.utils import ConfigParser
from scripts.experiments import run_expert_baseline as expert


@pytest.fixture
def setup_expert(tmp_path, monkeypatch):
    train, evaluation = tmp_path / "expert.yaml", tmp_path / "eval.yaml"
    train.write_text(
        "seed: 41\nnumber_of_epochs: 7\nembedding_model: fake\nclassifier: fake\n"
        "output_folder: shared\nsave_folder: shared/save\ntrain_log: shared/log\n"
        "lr: 0.00123\nbatch_size: 13\nmargin: 0.27\nweight_decay: 0.000004\n"
        "step_size: 456\ncustom_expert_setting: 19\n",
        encoding="utf-8",
    )
    evaluation.write_text(
        "seed: 55\nverification_file: forbidden_default_test\n",
        encoding="utf-8",
    )
    validation, test = tmp_path / "val.txt", tmp_path / "test.txt"
    validation.write_text(
        "1 val1/x/a.wav val1/x/b.wav\n0 val1/x/a.wav val2/x/b.wav\n",
        encoding="utf-8",
    )
    test.write_text(
        "1 test1/x/a.wav test1/x/b.wav\n0 test1/x/a.wav test2/x/b.wav\n",
        encoding="utf-8",
    )
    data = tmp_path / "dataset"
    data.mkdir()
    definition = tmp_path / "expert.json"
    config = expert.protocol.read_json(expert.DEFAULT_CONFIG)
    config.update(
        train_config=str(train),
        validation_config=str(evaluation),
        runner="expert_fake",
    )
    expert.protocol.write_json(definition, config)
    output = tmp_path / "expert_run"
    args = [
        "--config",
        str(definition),
        "--data-folder",
        str(data),
        "--validation-pairs",
        str(validation),
        "--test-pairs",
        str(test),
        "--output-dir",
        str(output),
    ]

    class Runner:
        runner = "expert_fake"
        supported_implementations = {"speechbrain"}
        supported_model_families = {"ecapa_tdnn"}

        def __init__(self):
            self.training = []
            self.evaluations = []
            self.interrupt_train = False
            self.interrupt_eval = False
            self.invalid_metric = None
            self.failure = None
            self.epochs = 7
            self.missing_checkpoint = False
            self.foreign_checkpoint = False

        def run_training(self, config_path, overrides):
            cfg = ConfigParser(config_path).load_config()
            self.training.append((cfg, overrides))
            folder = Path(overrides["output_folder"])
            folder.mkdir(parents=True, exist_ok=True)
            checkpoint = folder / "embedding.ckpt"
            checkpoint.write_text("fixed expert checkpoint", encoding="utf-8")
            if self.interrupt_train:
                self.interrupt_train = False
                raise KeyboardInterrupt()
            if self.failure:
                return {"status": "failed", "error": self.failure}
            if self.foreign_checkpoint:
                checkpoint = tmp_path / "foreign.ckpt"
                checkpoint.write_text("foreign", encoding="utf-8")
            return {
                "status": "success",
                "metrics": {
                    "final_epoch": self.epochs,
                    "valid_error_rate": 0.1,
                },
                "model_paths": []
                if self.missing_checkpoint
                else [str(checkpoint)],
            }

        def run_evaluation(self, config_path, model_path, data_path, overrides):
            cfg = ConfigParser(config_path).load_config()
            self.evaluations.append((cfg, model_path, overrides))
            if self.interrupt_eval:
                self.interrupt_eval = False
                raise KeyboardInterrupt()
            metrics = {"eer": 0.025, "min_dcf": 0.4}
            if self.invalid_metric:
                metrics[self.invalid_metric] = float("nan")
            return {"status": "success", "metrics": metrics}

    runner = Runner()
    monkeypatch.setitem(RUNNER_ADAPTERS, runner.runner, runner)
    original_import = builtins.__import__

    def no_search_imports(name, *pos, **kw):
        if name.startswith(
            (
                "agent.agents",
                "agent.hpo",
                "agent.tools",
                "optuna",
                "langchain",
                "langgraph",
            )
        ):
            raise AssertionError(
                f"expert baseline must not import agent/search dependencies: {name}"
            )
        return original_import(name, *pos, **kw)

    monkeypatch.setattr(builtins, "__import__", no_search_imports)
    return SimpleNamespace(
        args=args,
        output=output,
        runner=runner,
        train=train,
        validation=validation,
        config=definition,
        evaluation=evaluation,
    )


def test_fixed_expert_parameters_and_separate_test_without_hpo(setup_expert):
    env = setup_expert
    assert expert.main(env.args) == 0
    assert len(env.runner.training) == len(env.runner.evaluations) == 1
    cfg, overrides = env.runner.training[0]
    assert {
        k: cfg[k]
        for k in (
            "seed",
            "number_of_epochs",
            "lr",
            "batch_size",
            "margin",
            "weight_decay",
            "step_size",
            "custom_expert_setting",
        )
    } == {
        "seed": 41,
        "number_of_epochs": 7,
        "lr": 0.00123,
        "batch_size": 13,
        "margin": 0.27,
        "weight_decay": 0.000004,
        "step_size": 456,
        "custom_expert_setting": 19,
    }
    assert set(overrides) == {"output_folder", "_run_opts"}
    assert cfg["verification_file"].endswith("training_exclusions.txt")
    assert env.runner.evaluations[0][0]["verification_file"].endswith(
        "validation_pairs.txt"
    )
    assert not list(env.output.rglob("study.json")) and not list(
        env.output.rglob("experiments_history.json")
    )
    assert expert.main(["--resume", "--output-dir", str(env.output)]) == 0
    assert len(env.runner.training) == len(env.runner.evaluations) == 1
    assert (
        expert.main(["--evaluate-test", "--output-dir", str(env.output)]) == 0
    )
    assert len(env.runner.training) == 1 and len(env.runner.evaluations) == 2
    assert env.runner.evaluations[-1][0]["verification_file"].endswith(
        "test_pairs.txt"
    )
    assert env.runner.evaluations[0][1] == env.runner.evaluations[1][1]
    assert (
        expert.main(["--evaluate-test", "--output-dir", str(env.output)]) == 0
    )
    assert len(env.runner.evaluations) == 2
    row = expert.protocol.read_json(env.output / "heldout_results.json")[0]
    assert (
        row["status"] == "success"
        and row["eer"] == 0.025
        and row["min_dcf"] == 0.4
    )


@pytest.mark.parametrize("stage", ["train", "eval"])
def test_resume_reuses_training_directory_and_skips_successful_stages(
    setup_expert, stage
):
    env = setup_expert
    setattr(env.runner, f"interrupt_{stage}", True)
    assert expert.main(env.args) == 130
    env.train.write_text("changed_original: true\n", encoding="utf-8")
    assert expert.main(["--resume", "--output-dir", str(env.output)]) == 0
    assert len(env.runner.training) == (2 if stage == "train" else 1)
    assert {call[0]["number_of_epochs"] for call in env.runner.training} == {7}
    assert len({call[1]["output_folder"] for call in env.runner.training}) == 1
    assert len(env.runner.evaluations) == (2 if stage == "eval" else 1)


@pytest.mark.parametrize("metric", ["eer", "min_dcf"])
def test_invalid_evaluation_is_retried_without_retraining(setup_expert, metric):
    env = setup_expert
    env.runner.invalid_metric = metric
    assert expert.main(env.args) == 1
    row = expert.protocol.read_json(env.output / "results.json")[0]
    assert row["status"] == "failed" and row["eer"] is None
    env.runner.invalid_metric = None
    assert expert.main(["--resume", "--output-dir", str(env.output)]) == 0
    assert len(env.runner.training) == 1 and len(env.runner.evaluations) == 2


@pytest.mark.parametrize(
    "problem", ["failure", "epochs", "missing_checkpoint", "foreign_checkpoint"]
)
def test_unconfirmed_training_never_reaches_evaluation(setup_expert, problem):
    env = setup_expert
    setattr(
        env.runner,
        problem,
        {"failure": "simulated OOM", "epochs": 2}.get(problem, True),
    )
    assert expert.main(env.args) == 1
    assert not env.runner.evaluations
    row = expert.protocol.read_json(env.output / "results.json")[0]
    assert row["status"] == "failed"
    with pytest.raises(ValueError, match="finish training"):
        expert.main(["--evaluate-test", "--output-dir", str(env.output)])


def test_repeated_seeds_and_separate_output_roots_do_not_share_state(
    setup_expert,
):
    env = setup_expert
    assert expert.main(env.args + ["--seeds", "11,22,33"]) == 0
    assert [c[0]["seed"] for c in env.runner.training] == [11, 22, 33]
    assert len({c[0]["prep_cache_root"] for c in env.runner.training}) == 3
    assert {c[0]["data_prep_seed"] for c in env.runner.training} == {0}
    other = env.output.with_name("another_expert_run")
    assert expert.main(env.args[:-1] + [str(other)]) == 0
    assert len({c[1]["output_folder"] for c in env.runner.training}) == 4
    summary = expert.protocol.read_json(env.output / "summary.json")
    assert summary["success"] == 3 and summary["eer"]["mean"] == 0.025


@pytest.mark.parametrize(
    "mutation", ["checkpoint", "snapshot", "missing_manifest", "copied_root"]
)
def test_resume_and_test_reject_tampered_or_foreign_state(
    setup_expert, mutation
):
    import shutil

    env = setup_expert
    assert expert.main(env.args) == 0
    run_dir = env.output / "runs/expert_seed_41"
    output = env.output
    if mutation == "checkpoint":
        training = expert.protocol.read_json(run_dir / "training.json")
        Path(training["checkpoint"]).write_text("changed", encoding="utf-8")
    elif mutation == "snapshot":
        (run_dir / "train.yaml").write_text("changed: true", encoding="utf-8")
    elif mutation == "missing_manifest":
        (run_dir / "run_inputs.json").unlink()
    else:
        output = env.output.with_name("copied")
        shutil.copytree(env.output, output)
    with pytest.raises(ValueError):
        expert.main(["--resume", "--output-dir", str(output)])
    assert len(env.runner.training) == 1 and len(env.runner.evaluations) == 1


def test_overwrite_overlap_and_search_config_rejected(setup_expert):
    env = setup_expert
    config = expert.protocol.read_json(env.config)
    config["search_space"] = {}
    expert.protocol.write_json(env.config, config)
    with pytest.raises(ValueError, match="no search/budget overrides"):
        expert.main(env.args)
    config.pop("search_space")
    expert.protocol.write_json(env.config, config)
    assert expert.main(env.args) == 0
    with pytest.raises(ValueError, match="already contains"):
        expert.main(env.args)
    with pytest.raises(ValueError, match="resume/test accepts"):
        expert.main(
            ["--resume", "--output-dir", str(env.output), "--seeds", "9"]
        )


def test_speechbrain_overrides_bind_explicit_checkpoint_file(tmp_path):
    checkpoint = tmp_path / "CKPT+selected"
    checkpoint.mkdir()
    (checkpoint / "embedding_model.ckpt").write_text(
        "selected", encoding="utf-8"
    )
    training = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": expert.protocol.checkpoint_hash(checkpoint),
    }
    captured = []

    def evaluate(*args):
        captured.append(args)
        return {"status": "success", "eer": 0.03, "min_dcf": 0.3}

    result = expert.evaluation_stage(
        {"config": {"runner": "speechbrain"}, "data_folder": "dataset"},
        SimpleNamespace(run_evaluation=evaluate),
        tmp_path,
        {
            "paths": {"validation": "validation.yaml"},
            "sha256": {"validation.yaml": "frozen-eval"},
        },
        training,
        "validation",
    )
    assert result["metrics"] == {"eer": 0.03, "min_dcf": 0.3}
    assert captured[0][1] == str(checkpoint)
    assert captured[0][3]["pretrainer"]["paths"]["embedding_model"] == str(
        checkpoint / "embedding_model.ckpt"
    )


def test_heldout_interruption_does_not_retrain_or_revalidate(setup_expert):
    env = setup_expert
    assert expert.main(env.args) == 0
    env.runner.interrupt_eval = True
    assert (
        expert.main(["--evaluate-test", "--output-dir", str(env.output)]) == 130
    )
    assert (
        expert.main(["--evaluate-test", "--output-dir", str(env.output)]) == 0
    )
    assert len(env.runner.training) == 1 and len(env.runner.evaluations) == 3
    assert all(
        call[0]["verification_file"].endswith("test_pairs.txt")
        for call in env.runner.evaluations[1:]
    )


@pytest.mark.parametrize("stage", ["training", "validation"])
def test_record_from_another_seed_is_rejected(setup_expert, stage):
    import shutil

    env = setup_expert
    assert expert.main(env.args + ["--seeds", "11,22"]) == 0
    shutil.copyfile(
        env.output / f"runs/expert_seed_11/{stage}.json",
        env.output / f"runs/expert_seed_22/{stage}.json",
    )
    with pytest.raises(ValueError, match="does not belong"):
        expert.main(["--resume", "--output-dir", str(env.output)])
    assert len(env.runner.training) == len(env.runner.evaluations) == 2


def test_expert_and_benchmark_share_device_lock(
    setup_expert, tmp_path, monkeypatch
):
    env = setup_expert
    cfg = expert.protocol.read_json(env.config)
    cfg["runner"] = "speechbrain"
    expert.protocol.write_json(env.config, cfg)
    monkeypatch.setattr(expert.tempfile, "gettempdir", lambda: str(tmp_path))
    with expert.protocol.exclusive_run(
        tmp_path / "sr-agent-speechbrain-benchmark"
    ):
        with pytest.raises(OSError, match="execution lock"):
            expert.main(env.args)
    assert not env.runner.training and not env.runner.evaluations


def test_overlapping_speakers_fail_before_training(setup_expert):
    env = setup_expert
    env.validation.write_text(
        "1 test1/x/a.wav test1/x/b.wav\n0 test1/x/a.wav val2/x/b.wav\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="speakers overlap"):
        expert.main(env.args)
    assert not env.runner.training


@pytest.mark.parametrize("corruption", ["eer", "min_dcf", "ownership"])
def test_export_never_counts_invalid_saved_results_as_success(
    setup_expert, corruption
):
    env = setup_expert
    assert expert.main(env.args) == 0
    path = env.output / "runs/expert_seed_41/validation.json"
    record = expert.protocol.read_json(path)
    if corruption == "ownership":
        record["evaluation_config_sha256"] = "different-protocol"
    else:
        record["metrics"][corruption] = None
    expert.protocol.write_json(path, record)
    plan = expert.protocol.read_json(env.output / "expert_plan.json")
    rows = expert.export_results(plan, env.output)
    assert rows[0]["status"] == "failed" and rows[0]["eer"] is None
    assert (
        expert.protocol.read_json(env.output / "summary.json")["success"] == 0
    )


def test_default_preview_is_read_only_and_does_not_import_agents(tmp_path):
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('torch', 'speechbrain', 'agent.agents', 'agent.hpo', 'optuna', 'langchain', 'langgraph')):
            raise AssertionError('unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, Guard())
from scripts.experiments.run_expert_baseline import main
raise SystemExit(main(sys.argv[1:]))
"""
    output = tmp_path / "preview"
    result = subprocess.run(
        [sys.executable, "-c", code, "--dry-run", "--output-dir", str(output)],
        cwd=expert.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["config"]["full_epochs"] == 10 and plan["config"]["seeds"] == [
        1986
    ]
    assert plan["expert_parameters"]["batch_size"] == 32
    assert not output.exists()
