from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from agent.agents.base_agent import AdvisoryAgentBase
from agent.hpo import HPOService
from agent.hpo.feedback import HPOFeedbackAnalyzer
from agent.runners import RUNNER_ADAPTERS, register_runner_adapter
from agent.utils import ConfigParser, ExperimentTracker
from scripts.experiments import run_hpo_benchmark as benchmark
from tests.fakes import FakeRunnerAdapter


@pytest.fixture
def setup_benchmark(tmp_path, monkeypatch):
    config = benchmark.read_json(benchmark.DEFAULT_CONFIG)
    train = tmp_path / "train.yaml"
    train.write_text(
        "seed: 999\nembedding_model: fake\nclassifier: fake\noutput_folder: fake\n"
        "lr: 0.0017\nbatch_size: 24\nmargin: 0.23\nweight_decay: 0.000003\n",
        encoding="utf-8",
    )
    evaluation = tmp_path / "eval.yaml"
    evaluation.write_text(
        "verification_file: forbidden_official_test\nseed: 999\n",
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
    config.update(
        {
            "train_config": str(train),
            "validation_config": str(evaluation),
            "runner": "benchmark_fake",
            "seeds": [7],
            "full_epochs": 2,
            "full_trials": 4,
            "sh_initial_trials": 6,
            "promotion_limits": [2],
            "candidate_batch_size": 2,
            "budgets": [
                {"stage": "screen", "epochs": 1, "data_fraction": 0.5},
                {"stage": "full", "epochs": 2, "data_fraction": 1.0},
            ],
        }
    )
    config_path = tmp_path / "config.json"
    benchmark.write_json(config_path, config)
    output = tmp_path / "benchmark"
    args = [
        "--config",
        str(config_path),
        "--data-folder",
        str(data),
        "--validation-pairs",
        str(validation),
        "--test-pairs",
        str(test),
        "--output-dir",
        str(output),
    ]

    class Runner(FakeRunnerAdapter):
        def __init__(self):
            super().__init__(runner="benchmark_fake")
            self.training = []
            self.evaluations = []
            self.interrupt_on = None

        def run_training(self, config_path, overrides):
            self.training.append(
                {
                    "config": ConfigParser(config_path).load_config(),
                    "overrides": dict(overrides),
                }
            )
            if len(self.training) == self.interrupt_on:
                raise KeyboardInterrupt()
            raw = super().run_training(config_path, overrides)
            path = Path(raw["model_paths"][0])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"fake checkpoint {len(self.training)}", encoding="utf-8"
            )
            return raw

        def run_evaluation(self, config_path, model_path, data_path, overrides):
            cfg = ConfigParser(config_path).load_config()
            self.evaluations.append(
                {
                    "config": cfg,
                    "model_path": model_path,
                    "output": overrides["output_folder"],
                }
            )
            assert Path(model_path).is_file()
            raw = super().run_evaluation(
                config_path, model_path, data_path, overrides
            )
            raw["metrics"]["eer"] = 0.02 + (len(self.training) % 3) * 0.001
            return raw

    runner = Runner()
    old = RUNNER_ADAPTERS.get(runner.runner)
    register_runner_adapter(runner)
    calls = []

    class Model:
        def invoke(self, prompt):
            calls.append(prompt)
            payload = json.loads(prompt)
            proposal = json.loads(json.dumps(payload["schema"]))
            proposal.update(
                {
                    "action": "switch_strategy",
                    "requested_sampler": "random_search"
                    if len(calls) == 1
                    else "tpe",
                    "budgets": [{"stage": "bad", "epochs": 999}],
                    "max_training_runs": 999,
                    "candidate_proposals": [
                        {
                            "parameters": {"lr": 0.001},
                            "hypothesis_id": None,
                            "role": "invalid_for_benchmark",
                            "rationale": "Exercise benchmark restrictions.",
                            "expected_signal": {"primary_metric": "lower"},
                            "confidence": 0.8,
                        }
                    ],
                    "reason_codes": ["test_restrictions"],
                    "confidence": 0.8,
                }
            )
            return SimpleNamespace(
                content=json.dumps(proposal),
                usage_metadata={"input_tokens": 10, "output_tokens": 5},
            )

    monkeypatch.setattr(
        AdvisoryAgentBase, "llm", property(lambda self: Model())
    )

    def no_rules(*args, **kwargs):
        raise AssertionError(
            "fixed benchmark must never call the default rule proposer"
        )

    monkeypatch.setattr(HPOFeedbackAnalyzer, "propose", no_rules)
    yield SimpleNamespace(
        args=args,
        output=output,
        runner=runner,
        calls=calls,
        config=config_path,
        validation=validation,
    )
    if old is None:
        RUNNER_ADAPTERS.pop(runner.runner, None)
    else:
        RUNNER_ADAPTERS[runner.runner] = old


def test_five_groups_use_real_scheduler_and_separate_test(setup_benchmark):
    env = setup_benchmark
    assert benchmark.main(env.args) == 0
    rows = benchmark.read_json(env.output / "results.json")
    assert len(rows) == 5 and all(r["status"] == "success" for r in rows)
    assert len(env.runner.training) == 1 + 4 + 4 + 8 + 8
    assert env.calls
    advisor_payloads = [json.loads(prompt) for prompt in env.calls]
    assert all(
        payload["context"]["execution_constraints"]["agent_proposal_allowed"]
        is False
        for payload in advisor_payloads
    )
    assert all(
        r["advisor_requests"] == 0 for r in rows if r["variant"] != "system"
    )
    assert all(
        e["config"]["verification_file"].endswith("validation_pairs.txt")
        for e in env.runner.evaluations
    )
    assert all(t["config"]["seed"] == 7 for t in env.runner.training)
    cache_roots = {t["config"]["prep_cache_root"] for t in env.runner.training}
    assert len(cache_roots) == 5
    assert all(Path(p).is_relative_to(env.output / "runs") for p in cache_roots)
    assert all(t["config"]["data_prep_seed"] == 0 for t in env.runner.training)
    assert all(
        t["config"]["voxceleb_source"] is None for t in env.runner.training
    )
    assert all("checkpoint" not in t["overrides"] for t in env.runner.training)
    assert (
        len({t["overrides"]["output_folder"] for t in env.runner.training})
        == 25
    )
    default = next(r for r in rows if r["variant"] == "default_parameters")
    winner = benchmark.read_json(
        Path(default["run_dir"]) / "locked_winner.json"
    )
    assert (
        winner["parameters"]["lr"] == 0.0017
    )  # actual YAML, not a hardcoded baseline
    assert all(
        e["output"].startswith(str(env.output)) for e in env.runner.evaluations
    )
    baseline_records = len(env.runner.training)
    assert benchmark.main(["--resume", "--output-dir", str(env.output)]) == 0
    assert len(env.runner.training) == baseline_records
    assert (
        benchmark.main(["--evaluate-test", "--output-dir", str(env.output)])
        == 0
    )
    assert len(env.runner.training) == baseline_records
    assert len(env.runner.evaluations) == baseline_records + 5
    assert all(
        e["config"]["verification_file"].endswith("test_pairs.txt")
        for e in env.runner.evaluations[-5:]
    )
    assert (env.output / "heldout_results.json").exists()


def test_resume_reuses_same_trial_and_frozen_config(setup_benchmark):
    env = setup_benchmark
    env.runner.interrupt_on = 2
    assert benchmark.main(env.args + ["--only", "random_search"]) == 130
    config_path = benchmark.read_json(env.config)["train_config"]
    Path(config_path).write_text("changed_source: true\n", encoding="utf-8")
    env.runner.interrupt_on = None
    assert benchmark.main(["--resume", "--output-dir", str(env.output)]) == 0
    assert len(env.runner.training) == 5
    assert (
        env.runner.training[1]["overrides"]["output_folder"]
        == env.runner.training[2]["overrides"]["output_folder"]
    )
    assert all(t["config"]["seed"] == 7 for t in env.runner.training)
    row = benchmark.read_json(env.output / "results.json")[0]
    assert row["cost_may_be_incomplete"] is True
    tracker = ExperimentTracker(Path(row["run_dir"]) / "hpo")
    assert len(HPOService(tracker).list_trials(row["experiment_id"])) == 4


def test_no_test_before_all_search_runs_and_no_overwrite(setup_benchmark):
    env = setup_benchmark
    env.runner.interrupt_on = 1
    assert benchmark.main(env.args + ["--only", "tpe"]) == 130
    with pytest.raises(ValueError, match="all planned"):
        benchmark.main(["--evaluate-test", "--output-dir", str(env.output)])
    with pytest.raises(ValueError, match="already contains"):
        benchmark.main(env.args)


def test_overlapping_validation_test_speakers_fail_before_training(
    setup_benchmark,
):
    env = setup_benchmark
    env.validation.write_text(
        "1 test1/x/a.wav test1/x/b.wav\n0 test1/x/a.wav val2/x/b.wav\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="speakers overlap"):
        benchmark.main(env.args)
    assert not env.runner.training


def test_distinct_repetitions_really_change_training_seed(setup_benchmark):
    env = setup_benchmark
    assert (
        benchmark.main(
            env.args + ["--only", "default_parameters", "--seeds", "7,8"]
        )
        == 0
    )
    assert [r["config"]["seed"] for r in env.runner.training] == [7, 8]
    assert not env.calls


def test_heldout_rejects_modified_snapshot(setup_benchmark):
    env = setup_benchmark
    assert benchmark.main(env.args + ["--only", "default_parameters"]) == 0
    row = benchmark.read_json(env.output / "results.json")[0]
    path = Path(row["run_dir"]) / "test.yaml"
    path.write_text("modified: true\n", encoding="utf-8")
    before = len(env.runner.evaluations)
    with pytest.raises(ValueError, match="snapshot changed"):
        benchmark.main(["--evaluate-test", "--output-dir", str(env.output)])
    assert len(env.runner.evaluations) == before


@pytest.mark.parametrize(
    "mutation", ["moved", "source", "version", "dependency"]
)
def test_resume_rejects_foreign_root_code_and_protocol(
    setup_benchmark, mutation
):
    import shutil

    env = setup_benchmark
    assert benchmark.main(env.args + ["--only", "default_parameters"]) == 0
    output = env.output
    expected = "source changed"
    if mutation == "moved":
        output = env.output.with_name("copied_benchmark")
        shutil.copytree(env.output, output)
        expected = "moved/copied"
    else:
        path = output / "benchmark_plan.json"
        plan = benchmark.read_json(path)
        if mutation == "source":
            key = next(iter(plan["environment"]["source_sha256"]))
            plan["environment"]["source_sha256"][key] = "not-current-code"
        elif mutation == "dependency":
            plan["environment"]["packages"]["optuna"] = "not-current-version"
            expected = "dependency versions changed"
        else:
            plan["schema_version"] = 1
            expected = "protocol version changed"
        benchmark.write_json(path, plan)
    before = len(env.runner.training)
    with pytest.raises(ValueError, match=expected):
        benchmark.main(["--resume", "--output-dir", str(output)])
    assert len(env.runner.training) == before


def test_different_output_roots_share_real_runner_exclusion(
    setup_benchmark, tmp_path, monkeypatch
):
    env = setup_benchmark
    cfg = benchmark.read_json(env.config)
    cfg["runner"] = "speechbrain"
    benchmark.write_json(env.config, cfg)
    monkeypatch.setattr(benchmark.tempfile, "gettempdir", lambda: str(tmp_path))
    with benchmark.exclusive_run(tmp_path / "sr-agent-speechbrain-benchmark"):
        with pytest.raises(OSError):
            benchmark.main(env.args + ["--only", "default_parameters"])
    assert not env.runner.training and not env.calls
