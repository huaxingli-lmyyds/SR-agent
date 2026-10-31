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


def test_system_failure_blocks_other_groups_and_survives_resume(setup_benchmark, monkeypatch):
    env = setup_benchmark
    evaluations = []

    def broken_evaluation(*args, **kwargs):
        evaluations.append(kwargs.get("model_path"))
        raise AttributeError("Can't get local object 'dataio_prep.<locals>.audio_pipeline'")

    monkeypatch.setattr(env.runner, "run_evaluation", broken_evaluation)
    assert benchmark.main(env.args) == 1
    assert len(env.runner.training) == 1
    assert len(evaluations) == 1
    rows = benchmark.read_json(env.output / "results.json")
    assert len(rows) == 1
    assert rows[0]["failure_guard"]["status"] == "open"
    assert rows[0]["failure_guard"]["failure_stage"] == "evaluation"
    assert "AttributeError" in rows[0]["failure_guard"]["error_summary"]
    assert Path(evaluations[0]).is_file()

    assert benchmark.main(["--resume", "--output-dir", str(env.output)]) == 1
    assert len(env.runner.training) == 1
    assert len(evaluations) == 1


def test_candidate_resource_failure_does_not_stop_other_groups(setup_benchmark, monkeypatch):
    env = setup_benchmark
    original = env.runner.run_evaluation
    calls = []

    def evaluation(*args, **kwargs):
        calls.append(kwargs.get("model_path"))
        if len(calls) == 1:
            return {"status": "failed", "error": "CUDA out of memory"}
        return original(*args, **kwargs)

    monkeypatch.setattr(env.runner, "run_evaluation", evaluation)
    assert benchmark.main(env.args + ["--only", "random_search,tpe"]) == 0
    assert len(env.runner.training) == 8
    rows = benchmark.read_json(env.output / "results.json")
    assert len(rows) == 2
    assert all(row["status"] == "success" for row in rows)
    assert all(row["failure_guard"]["status"] == "closed" for row in rows)


def test_system_failure_survives_late_summary_error(setup_benchmark, monkeypatch):
    env = setup_benchmark
    original = benchmark.run_variant

    def broken_evaluation(*args, **kwargs):
        raise AttributeError("broken shared evaluation pipeline")

    def interrupted_summary(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("summary finalization failed")

    monkeypatch.setattr(env.runner, "run_evaluation", broken_evaluation)
    monkeypatch.setattr(benchmark, "run_variant", interrupted_summary)
    assert benchmark.main(env.args) == 1
    assert len(env.runner.training) == 1
    rows = benchmark.read_json(env.output / "results.json")
    assert rows[0]["failure_guard"]["status"] == "open"
    agent_result = benchmark.read_json(Path(rows[0]["run_dir"]) / "agent_result.json")
    assert agent_result["summary"]["campaign"]["status"] == "failed"
    assert agent_result["summary"]["failure_guard"]["status"] == "open"
    assert agent_result["experiment_ids"]["hpo"]
    assert benchmark.main(["--resume", "--output-dir", str(env.output)]) == 1
    assert len(env.runner.training) == 1


def test_shared_error_outside_scheduler_stops_benchmark(setup_benchmark, monkeypatch):
    env = setup_benchmark

    def broken_setup(*args, **kwargs):
        raise ModuleNotFoundError("shared backend dependency unavailable")

    monkeypatch.setattr(benchmark, "run_variant", broken_setup)
    assert benchmark.main(env.args) == 1
    assert env.runner.training == []
    rows = benchmark.read_json(env.output / "results.json")
    assert len(rows) == 1
    assert rows[0]["failure_guard"]["reason_code"] == "benchmark_system_error"


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


@pytest.mark.parametrize("resource", ["gpu", "wall"])
def test_benchmark_resource_limits_reach_real_training_and_evaluation(
    setup_benchmark, monkeypatch, resource,
):
    from time import time
    import agent.agents.hpo_agent as module
    from agent.hpo.resources import CampaignResourceBudget

    env = setup_benchmark
    cfg = benchmark.read_json(env.config)
    cfg.update(device="cuda:0", ddp_devices=[0, 1, 2])
    cfg.update({"max_gpu_hours": 4 / 3600} if resource == "gpu" else {"max_wall_seconds": 2})
    benchmark.write_json(env.config, cfg)
    now = [time() + 100]
    monkeypatch.setattr(module, "CampaignResourceBudget", lambda *a, **k: CampaignResourceBudget(*a, clock=lambda: now[0], **k))
    original_training = env.runner.run_training
    original_evaluation = env.runner.run_evaluation

    def train(*args, **kwargs):
        assert args[1]["_hpo_max_duration_seconds"] > 0
        result = original_training(*args, **kwargs)
        now[0] += 1
        return result

    def evaluate(*args, **kwargs):
        assert kwargs["overrides"]["_hpo_max_duration_seconds"] > 0
        result = original_evaluation(*args, **kwargs)
        now[0] += 1
        return result

    monkeypatch.setattr(env.runner, "run_training", train)
    monkeypatch.setattr(env.runner, "run_evaluation", evaluate)
    assert benchmark.main(env.args + ["--only", "random_search"]) == 1
    assert len(env.runner.training) == len(env.runner.evaluations) == 1
    result = benchmark.read_json(env.output / "results.json")[0]
    assert result["study_status"] == "cancelled"
    assert result["confirmation_satisfied"] is False
    assert result["resource_budget"]["gpu_seconds"] == 4  # three GPUs training, one GPU verification
    assert result["resource_budget"]["stop_reason"]
    agent_result = benchmark.read_json(Path(result["run_dir"]) / "agent_result.json")
    assert agent_result["experiment_ids"]["hpo"] == result["experiment_id"]
    assert not (Path(result["run_dir"]) / "locked_winner.json").exists()
    ledgers = list((Path(result["run_dir"]) / "hpo" / "campaign_resources").glob("*.json"))
    assert len(ledgers) == 1
    assert len(benchmark.read_json(ledgers[0])["stages"]) == 2
    # Ordinary benchmark resume must not start further work for an exhausted run.
    assert benchmark.main(["--resume", "--output-dir", str(env.output)]) == 1
    assert len(env.runner.training) == 1


def test_different_output_roots_share_real_runner_exclusion(
    setup_benchmark, tmp_path, monkeypatch
):
    env = setup_benchmark
    cfg = benchmark.read_json(env.config)
    cfg["runner"] = "speechbrain"
    benchmark.write_json(env.config, cfg)
    monkeypatch.setattr(benchmark.tempfile, "gettempdir", lambda: str(tmp_path))
    # This fixture uses a fake dataset; isolate the lock contract from real
    # VoxCeleb inventory preparation.
    monkeypatch.setattr(benchmark, "freeze_inputs", lambda *args: None)
    with benchmark.exclusive_run(tmp_path / "sr-agent-speechbrain-benchmark"):
        with pytest.raises(OSError):
            benchmark.main(env.args + ["--only", "default_parameters"])
    assert not env.runner.training and not env.calls


def test_evaluate_on_another_gpu_records_assignment_without_changing_training(setup_benchmark, monkeypatch):
    env = setup_benchmark
    first = "GPU-00000000-0000-0000-0000-000000000001"
    second = "GPU-00000000-0000-0000-0000-000000000002"
    cfg = benchmark.read_json(env.config)
    cfg.update(parallel_single_gpu=True, device="cuda:0")
    benchmark.write_json(env.config, cfg)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", first)
    assert benchmark.main(env.args + ["--only", "default_parameters"]) == 0
    before = (env.output / "benchmark_plan.json").read_bytes()
    assert benchmark.main(["--evaluate-test", "--output-dir", str(env.output), "--evaluation-gpu", second]) == 0
    assert (env.output / "benchmark_plan.json").read_bytes() == before
    rows = benchmark.read_json(env.output / "heldout_results.json")
    assert rows[0]["evaluation_gpu_uuid"] == second
    assert rows[0]["training_gpu_assignment"] == {"physical_gpu_uuid": first}
    audit = benchmark.read_events(env.output / "evaluation_assignments.jsonl")
    assert audit[0]["evaluation_gpu_uuid"] == second
    assert len(env.runner.training) == 1


def test_hypothesis_batches_keep_budget_and_feed_tpe(
    setup_benchmark, monkeypatch
):
    """Exercise real agent parsing, gates, scheduler, promotions and reporting."""
    from agent.hpo.strategies import STRATEGIES

    env = setup_benchmark
    config = benchmark.read_json(env.config)
    config.update(
        hypothesis_probes=True,
        sh_initial_trials=9,
        candidate_batch_size=3,
        promotion_limits=[3, 1],
        full_epochs=6,
        budgets=[
            {"stage": "screen", "epochs": 1, "data_fraction": 1.0},
            {"stage": "promote", "epochs": 3, "data_fraction": 1.0},
            {"stage": "full", "epochs": 6, "data_fraction": 1.0},
        ],
    )
    benchmark.write_json(env.config, config)
    calls, history_seen = [], []
    tpe = STRATEGIES.get("tpe")
    original_suggest = tpe.suggest

    def record_history(*args, **kwargs):
        history_seen.append(list(kwargs.get("history") or []))
        return original_suggest(*args, **kwargs)

    monkeypatch.setattr(tpe, "suggest", record_history)

    class ProbeModel:
        def invoke(self, prompt):
            payload = json.loads(prompt)
            calls.append(payload)
            status = payload["context"]["feedback"]["hypothesis_probe"]
            assert status["eligible"] and status["maximum_candidates"] == 3
            anchor = status["observed_trials"][0]
            base = anchor["parameters"]
            lr, margin = (base["lr"] + 0.0005) / 2, (base["margin"] + 0.15) / 2
            proposal = payload["schema"]
            proposal.update(
                action="switch_strategy",
                requested_sampler="agent_proposal",
                confidence=0.9,
                hypotheses=[
                    {
                        "id": "lr_margin",
                        "claim": "LR and margin changes interact locally.",
                        "anchor_trial_id": anchor["trial_id"],
                        "evidence_trial_ids": [anchor["trial_id"]],
                        "probe_design": "interaction",
                        "tested_parameters": ["lr", "margin"],
                        "expected_signal": {"eer": "lower"},
                        "falsification_criterion": {
                            "metric": "eer",
                            "comparison": "interaction_gain",
                            "minimum_improvement": 0.001,
                        },
                    }
                ],
                candidate_proposals=[
                    {
                        "parameters": {**base, **change},
                        "hypothesis_id": "lr_margin",
                        "role": "joint"
                        if len(change) == 2
                        else "single_factor_control",
                        "rationale": "Compare with the observed anchor and the other controls.",
                        "expected_signal": {"eer": "lower"},
                        "confidence": 0.9,
                    }
                    for change in [
                        {"lr": lr},
                        {"margin": margin},
                        {"lr": lr, "margin": margin},
                    ]
                ],
            )
            return SimpleNamespace(
                content=json.dumps(proposal),
                usage_metadata={"input_tokens": 10, "output_tokens": 5},
            )

    monkeypatch.setattr(
        AdvisoryAgentBase, "llm", property(lambda self: ProbeModel())
    )
    assert (
        benchmark.main(env.args + ["--only", "tpe_successive_halving,system"])
        == 0
    )
    rows = benchmark.read_json(env.output / "results.json")
    assert len(rows) == 2 and all(row["status"] == "success" for row in rows)
    assert len(env.runner.training) == 26
    row = next(row for row in rows if row["variant"] == "system")
    service = HPOService(ExperimentTracker(Path(row["run_dir"]) / "hpo"))
    study = service.load_study(row["experiment_id"])
    trials = service.list_trials(row["experiment_id"])
    by_id = {t.trial_id: t for t in trials}
    initial = [by_id[tid] for tid in study.trial_ids if by_id[tid].rung == 0]
    assert [t.candidate_source for t in initial] == ["sampler:tpe"] * 3 + [
        "sampler:agent_proposal"
    ] * 3 + ["sampler:tpe"] * 3
    assert [sum(t.rung == rung for t in trials) for rung in range(3)] == [
        9,
        3,
        1,
    ]
    from agent.hpo.records import search_space_from_dict

    assert (
        study.search_space.to_dict()
        == search_space_from_dict(config["search_space"]).to_dict()
    )
    assert (
        len(calls) == row["advisor_requests"] == row["llm_valid_proposals"] == 1
    )
    assert not row["controller_degraded"]
    assert row["hypothesis_probes"]["agent_candidate_count"] == 3
    assert (
        row["hypothesis_probes"]["confirmed_best_metric"]
        == row["validation_eer"]
    )
    assert len(row["hypothesis_probes"]["outcomes"]) == 1
    assert any(
        sum(t.candidate_source == "sampler:agent_proposal" for t in history)
        == 3
        for history in history_seen
    )
    # A completed resume consumes neither additional LLM calls nor training runs.
    assert benchmark.main(["--resume", "--output-dir", str(env.output)]) == 0
    assert len(calls) == 1 and len(env.runner.training) == 26


def test_hypothesis_llm_format_failure_preserves_tpe_budget(
    setup_benchmark, monkeypatch
):
    env = setup_benchmark
    config = benchmark.read_json(env.config)
    config["hypothesis_probes"] = True
    benchmark.write_json(env.config, config)
    calls = []

    class MalformedModel:
        def invoke(self, prompt):
            calls.append(prompt)
            return SimpleNamespace(
                content="not a JSON proposal", usage_metadata={}
            )

    monkeypatch.setattr(
        AdvisoryAgentBase, "llm", property(lambda self: MalformedModel())
    )
    assert benchmark.main(env.args + ["--only", "system"]) == 0
    row = benchmark.read_json(env.output / "results.json")[0]
    assert row["status"] == "success" and row["controller_degraded"]
    assert row["hypothesis_probes"]["agent_candidate_count"] == 0
    assert len(env.runner.training) == 8
    assert (
        len(calls) == 2
    )  # one eligible consultation plus one format regeneration
    service = HPOService(ExperimentTracker(Path(row["run_dir"]) / "hpo"))
    assert all(
        t.candidate_source == "sampler:tpe"
        for t in service.list_trials(row["experiment_id"])
        if t.rung == 0
    )


def test_cross_stage_mode_uses_real_scheduler_and_bounded_llm_reviews(
    setup_benchmark, monkeypatch
):
    env = setup_benchmark
    cfg = benchmark.read_json(env.config)
    cfg.update(
        cross_stage_probes=True,
        hypothesis_probes=False,
        sh_initial_trials=9,
        candidate_batch_size=3,
        promotion_limits=[3, 1],
        full_epochs=6,
        budgets=[
            {"stage": "screen", "epochs": 1, "data_fraction": 1.0},
            {"stage": "promote", "epochs": 3, "data_fraction": 1.0},
            {"stage": "full", "epochs": 6, "data_fraction": 1.0},
        ],
    )
    benchmark.write_json(env.config, cfg)
    calls = []

    class Model:
        def invoke(self, prompt):
            payload = json.loads(prompt)
            calls.append(payload)
            ctx = payload["context"]
            info = ctx["cross_stage"]
            status = info["opportunity"]
            result = payload["schema"]
            result.update(
                confidence=0.8,
                reason="Compare observed configurations across the two budgets.",
            )
            observations = {
                o["trial_id"]: {**o, "parameters": line["parameters"]}
                for line in info["evidence"]["lineages"]
                for o in line["observations"]
            }
            if ctx["phase"] == "paired_promotion":
                ranked = sorted(
                    (observations[tid] for tid in status["eligible_trial_ids"]),
                    key=lambda t: t["metric"],
                )
                a, b = ranked[0], ranked[-1]
                result.update(
                    action="paired_promotion",
                    hypothesis="The initial ranking may reverse at three epochs.",
                    anchor_trial_id=a["trial_id"],
                    challenger_trial_id=b["trial_id"],
                    evidence_trial_ids=[a["trial_id"], b["trial_id"]],
                    minimum_late_advantage=0.0001,
                )
            else:
                outcome = status["pair_outcome"]
                winner = observations[outcome["target_winner_trial_id"]]
                p = next(
                    p
                    for p in ctx["study"]["search_space"]["parameters"]
                    if p["name"] == "margin"
                )
                quarter = (p["high"] - p["low"]) / 4
                region = (
                    {"low": p["low"] + quarter, "high": p["high"]}
                    if winner["parameters"]["margin"] >= p["low"] + quarter
                    else {"low": p["low"], "high": p["high"] - quarter}
                )
                result.update(
                    action="focus_region",
                    region={"margin": region},
                    evidence_trial_ids=outcome["source_trial_ids"]
                    + outcome["child_trial_ids"],
                )
            return SimpleNamespace(
                content=json.dumps(result),
                usage_metadata={"input_tokens": 10, "output_tokens": 5},
            )

    monkeypatch.setattr(
        AdvisoryAgentBase, "llm", property(lambda self: Model())
    )
    assert (
        benchmark.main(env.args + ["--only", "tpe_successive_halving,system"])
        == 0
    )
    rows = benchmark.read_json(env.output / "results.json")
    assert all(row["status"] == "success" for row in rows)
    assert len(env.runner.training) == 26
    row = next(r for r in rows if r["variant"] == "system")
    assert row["advisor_requests"] == row["llm_valid_proposals"] == 2
    assert not row["controller_degraded"]
    assert row["cross_stage"]["focused_candidate_count"] == 3
    assert len(calls) == 2 and [p["context"]["phase"] for p in calls] == [
        "paired_promotion",
        "region_review",
    ]
    assert row["cross_stage"]["pair_outcome"]["status"] in {
        "supported_observation",
        "not_supported",
    }
    service = HPOService(ExperimentTracker(Path(row["run_dir"]) / "hpo"))
    study = service.load_study(row["experiment_id"])
    trials = [
        service.load_trial(row["experiment_id"], tid) for tid in study.trial_ids
    ]
    assert [t.rung for t in trials] == [0] * 3 + [1] * 2 + [0] * 6 + [1, 2]
    from agent.hpo.records import search_space_from_dict

    assert (
        study.search_space.to_dict()
        == search_space_from_dict(cfg["search_space"]).to_dict()
    )
    assert benchmark.main(["--resume", "--output-dir", str(env.output)]) == 0
    assert len(calls) == 2 and len(env.runner.training) == 26
