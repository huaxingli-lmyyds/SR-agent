from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.experiments import run_hpo_benchmark as benchmark


def config():
    return benchmark.read_json(benchmark.DEFAULT_CONFIG)


def test_five_groups_have_explicit_budgets():
    cfg = config()
    specs = [benchmark.variant_spec(cfg, name) for name in benchmark.VARIANTS]
    assert [s["max_training_runs"] for s in specs] == [1, 9, 9, 13, 13]
    assert all(s["budgets"][-1]["epochs"] == 24 for s in specs)
    assert all(s["pruner"] == "none" for s in specs[:3])
    assert specs[3]["budgets"] == specs[4]["budgets"]
    assert specs[3]["nominal_full_data_epochs"] == 42.75


@pytest.mark.parametrize(
    "change",
    [
        {"seeds": [1, 1]},
        {"seeds": [True]},
        {"full_trials": 0},
        {"full_epochs": 20},
        {"max_retries": -1},
        {"promotion_limits": [4, 1]},
        {"promotion_limits": [3]},
    ],
)
def test_invalid_config_rejected(change):
    cfg = config()
    cfg.update(change)
    with pytest.raises(ValueError):
        benchmark.validate_config(cfg)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_numeric_budget_and_domain(value):
    cfg = config()
    cfg["budgets"][0]["data_fraction"] = value
    with pytest.raises(ValueError):
        benchmark.validate_config(cfg)
    cfg = config()
    cfg["search_space"]["parameters"][0]["low"] = value
    with pytest.raises(ValueError):
        benchmark.validate_config(cfg)


def test_dry_run_is_dependency_light_and_does_not_write(tmp_path):
    destination = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(benchmark.ROOT / "scripts/experiments/run_hpo_benchmark.py"),
            "--dry-run",
            "--seeds",
            "12",
            "--output-dir",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert len(plan["run_order"]) == 5
    assert {r["seed"] for r in plan["run_order"]} == {12}
    assert not destination.exists()


def test_summary_excludes_failed_and_nonfinite_results():
    rows = [
        {
            "variant": "tpe",
            "seed": 1,
            "status": "success",
            "confirmation_satisfied": True,
            "validation_eer": 0.04,
        },
        {
            "variant": "tpe",
            "seed": 2,
            "status": "failed",
            "validation_eer": 0.0,
        },
        {
            "variant": "tpe",
            "seed": 3,
            "status": "success",
            "confirmation_satisfied": True,
            "validation_eer": float("nan"),
        },
        {
            "variant": "system",
            "seed": 1,
            "status": "success",
            "confirmation_satisfied": True,
            "validation_eer": 0.03,
        },
    ]
    result = benchmark.summarize(rows)
    assert result["by_variant"]["tpe"]["validation_eer_mean"] == 0.04
    assert result["by_variant"]["tpe"]["failures"] == 2
    assert result["paired_validation_differences"]["tpe"] == [
        {"seed": 1, "system_minus_baseline": pytest.approx(-0.01)}
    ]


def test_proposal_constraints_lock_resource_and_tool_boundaries():
    from agent.hpo import StrategyProposal
    from scripts.experiments.benchmark_runtime import (
        constrain_proposal,
        within_envelope,
    )

    space = config()["search_space"]
    narrowed = deepcopy(space)
    narrowed["parameters"][0]["low"] = 0.001
    assert within_envelope(narrowed, space)
    expanded = deepcopy(space)
    expanded["parameters"][0]["low"] = 0.00001
    assert not within_envelope(expanded, space)
    proposal = StrategyProposal(
        action="switch_strategy",
        requested_sampler="agent_proposal",
        requested_pruner="none",
        max_training_runs=999,
        budgets=[],
        initial_trial_count=99,
        promotion_limits=[9],
        candidate_proposals=[{"parameters": {"lr": 0.001}}],
        search_space=expanded,
    )
    actual = constrain_proposal(proposal, space)
    assert actual.requested_sampler is None and actual.requested_pruner is None
    assert actual.budgets is None and actual.max_training_runs is None
    assert actual.search_space is None and actual.candidate_proposals == []
    assert "requested_sampler" in actual.evidence["benchmark_blocked_fields"]
    assert constrain_proposal(None, space).action == "keep_strategy"
    assert proposal.max_training_runs == 999


def test_pairs_reject_bad_labels_and_path_traversal(tmp_path):
    path = tmp_path / "pairs.txt"
    for text in (
        "2 a/x/a.wav a/x/b.wav",
        "0 ../x/a.wav a/x/b.wav",
        "1 a/x/a.wav a/x/b.wav",
    ):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            benchmark.read_pairs(path)


def test_snapshot_keeps_hyperpyyaml_tags_and_changes_seed(tmp_path):
    from agent.utils import ConfigParser

    source = tmp_path / "source.yaml"
    dest = tmp_path / "frozen.yaml"
    source.write_text(
        "seed: 42\noutput_folder: !ref results/<seed>\nmodel: !new:some.Model {}\n",
        encoding="utf-8",
    )
    benchmark.snapshot_yaml(source, dest, {"seed": 7})
    text = dest.read_text(encoding="utf-8")
    assert "!new:some.Model" in text and "!ref" in text
    assert ConfigParser(dest).load_config()["seed"] == 7
    assert "seed: 42" in source.read_text(encoding="utf-8")


def test_output_lock_rejects_concurrent_writer(tmp_path):
    with benchmark.exclusive_run(tmp_path):
        with pytest.raises(OSError):
            with benchmark.exclusive_run(tmp_path):
                pass


def test_isolated_snapshot_rebinds_hardcoded_artifact_paths(tmp_path):
    from agent.utils import ConfigParser

    source, dest = tmp_path / "source.yaml", tmp_path / "isolated.yaml"
    source.write_text(
        "output_folder: shared\nsave_folder: shared/save\ntrain_log: shared/log\n"
        "train_annotation: shared/train.csv\ntrain_data: shared/train.csv\n"
        "checkpointer: !new:some.Checkpointer {checkpoints_dir: shared/ckpt}\n"
        "train_logger: !new:some.Logger {save_file: shared/log}\n"
        "pretrainer: !new:some.Pretrainer {collect_in: shared/pretrained}\n"
        "prepare_noise_data: !name:some.writer {}\n",
        encoding="utf-8",
    )
    benchmark.snapshot_yaml(
        source, dest, {"output_folder": str(tmp_path / "trial")}, isolate=True
    )
    cfg = ConfigParser(dest).load_config()
    assert cfg["checkpointer"]["checkpoints_dir"] == cfg["save_folder"]
    assert cfg["pretrainer"]["collect_in"] == cfg["save_folder"]
    assert cfg["train_logger"]["save_file"] == cfg["train_log"]
    assert "prepare_noise_data" not in cfg
    assert all(
        str(tmp_path / "trial") in cfg[k]
        for k in ("save_folder", "train_log", "train_annotation", "train_data")
    )
    assert "!ref <output_folder>/save" in dest.read_text(encoding="utf-8")


def test_augmentation_is_prepared_once_outside_dataset(tmp_path, monkeypatch):
    from recipes.voxceleb import augmentation_prepare

    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        path = benchmark.Path(kwargs["csv_file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "ID,duration,wav\na,1,immutable.wav\n", encoding="utf-8"
        )

    monkeypatch.setattr(
        augmentation_prepare, "prepare_dataset_from_URL", prepare
    )
    plan = {
        "frozen_inputs": {
            "train_config": str(
                benchmark.ROOT / "configs/train_ecapa_tdnn.yaml"
            )
        }
    }
    first = benchmark.prepare_assets(plan, tmp_path)
    assert len(calls) == 2 and calls[0]["URL"].startswith("https://")
    assert all(
        benchmark.Path(c["dest_folder"]).is_relative_to(tmp_path / "assets")
        for c in calls
    )
    assert benchmark.prepare_assets(plan, tmp_path) == first
    assert len(calls) == 2
    benchmark.Path(first["noise_annotation"]).write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="annotation changed"):
        benchmark.prepare_assets(plan, tmp_path)


def test_real_configs_are_compatible_but_missing_audio_fails_early(tmp_path):
    pairs = tmp_path / "pairs.txt"
    pairs.write_text(
        "1 a/x/a.wav a/x/b.wav\n0 a/x/a.wav b/x/b.wav\n", encoding="utf-8"
    )
    plan = {
        "data_folder": str(tmp_path),
        "frozen_inputs": {
            "train_config": str(
                benchmark.ROOT / "configs/train_ecapa_tdnn.yaml"
            ),
            "validation_config": str(
                benchmark.ROOT / "configs/verification_ecapa.yaml"
            ),
            "validation_pairs": str(pairs),
            "test_pairs": str(pairs),
        },
    }
    with pytest.raises(ValueError, match="missing validation_pairs audio"):
        benchmark.validate_speechbrain_inputs(plan)
    for utterance in ("a/x/a.wav", "a/x/b.wav", "b/x/b.wav"):
        path = tmp_path / "wav" / utterance
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    benchmark.validate_speechbrain_inputs(plan)
    bad = tmp_path / "bad.yaml"
    benchmark.snapshot_yaml(
        plan["frozen_inputs"]["validation_config"], bad, {"n_mels": 40}
    )
    plan["frozen_inputs"]["validation_config"] = str(bad)
    with pytest.raises(ValueError, match="feature mismatch"):
        benchmark.validate_speechbrain_inputs(plan)


def test_torn_journal_append_is_preserved_and_repaired(tmp_path):
    path = tmp_path / "attempts.jsonl"
    path.write_bytes(b'{"event":"started"}\n{"event":"fin')
    benchmark.append_event(tmp_path, "attempts", {"event": "resumed"})
    events = benchmark.read_events(path)
    assert [e["event"] for e in events] == ["started", "resumed"]
    assert events[-1]["repaired_truncated_event"] is True
    assert next(tmp_path.glob("*.partial")).read_bytes() == b'{"event":"fin'


def test_valid_journal_entry_without_newline_is_not_lost(tmp_path):
    path = tmp_path / "attempts.jsonl"
    path.write_bytes(b'{"event":"started"}')
    benchmark.append_event(tmp_path, "attempts", {"event": "finished"})
    assert len(benchmark.read_events(path)) == 2


@pytest.mark.parametrize("has_checkpoint", [True, False])
def test_executor_binds_current_checkpoint_and_validation_protocol(
    monkeypatch, has_checkpoint
):
    from agent.agents.hpo_agent import HPOAgent
    from agent.hpo import Trial, TrialBudget
    from agent.tools import evaluation_tools, training_tools

    captured = []
    monkeypatch.setattr(
        training_tools,
        "TrainModel",
        SimpleNamespace(
            invoke=lambda kwargs: json.dumps(
                {
                    "status": "success",
                    "artifacts": [
                        {"type": "checkpoint", "path": "current.ckpt"}
                    ]
                    if has_checkpoint
                    else [],
                }
            )
        ),
    )
    monkeypatch.setattr(
        evaluation_tools,
        "RunEvaluation",
        SimpleNamespace(
            invoke=lambda kwargs: (
                captured.append(kwargs)
                or json.dumps(
                    {"status": "success", "metrics": {"test": {"eer": 0.03}}}
                )
            )
        ),
    )
    agent = SimpleNamespace(
        task_type="speaker_verification",
        model_family="ecapa_tdnn",
        implementation="speechbrain",
        runner="fake",
        experiments_dir="isolated",
    )
    trial = Trial(
        "trial1",
        {"lr": 0.001},
        TrialBudget("full", epochs=2, data_fraction=1.0),
    )
    execute = HPOAgent._trial_executor(
        agent, "exp1", "data", {"verification_config": "validation.yaml"}
    )
    result = execute(trial, 1)
    if has_checkpoint:
        assert captured[0]["model_path"] == "current.ckpt"
        assert captured[0]["verification_config"] == "validation.yaml"
        assert result["cost"]["evaluated_checkpoint"] == "current.ckpt"
    else:
        assert (
            result["status"] == "failed"
            and "missing checkpoint" in result["error"]
        )
        assert not captured
