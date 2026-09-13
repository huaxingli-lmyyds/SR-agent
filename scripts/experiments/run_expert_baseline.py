#!/usr/bin/env python3
"""Train a fixed expert YAML once per seed, then evaluate its locked checkpoint.

No agents, optimizers, Study, Campaign, or parameter selection are involved.
Default execution trains and validates; --evaluate-test is a separate phase.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import statistics
import sys
import tempfile
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These helpers are lazy with respect to GPU, agent, and optimizer dependencies.
from scripts.experiments import run_hpo_benchmark as protocol

DEFAULT_CONFIG = ROOT / "configs/experiments/expert_ecapa.json"
KIND = "expert_baseline"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config", help="Expert run definition; defaults to expert_ecapa.json"
    )
    result.add_argument(
        "--train-config",
        help="Expert training YAML; epochs and hyperparameters are preserved",
    )
    result.add_argument(
        "--validation-config", help="Matching speaker-verification YAML"
    )
    result.add_argument("--data-folder")
    result.add_argument("--validation-pairs")
    result.add_argument("--test-pairs")
    result.add_argument(
        "--seeds",
        help="Optional repetitions; otherwise use JSON seeds or the training YAML seed",
    )
    result.add_argument("--output-dir")
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect fixed configuration without training or output writes",
    )
    modes = result.add_mutually_exclusive_group()
    modes.add_argument(
        "--resume",
        action="store_true",
        help="Retry incomplete/failed stages with frozen inputs",
    )
    modes.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate locked checkpoints only; never train or choose a winner",
    )
    return result


def build_plan(args):
    from agent.utils import ConfigParser

    definition = Path(args.config or DEFAULT_CONFIG).resolve()
    config = protocol.read_json(definition)
    allowed = {
        "model_family",
        "implementation",
        "runner",
        "train_config",
        "validation_config",
        "seeds",
        "data_prep_seed",
        "device",
        "precision",
        "eval_precision",
    }
    if set(config) - allowed:
        raise ValueError(
            f"unsupported expert settings (no search/budget overrides): {sorted(set(config) - allowed)}"
        )
    inputs = {"definition": str(definition)}
    for key in ("train_config", "validation_config"):
        inputs[key] = str(
            protocol.project_path(getattr(args, key) or config[key])
        )
        config[key] = inputs[key]
    train = ConfigParser(inputs["train_config"]).load_config()
    epochs = train.get("number_of_epochs")
    protocol.positive_int(epochs, "expert YAML number_of_epochs")
    if (
        isinstance(train.get("epoch_counter"), dict)
        and train["epoch_counter"].get("limit") != epochs
    ):
        raise ValueError(
            "epoch_counter.limit must agree with expert number_of_epochs"
        )
    seeds = (
        [int(s) for s in args.seeds.split(",")]
        if args.seeds
        else config.get("seeds")
    )
    seeds = [train.get("seed")] if seeds is None else seeds
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int or not 0 <= s < 2**32 - 1 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "seeds must be unique nonnegative integers below 2**32-1"
        )
    prep_seed = config.get("data_prep_seed", 0)
    if type(prep_seed) is not int or not 0 <= prep_seed < 2**32 - 1:
        raise ValueError("invalid data_prep_seed")
    config.update(seeds=seeds, data_prep_seed=prep_seed, full_epochs=epochs)
    for key in ("validation_pairs", "test_pairs"):
        value = getattr(args, key)
        if value:
            inputs[key] = str(Path(value).resolve())
        elif not args.dry_run:
            raise ValueError(f"--{key.replace('_', '-')} is required")
    if "validation_pairs" in inputs and "test_pairs" in inputs:
        _, validation = protocol.read_pairs(inputs["validation_pairs"])
        _, test = protocol.read_pairs(inputs["test_pairs"])
        if validation & test:
            raise ValueError("validation/test speakers overlap")
    data_folder = (
        str(Path(args.data_folder).resolve()) if args.data_folder else None
    )
    if (not data_folder and not args.dry_run) or (
        data_folder and not Path(data_folder).is_dir()
    ):
        raise ValueError(
            "--data-folder must name an existing extracted dataset"
        )
    return {
        "kind": KIND,
        "schema_version": protocol.SCHEMA_VERSION,
        "created_at": protocol.now(),
        "config": config,
        "inputs": inputs,
        "data_folder": data_folder,
        "input_sha256": {k: protocol.file_hash(v) for k, v in inputs.items()},
        "run_order": [
            {"run_id": f"expert_seed_{seed}", "variant": KIND, "seed": seed}
            for seed in seeds
        ],
        "expert_parameters": {
            k: train.get(k)
            for k in (
                "number_of_epochs",
                "lr",
                "batch_size",
                "margin",
                "weight_decay",
            )
        },
        "protocol": {
            "training_data_fraction": 1.0,
            "parameter_search": False,
            "llm": False,
            "checkpoint_selection": "runner/recipe policy (SpeechBrain: minimum classification ErrorRate); never verification/test scores",
            "evaluation_metrics": ["eer", "min_dcf"],
            "speechbrain_metric_convention": "EER ratio in [0,1] and minDCF*100; backend defaults c_miss=c_fa=1, p_target=0.01",
        },
        "warnings": [
            "The default YAML is the repository reference recipe, not a verified globally optimal expert configuration.",
            "Epochs come from the expert YAML, NOT the five-group full_epochs; budgets may differ.",
            "Speaker exclusions, data preparation seed, output paths, and augmentation storage follow the isolated comparison protocol.",
        ],
    }


def preflight(plan):
    from agent.core.adapters import resolve_adapter_bundle
    from agent.utils import ConfigParser

    config = plan["config"]
    bundle = resolve_adapter_bundle(
        "speaker_verification",
        config["model_family"],
        config.get("implementation", "speechbrain"),
        config.get("runner", "speechbrain"),
    )
    inputs = plan.get("frozen_inputs", plan["inputs"])
    bundle.model.validate_config(
        ConfigParser(inputs["train_config"]).load_config()
    )
    if config.get("runner", "speechbrain") == "speechbrain":
        protocol.validate_speechbrain_inputs({**plan, "frozen_inputs": inputs})
        import torch
        from agent.runners.speechbrain_dependency import require_speechbrain
        from agent.runners.speechbrain_backend import _resolve_run_opts

        require_speechbrain()
        _resolve_run_opts(torch, run_options(plan))
    return bundle.runner


def run_options(plan):
    return {
        k: plan["config"][k]
        for k in ("device", "precision", "eval_precision")
        if plan["config"].get(k) is not None
    }


def prepare_run(plan, item, output):
    run_dir = output / "runs" / item["run_id"]
    if not (run_dir / "run_inputs.json").exists() and any(
        (run_dir / name).exists()
        for name in ("training", "training.json", "attempts.jsonl")
    ):
        raise ValueError(
            "existing training state has no frozen run_inputs.json; refusing to recreate inputs"
        )
    return run_dir, protocol.prepare_run(plan, item, run_dir)


def check_checkpoint(training):
    path = Path(training["checkpoint"])
    if protocol.checkpoint_hash(path) != training["checkpoint_sha256"]:
        raise ValueError("locked expert checkpoint changed")
    return path


def training_stage(plan, runner, run_dir, metadata):
    from agent.runners import collect_training_result

    output = run_dir / "training"
    overrides = {"output_folder": str(output), "_run_opts": run_options(plan)}
    raw = runner.run_training(metadata["paths"]["train"], overrides)
    if raw.get("status") != "success":
        raise RuntimeError(raw.get("error") or "expert training failed")
    collected = collect_training_result(runner, raw, output, run_dir)
    metrics = collected.get("metrics") or {}
    if (
        not protocol.finite(metrics.get("final_epoch"))
        or metrics["final_epoch"] != plan["config"]["full_epochs"]
    ):
        raise ValueError(
            "training did not confirm all expert epochs (missing/mismatched final_epoch)"
        )
    paths = collected.get("model_paths") or []
    if len(paths) != 1:
        raise ValueError("training must return exactly one selected checkpoint")
    checkpoint = Path(paths[0]).resolve()
    if not checkpoint.is_relative_to(output.resolve()):
        raise ValueError(
            "checkpoint is outside this expert run's training directory"
        )
    if plan["config"].get("runner", "speechbrain") == "speechbrain":
        if checkpoint.is_file() and checkpoint.name == "embedding_model.ckpt":
            checkpoint = checkpoint.parent
        if not (checkpoint / "embedding_model.ckpt").is_file():
            raise ValueError(
                "selected SpeechBrain checkpoint has no embedding_model.ckpt"
            )
    return {
        "status": "success",
        "metrics": metrics,
        "training_config_sha256": metadata["sha256"][
            metadata["paths"]["train"]
        ],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": protocol.checkpoint_hash(checkpoint),
        "runtime_overrides": overrides,
    }


def validate_evaluation_metrics(metrics, split):
    for key in ("eer", "min_dcf"):
        if not protocol.finite(metrics.get(key)) or metrics[key] < 0:
            raise ValueError(f"invalid/missing {split} metric: {key}")
    if metrics["eer"] > 1:
        raise ValueError("EER must be a ratio in [0,1]")


def evaluation_stage(plan, runner, run_dir, metadata, training, split):
    from agent.utils import resolve_evaluation_metrics

    checkpoint = check_checkpoint(training)
    output = run_dir / (
        "validation_output" if split == "validation" else "test_output"
    )
    overrides = {"output_folder": str(output), "_run_opts": run_options(plan)}
    if plan["config"].get("runner", "speechbrain") == "speechbrain":
        # Also bind hardcoded YAML pretrainer paths, not only <pretrain_path> refs.
        overrides["pretrainer"] = {
            "paths": {
                "embedding_model": str(checkpoint / "embedding_model.ckpt")
            }
        }
    raw = runner.run_evaluation(
        metadata["paths"][split],
        str(checkpoint),
        plan["data_folder"],
        overrides,
    )
    if raw.get("status") != "success":
        raise RuntimeError(raw.get("error") or f"expert {split} failed")
    metrics = resolve_evaluation_metrics(raw)
    validate_evaluation_metrics(metrics, split)
    return {
        "status": "success",
        "metrics": metrics,
        "evaluation_config_sha256": metadata["sha256"][
            metadata["paths"][split]
        ],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": training["checkpoint_sha256"],
        "runtime_overrides": overrides,
        "scores_path": raw.get("scores_path"),
    }


def execute_stage(run_dir, stage, action):
    path = run_dir / f"{stage}.json"
    if path.exists():
        previous = protocol.read_json(path)
        if previous.get("status") == "success":
            return previous
    attempt = uuid4().hex
    event = {"stage": stage, "attempt_id": attempt}
    protocol.append_event(run_dir, "attempts", {**event, "event": "started"})
    started = perf_counter()
    result = {"status": "interrupted"}
    try:
        result = action()
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        result["duration_seconds"] = perf_counter() - started
        protocol.append_event(
            run_dir, "attempts", {**event, **result, "event": "finished"}
        )
    protocol.write_json(path, result)
    return result


def export_results(plan, output, *, heldout=False):
    rows = []
    for item in plan["run_order"]:
        run_dir = output / "runs" / item["run_id"]
        stage = "test" if heldout else "validation"
        train_path, eval_path = (
            run_dir / "training.json",
            run_dir / f"{stage}.json",
        )
        train = protocol.read_json(train_path) if train_path.exists() else {}
        evaluation = protocol.read_json(eval_path) if eval_path.exists() else {}
        events = protocol.read_events(run_dir / "attempts.jsonl")
        finished = [e for e in events if e.get("event") == "finished"]
        status = (
            "success"
            if train.get("status") == evaluation.get("status") == "success"
            else "failed"
            if "failed" in (train.get("status"), evaluation.get("status"))
            else "incomplete"
        )
        if status == "success":
            try:
                validate_evaluation_metrics(
                    evaluation.get("metrics") or {}, stage
                )
                metadata = protocol.read_json(run_dir / "run_inputs.json")
                if (
                    not Path(train["checkpoint"])
                    .resolve()
                    .is_relative_to((run_dir / "training").resolve())
                    or train.get("training_config_sha256")
                    != metadata["sha256"][metadata["paths"]["train"]]
                    or evaluation.get("evaluation_config_sha256")
                    != metadata["sha256"][metadata["paths"][stage]]
                    or evaluation.get("checkpoint") != train["checkpoint"]
                    or evaluation.get("checkpoint_sha256")
                    != train["checkpoint_sha256"]
                ):
                    raise ValueError("mismatched training/evaluation ownership")
            except (ValueError, KeyError, OSError) as exc:
                status = "failed"
                evaluation = {
                    **evaluation,
                    "error": f"invalid saved result: {exc}",
                }
        rows.append(
            {
                **item,
                "status": status,
                "split": stage,
                "epochs": plan["config"]["full_epochs"],
                "eer": (evaluation.get("metrics") or {}).get("eer")
                if status == "success"
                else None,
                "min_dcf": (evaluation.get("metrics") or {}).get("min_dcf")
                if status == "success"
                else None,
                "checkpoint": train.get("checkpoint"),
                "error": evaluation.get("error") or train.get("error"),
                "training_attempts": sum(
                    e.get("event") == "started" and e["stage"] == "training"
                    for e in events
                ),
                "training_seconds": sum(
                    e["duration_seconds"]
                    for e in finished
                    if e["stage"] == "training"
                ),
                "evaluation_seconds": sum(
                    e["duration_seconds"]
                    for e in finished
                    if e["stage"] == stage
                ),
                "cost_may_be_incomplete": sum(
                    e.get("event") == "started" for e in events
                )
                != len(finished)
                or any(run_dir.glob("*.partial")),
            }
        )
    prefix = "heldout_" if heldout else ""
    protocol.write_json(output / f"{prefix}results.json", rows)
    with (output / f"{prefix}results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "variant": KIND,
        "split": "test" if heldout else "validation",
        "planned": len(rows),
        "success": sum(r["status"] == "success" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "incomplete": sum(r["status"] == "incomplete" for r in rows),
        "note": "All fixed-config seeds are reported; no best seed/model is selected using verification scores.",
    }
    for metric in ("eer", "min_dcf"):
        values = [
            r[metric]
            for r in rows
            if r["status"] == "success" and protocol.finite(r[metric])
        ]
        summary[metric] = {
            "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
        }
    protocol.write_json(output / f"{prefix}summary.json", summary)
    return rows


def main(argv=None):
    args = parser().parse_args(argv)
    restoring = args.resume or args.evaluate_test
    if restoring and (
        args.dry_run
        or any(
            getattr(args, k)
            for k in (
                "config",
                "train_config",
                "validation_config",
                "data_folder",
                "validation_pairs",
                "test_pairs",
                "seeds",
            )
        )
    ):
        raise ValueError(
            "resume/test accepts only --output-dir and the mode flag"
        )
    if args.dry_run:
        print(json.dumps(build_plan(args), ensure_ascii=False, indent=2))
        return 0
    if not args.output_dir:
        raise ValueError("--output-dir is required")
    output = Path(args.output_dir).resolve()
    with ExitStack() as stack:
        stack.enter_context(protocol.exclusive_run(output))
        manifest = output / "expert_plan.json"
        if restoring:
            plan = protocol.read_json(manifest)
            if plan.get("kind") != KIND:
                raise ValueError("not an expert baseline plan")
            protocol.verify_frozen(plan, output)
        else:
            if any(
                (output / name).exists()
                for name in (
                    "expert_plan.json",
                    "benchmark_plan.json",
                    "inputs",
                    "runs",
                    "assets.json",
                )
            ):
                raise ValueError(
                    "output already contains an experiment; use --resume or a new directory"
                )
            plan = build_plan(args)
        if plan["config"].get("runner", "speechbrain") == "speechbrain":
            stack.enter_context(
                protocol.exclusive_run(
                    Path(tempfile.gettempdir())
                    / "sr-agent-speechbrain-benchmark"
                )
            )
        runner = preflight(plan)
        if not restoring:
            protocol.freeze_inputs(plan, output)
            plan["environment"]["source_sha256"][
                str(Path(__file__).relative_to(ROOT))
            ] = protocol.file_hash(__file__)
            protocol.write_json(manifest, plan)
        if args.evaluate_test and not all(
            (output / "runs" / i["run_id"] / "validation.json").exists()
            for i in plan["run_order"]
        ):
            raise ValueError(
                "finish training and validation for all seeds before --evaluate-test"
            )
        protocol.prepare_assets(plan, output)
        try:
            for item in plan["run_order"]:
                run_dir, metadata = prepare_run(plan, item, output)
                print(
                    f"{'Testing' if args.evaluate_test else 'Running'} {item['run_id']}",
                    flush=True,
                )
                if args.evaluate_test:
                    training = protocol.read_json(run_dir / "training.json")
                else:
                    training = execute_stage(
                        run_dir,
                        "training",
                        lambda: training_stage(plan, runner, run_dir, metadata),
                    )
                if training.get("status") != "success":
                    continue
                checkpoint = check_checkpoint(training)
                if (
                    not checkpoint.resolve().is_relative_to(
                        (run_dir / "training").resolve()
                    )
                    or training.get("training_config_sha256")
                    != metadata["sha256"][metadata["paths"]["train"]]
                    or (training.get("metrics") or {}).get("final_epoch")
                    != plan["config"]["full_epochs"]
                    or not protocol.finite(
                        (training.get("metrics") or {}).get("final_epoch")
                    )
                ):
                    raise ValueError(
                        "training record does not belong to this expert run/configuration"
                    )
                stage = "test" if args.evaluate_test else "validation"
                evaluation = execute_stage(
                    run_dir,
                    stage,
                    lambda: evaluation_stage(
                        plan, runner, run_dir, metadata, training, stage
                    ),
                )
                if evaluation.get("status") == "success":
                    validate_evaluation_metrics(
                        evaluation.get("metrics") or {}, stage
                    )
                    if (
                        evaluation.get("checkpoint_sha256")
                        != training["checkpoint_sha256"]
                        or evaluation.get("checkpoint")
                        != training["checkpoint"]
                        or evaluation.get("evaluation_config_sha256")
                        != metadata["sha256"][metadata["paths"][stage]]
                    ):
                        raise ValueError(
                            "evaluation does not belong to the locked checkpoint/configuration"
                        )
                export_results(plan, output, heldout=args.evaluate_test)
        except KeyboardInterrupt:
            export_results(plan, output, heldout=args.evaluate_test)
            print(
                "Interrupted; resume the frozen training run with --resume, or retry --evaluate-test.",
                file=sys.stderr,
            )
            return 130
        rows = export_results(plan, output, heldout=args.evaluate_test)
        print(f"Expert baseline results: {output}")
        return int(any(r["status"] != "success" for r in rows))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        print(f"Expert baseline error: {exc}", file=sys.stderr)
        raise SystemExit(2)
