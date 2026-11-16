"""Validate an experiment matrix and emit existing HPO comparison requests; no training."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))


def validate(matrix):
    seeds = matrix["seeds"]
    if not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    hours = matrix.get("max_gpu_hours_per_arm_seed")
    if hours is not None and (isinstance(hours, bool) or not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours <= 0):
        raise ValueError("GPU budget must be finite and positive")
    count = matrix["max_total_training_runs"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("training run limit must be a positive integer")
    budgets = matrix["budgets"]
    epochs = [b["epochs"] for b in budgets]
    if not epochs or any(isinstance(e, bool) or not isinstance(e, int) or e <= 0 for e in epochs) or epochs != sorted(set(epochs)):
        raise ValueError("epochs must be strictly increasing positive integers")
    if any(b.get("data_fraction") != 1.0 for b in budgets):
        raise ValueError("this same-data continuation experiment requires data_fraction=1.0")
    roles = [matrix.get(k) for k in ("validation_pairs", "final_test_pairs")]
    if all(roles) and Path(roles[0]).resolve() == Path(roles[1]).resolve():
        raise ValueError("validation and final test must be separate files")


def build(matrix):
    from scripts.experiments.run_hpo_optimization_comparison import plans
    validate(matrix)
    missing = []
    hashes = {}
    file_fields = ("training_manifest", "verification_config", "validation_pairs", "training_exclusion_pairs", "final_test_pairs")
    for key in file_fields:
        value = matrix.get(key)
        if not value or not Path(value).is_file():
            missing.append(key)
        else:
            hashes[key] = hashlib.sha256(Path(value).read_bytes()).hexdigest()
    if not matrix.get("data_folder") or not Path(matrix["data_folder"]).is_dir():
        missing.append("data_folder")
    if matrix.get("max_gpu_hours_per_arm_seed") is None:
        missing.append("max_gpu_hours_per_arm_seed")
    requests, previews = {}, {}
    for seed in matrix["seeds"]:
        config = matrix.get("train_configs_by_seed", {}).get(str(seed))
        if not config or not Path(config).is_file():
            missing.append(f"train_configs_by_seed.{seed}")
        else:
            from agent.utils import ConfigParser
            saved_seed = ConfigParser(config).load_config(resolve_references=True).get("seed")
            if saved_seed != seed:
                raise ValueError(f"training config seed {saved_seed!r} does not match {seed}")
            hashes[f"train_config.{seed}"] = hashlib.sha256(Path(config).read_bytes()).hexdigest()
        request = {
            "config_path": config, "enable_llm_advisor": True,
            "context": {
                "data_folder": matrix.get("data_folder"), "hpo_seed": seed,
                "primary_metric": "eer", "metric_mode": "min", "controller_mode": "llm",
                "strategy": "successive_halving", "candidate_strategy": "tpe",
                "budgets": deepcopy(matrix["budgets"]), "search_space": deepcopy(matrix["search_space"]),
                "runtime_options": {
                    "device": "cuda:0", "precision": "fp32",
                    "verification_config": matrix.get("verification_config"),
                    "validation_pairs": matrix.get("validation_pairs"),
                    "training_exclusion_pairs": matrix.get("training_exclusion_pairs"),
                },
            },
            "budget": {
                "max_gpu_hours": matrix.get("max_gpu_hours_per_arm_seed"),
                "max_training_runs": matrix["max_total_training_runs"],
                "max_total_training_runs": matrix["max_total_training_runs"],
                "max_studies": 1, "max_retries": 0, "candidate_batch_size": 3,
            },
            "full_execution_options": {"max_concurrent_trials": 1, "confirmation_top_k": 2, "confirmation_seeds": 3},
        }
        requests[str(seed)] = request
        previews[str(seed)] = {name: arm["preview"] for name, arm in plans(request).items()}
    return {
        "mode": "preview_only", "ready_for_input_review": not missing,
        "missing_inputs": missing, "input_hashes": hashes,
        "experiment_matrix": matrix["experiments"], "hpo_previews": previews,
        "requests": requests,
        "notes": [
            "No training or external LLM was called; no performance results exist.",
            "File hashes do not prove speaker/audio disjointness or bind every training audio file; run data protocol audit before training.",
            "Equal budget ceilings do not prove equal consumed cost; compare matched GPU-cost horizons.",
            "Six 12-hour confirmation runs alone are roughly 72 GPU-hours under the user's timing, excluding changed evaluation costs; reservation counts are not time guarantees.",
            "E2 requires explicit role-specific advisors; E3 is a bundle comparison, not causal attribution to one feature.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "experiments/matrix.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/generated")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args(argv)
    # Keep every generated artifact inside the user-selected delivery root.
    output = args.output.resolve()
    output.relative_to(ROOT)
    result = build(json.loads(args.matrix.read_text(encoding="utf-8")))
    output.mkdir(parents=True, exist_ok=True)
    (output / "preview.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for seed, request in result["requests"].items():
        (output / f"request_seed_{seed}.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("mode", "ready_for_input_review", "missing_inputs")}, ensure_ascii=False, indent=2))
    return 2 if args.check_inputs and result["missing_inputs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
