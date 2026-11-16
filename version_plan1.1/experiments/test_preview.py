import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("matrix_preview", Path(__file__).with_name("preview.py"))
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def matrix():
    return json.loads(Path(__file__).with_name("matrix.json").read_text(encoding="utf-8"))


def test_missing_inputs_produce_preview_not_fabricated_results():
    result = preview.build(matrix())
    assert not result["ready_for_input_review"]
    assert "max_gpu_hours_per_arm_seed" in result["missing_inputs"]
    assert result["mode"] == "preview_only"
    for arms in result["hpo_previews"].values():
        assert arms["legacy"]["checkpoint_evaluations_per_trial_max"] == 1
        assert arms["p0"]["checkpoint_evaluations_per_trial_max"] == 3
        assert arms["full"]["confirmation"]["reserved_runs"] == 6
        assert arms["full"]["search_total_cap"] + arms["full"]["confirmation"]["reserved_runs"] == 15


@pytest.mark.parametrize("change", [
    {"seeds": [11, 11]}, {"max_gpu_hours_per_arm_seed": -1},
    {"max_gpu_hours_per_arm_seed": float("nan")},
    {"budgets": [{"epochs": 3, "data_fraction": .25}]},
    {"validation_pairs": "same.txt", "final_test_pairs": "same.txt"},
])
def test_invalid_protocol_rejected(change):
    config = matrix()
    config.update(change)
    with pytest.raises(ValueError):
        preview.validate(config)


def test_budget_ceiling_is_shared_and_test_pairs_never_reach_search():
    config = matrix()
    config["max_gpu_hours_per_arm_seed"] = 120
    config["final_test_pairs"] = "heldout.txt"
    result = preview.build(config)
    for seed, request in result["requests"].items():
        assert request["budget"]["max_gpu_hours"] == 120
        assert request["context"]["hpo_seed"] == int(seed)
        assert "final_test_pairs" not in request["context"]["runtime_options"]
        assert request["context"]["budgets"][-1]["epochs"] == 12


def test_training_seed_must_match_sampler_seed(tmp_path):
    config = matrix()
    source = tmp_path / "train.yaml"
    source.write_text("seed: 999\n", encoding="utf-8")
    config["train_configs_by_seed"]["11"] = str(source)
    with pytest.raises(ValueError, match="seed"):
        preview.build(config)
