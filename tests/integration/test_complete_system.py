"""Run real coordination, data processing and HPO with only the GPU runner replaced."""

import json
from pathlib import Path

import pytest

from agent.agents.orchestrator import CoordinatorAgent
from agent.runners import RUNNER_ADAPTERS
from tests.fakes import FakeRunnerAdapter


@pytest.mark.parametrize("failure", [False, True])
def test_complete_system_isolated_records_and_failure_status(
    tmp_path, minimal_config, dataset_dir, monkeypatch, failure
):
    class Runner(FakeRunnerAdapter):
        def run_training(self, config_path, overrides):
            if failure:
                return {"status": "failed", "error": "RuntimeError: training backend failure"}
            return super().run_training(config_path, overrides)

    monkeypatch.setitem(RUNNER_ADAPTERS, "system_fake", Runner())
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("1 id1/a/1.wav id1/b/2.wav\n0 id1/a/1.wav id2/a/1.wav\n")
    output = tmp_path / "system_run"
    coordinator = CoordinatorAgent(
        config_path=str(minimal_config), output_dir=str(output),
        runner="system_fake", implementation="fake", verbose=False,
    )
    result = coordinator.run(
        context={
            "data_folder": str(dataset_dir), "controller_mode": "fixed",
            "strategy": "tpe", "sampler_config": {"n_startup_trials": 1},
            "search_space": {"parameters": [
                {"name": "lr", "parameter_type": "categorical", "choices": [0.001, 0.0015]},
            ]},
            "budgets": [{"stage": "full", "epochs": 1}],
            "runtime_options": {
                "device": "cpu", "verification_config": str(minimal_config),
                "validation_pairs": str(pairs), "training_exclusion_pairs": str(pairs),
            },
        },
        budget={"max_training_runs": 2, "max_retries": 0, "candidate_batch_size": 1},
    )
    assert result.status == ("failed" if failure else "success")
    assert len(result.task_results) == 2
    records = {}
    for kind in ("manage", "dp", "hpo"):
        paths = list((output / kind).glob("*/experiment_record.json"))
        assert len(paths) == 1
        records[kind] = json.loads(paths[0].read_text())
    assert coordinator.memory_service.root_dir == output / "memory"
    assert (output / "memory" / "episodes.jsonl").is_file()
    if failure:
        assert "training backend failure" in records["manage"]["error"]
        assert records["manage"]["status"] == "failed"
    else:
        study = next((output / "hpo").glob("*/hpo_study/study.json"))
        assert len(json.loads(study.read_text())["trial_ids"]) == 2
    lifecycle = records["dp"]["extensions"]["data_lifecycle"]
    assert "file_hashes" not in lifecycle["published_version"]
    version = json.loads(Path(lifecycle["dataset_version_path"]).read_text())
    assert lifecycle["file_hash_count"] == len(version["file_hashes"]) == 1
    assert version["quality_decision"]["training_allowed"]
