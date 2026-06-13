import json

import pytest

pytest.importorskip("langchain_core")

from agent.core.adapters import RUNNER_ADAPTERS, register_runner_adapter
from agent.tools import evaluation_tools
from agent.utils.experiment_tracker import ExperimentTracker
from tests.fakes import FakeRunnerAdapter


def test_evaluation_tool_uses_registered_runner_and_records_result(
    tmp_path,
    minimal_config,
    dataset_dir,
    monkeypatch,
) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    checkpoint = tmp_path / "fake.ckpt"
    checkpoint.write_text("fake", encoding="utf-8")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
        task={"type": "speaker_verification", "dataset": str(dataset_dir)},
        model={"family": "ecapa_tdnn", "implementation": "fake"},
        execution={"runner": "fake"},
    )
    tracker.update_hpo_experiment(
        experiment_id,
        artifacts=[{"type": "checkpoint", "name": "best", "path": str(checkpoint)}],
    )
    previous = RUNNER_ADAPTERS.get("fake")
    register_runner_adapter(FakeRunnerAdapter())
    monkeypatch.setattr(evaluation_tools, "ExperimentTracker", lambda: tracker)
    monkeypatch.setattr(
        evaluation_tools,
        "get_experiment_artifact_dir",
        lambda *args, **kwargs: tmp_path / "evaluation",
    )
    (tmp_path / "evaluation").mkdir()

    try:
        payload = json.loads(evaluation_tools._run_evaluation(experiment_id=experiment_id))
    finally:
        if previous is None:
            RUNNER_ADAPTERS.pop("fake", None)
        else:
            RUNNER_ADAPTERS["fake"] = previous

    record = tracker.get_experiment(experiment_id)
    assert payload["status"] == "success"
    assert payload["metrics"]["test"]["eer"] == 0.03
    assert record["metrics"]["test"]["min_dcf"] == 0.12
