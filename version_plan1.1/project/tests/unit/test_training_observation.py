from types import SimpleNamespace

from agent.runners.speechbrain import SpeechBrainRunnerAdapter
from agent.runners.training_observation import scheduler_snapshot


def test_cycle_phase_uses_actual_scheduler_steps_and_omits_nonfinite_values():
    scheduler = type("CyclicLRScheduler", (), {})()
    scheduler.clr_iterations = 3845
    scheduler.step_size = 3845
    scheduler.base_lr = 1e-8
    scheduler.max_lr = 0.0017
    scheduler.current_lr = float("nan")
    scheduler.mode = "triangular"
    snapshot = scheduler_snapshot(scheduler)
    assert snapshot["cycle_phase"] == 0.5
    assert snapshot["scheduler_iterations"] == 3845
    assert "lr_after_last_batch" not in snapshot
    scheduler.clr_iterations = 23070
    assert scheduler_snapshot(scheduler)["cycle_phase"] == 0.0
    assert "cycle_phase" not in scheduler_snapshot(
        SimpleNamespace(step_size=3845, clr_iterations=3845)
    )


def test_actual_selected_checkpoint_metadata_precedes_final_epoch(tmp_path):
    output = tmp_path / "run"
    save = output / "save"
    earlier = save / "CKPT+first"
    later = save / "CKPT+last"
    earlier.mkdir(parents=True)
    later.mkdir()
    (earlier / "CKPT.yaml").write_text(
        "ErrorRate: 0.1\ntraining_observation:\n  checkpoint_epoch: 1\n  learning_rate_schedule:\n    cycle_phase: 0.5\n"
    )
    (later / "CKPT.yaml").write_text(
        "ErrorRate: 0.2\ntraining_observation:\n  checkpoint_epoch: 6\n"
    )
    (output / "train_log.txt").write_text(
        "epoch: 1, lr: 0.0017 - train loss: 0.4 - valid loss: 0.3, valid ErrorRate: 0.1\nepoch: 6, lr: 0.0001 - train loss: 0.2 - valid loss: 0.3, valid ErrorRate: 0.2\n"
    )
    adapter = SpeechBrainRunnerAdapter()
    raw = adapter.collect_training_result(
        {"status": "success"}, output, tmp_path
    )
    assert raw["model_paths"] == [str(earlier)]
    assert raw["metrics"]["final_epoch"] == 6
    assert raw["selected_checkpoint"]["checkpoint_epoch"] == 1
    result = adapter.normalize_training_result(raw)
    assert (
        result.extensions["speechbrain"]["selected_checkpoint"][
            "learning_rate_schedule"
        ]["cycle_phase"]
        == 0.5
    )


def test_legacy_checkpoint_does_not_get_an_invented_epoch(tmp_path):
    folder = tmp_path / "save" / "CKPT+legacy"
    folder.mkdir(parents=True)
    (folder / "CKPT.yaml").write_text("ErrorRate: 0.1\n")
    assert (
        SpeechBrainRunnerAdapter().collect_training_result(
            {"status": "success"}, tmp_path, tmp_path
        )["selected_checkpoint"]
        == {}
    )
