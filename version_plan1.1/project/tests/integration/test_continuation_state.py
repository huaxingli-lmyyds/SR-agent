"""Real SpeechBrain/PyTorch state continuity on a small deterministic CPU model."""
from pathlib import Path
import random
from types import SimpleNamespace
import pytest


def test_full_epoch_state_restores_optimizer_scheduler_rng_and_model(tmp_path):
    torch = pytest.importorskip("torch")
    from agent.runners.speechbrain_dependency import require_speechbrain
    try:
        require_speechbrain()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    from speechbrain.utils.checkpoints import Checkpointer
    from speechbrain.utils.epoch_loop import EpochCounter
    from speechbrain.nnet.schedulers import CyclicLRScheduler
    from agent.runners.continuation_state import capture_rng, restore_rng, save_state, restore_state

    def brain(folder):
        torch.manual_seed(123); random.seed(123)
        import numpy as np
        np.random.seed(123)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        scheduler = CyclicLRScheduler(base_lr=.001, max_lr=.01, step_size=4)
        counter = EpochCounter(4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        checkpointer = Checkpointer(folder / "save", {"model": model, "optimizer": optimizer, "lr_annealing": scheduler, "counter": counter})
        return SimpleNamespace(model=model, optimizer=optimizer, scheduler=scheduler,
            optimizer_step=0, step=0, scaler=scaler, checkpointer=checkpointer,
            hparams=SimpleNamespace(output_folder=str(folder), hpo_continuation={"capture": True},
                hpo_training_signature={"test": "same_data"}, epoch_counter=counter))

    def epoch(b):
        import numpy as np
        for _ in range(3):
            x = torch.randn(4, 3) + float(np.random.random()) + random.random()
            loss = b.model(x).square().mean()
            b.optimizer.zero_grad(); loss.backward(); b.optimizer.step()
            b.scheduler.on_batch_end(b.optimizer); b.optimizer_step += 1
        b.hparams.epoch_counter.current += 1

    full = brain(tmp_path / "full")
    for _ in range(4): epoch(full)
    expected_rng = capture_rng()
    first = brain(tmp_path / "first")
    for _ in range(2): epoch(first)
    save_state(first, 2)
    second = brain(tmp_path / "second")
    second.hparams.hpo_continuation = {"enabled": True, "manifest": str(tmp_path / "first" / "continuation" / "epoch_2" / "manifest.json")}
    restore_state(second)
    assert second.hparams.epoch_counter.current == 2
    assert second.optimizer_step == 6
    for _ in range(2): epoch(second)
    assert all(torch.equal(a, b) for a,b in zip(full.model.parameters(), second.model.parameters()))
    assert full.optimizer.param_groups[0]["lr"] == second.optimizer.param_groups[0]["lr"]
    assert torch.equal(capture_rng()["torch"], expected_rng["torch"])
