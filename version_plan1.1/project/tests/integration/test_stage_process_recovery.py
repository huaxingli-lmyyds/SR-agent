"""Real OS processes exercise ownership and crash accounting without a GPU."""
import json
import os
import subprocess
import sys
import time

import pytest

from agent.hpo.resources import CampaignResourceBudget
from agent.runners.stage_pool import identity, terminate_owned


def test_reused_pid_is_never_signalled():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        owner = identity(child.pid)
        terminate_owned({**owner, "created_at": owner["created_at"] - 1})
        assert child.poll() is None
        terminate_owned(owner)
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_recovery_terminates_worker_before_settling_and_is_idempotent(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    path = tmp_path / "ledger.json"
    limits = {"max_gpu_hours": 1}
    try:
        ledger = CampaignResourceBudget(path, limits=limits)
        ledger.start_stage(experiment_id="e", trial_id="t", attempt=1,
                           stage="training", gpu_count=1, stage_id="stable", max_seconds=10)
        ledger.attach_worker("stable", identity(child.pid))
        ledger.close()
        resumed = CampaignResourceBudget(path, limits=limits, resume=True)
        try:
            child.wait(timeout=5)
            assert not resumed.data["active_stages"]
            assert resumed.data["gpu_seconds"] >= 10
            assert resumed.data["stages"][0]["estimated"]
            resumed.finish_stage(stage_id="stable", seconds=1)
            assert len(resumed.data["stages"]) == 1
        finally:
            resumed.close()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_uncertain_worker_exit_blocks_recovery_and_keeps_reservation(tmp_path, monkeypatch):
    path = tmp_path / "ledger.json"
    ledger = CampaignResourceBudget(path, limits={})
    ledger.start_stage(experiment_id="e", trial_id="t", attempt=1, stage="training", gpu_count=1, stage_id="s")
    ledger.attach_worker("s", identity(os.getpid()))
    ledger.close()
    def denied(owner):
        raise PermissionError("cannot establish worker identity")
    monkeypatch.setattr("agent.runners.stage_pool.terminate_owned", denied)
    with pytest.raises(PermissionError):
        CampaignResourceBudget(path, limits={}, resume=True)
    assert "s" in json.loads(path.read_text())["active_stages"]


def test_completed_result_recovers_actual_cost(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = CampaignResourceBudget(path, limits={"max_gpu_hours": 1}, clock=lambda: 100)
    ledger.start_stage(experiment_id="e", trial_id="t", attempt=1, stage="training", gpu_count=2, stage_id="s", max_seconds=20)
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"raw": {"status": "success"}, "finished_at": 105}))
    ledger.attach_worker("s", {"pid": os.getpid(), "created_at": -1, "result_path": str(result)})
    ledger.close()
    recovered = CampaignResourceBudget(path, limits={}, resume=True, clock=lambda: 1000)
    try:
        assert recovered.data["gpu_seconds"] == 10
        assert not recovered.data["stages"][0]["estimated"]
    finally:
        recovered.close()
