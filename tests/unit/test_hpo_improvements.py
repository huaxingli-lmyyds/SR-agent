import json
import pytest
import yaml

from agent.hpo import HPOStudy, Objective, SearchSpace, SearchParameter, Trial, TrialBudget
from agent.hpo.feedback import HPOFeedbackAnalyzer, cross_budget_diagnostics
from agent.hpo.opportunities import analyze
from agent.hpo.resources import CampaignResourceBudget, ResourceBudgetExhausted
from agent.runners.checkpoint_selection import candidates, choose
from agent.runners.speechbrain_distributed import training_runtime_signature
from agent.tools.hpo_analysis_tools import _json


def study():
    return HPOStudy("s", "e", "tpe", SearchSpace([SearchParameter("lr", "float", low=1e-6, high=1., scale="log")]),
        [Objective("eer", "min")], [TrialBudget("low", epochs=1), TrialBudget("high", epochs=3)],
        controller_mode="llm", pruner_strategy="successive_halving", initial_trial_count=9,
        max_training_runs=13, history_context={"training_runtime": {"decision_opportunities": True}})


def trials(s, count):
    return [Trial(str(i), {"lr": 1e-3}, s.budgets[0], status="completed", metrics={"eer": i+1}) for i in range(count)]


def test_opportunity_respects_evidence_remaining_and_active_batch():
    s = study()
    assert not analyze(s, trials(s, 2)).eligible
    assert analyze(s, trials(s, 3)).reason == "bounded_probe_only"
    assert "search_space" in analyze(s, trials(s, 6)).allowed_actions
    assert not analyze(s, trials(s, 9)).eligible
    assert not analyze(s, trials(s, 6), remaining_runs=0).eligible
    rows = trials(s, 6) + [Trial("pending", {}, s.budgets[0])]
    assert analyze(s, rows).reason == "batch_in_progress"


def test_feedback_uses_initial_fidelity_and_log_distance():
    s = study(); rows = trials(s, 6)
    rows += [Trial("high", {"lr": 1e-6}, s.budgets[1], rung=1, status="completed", metrics={"eer": .5})]
    feedback = HPOFeedbackAnalyzer().analyze(s, rows)
    assert feedback["comparable_rung"] == 0
    assert feedback["boundary_hits"] == []  # log midpoint is not a lower-bound hit
    assert feedback["best_metric"] == 1


def test_curve_detects_late_reversal_and_missing_values():
    result = HPOFeedbackAnalyzer._curve_summary([
        {"train_loss": 2., "valid_loss": v} for v in [9., 2., 3., 4., 5., 6.]])
    assert result["valid_loss"]["trend"] == "down"
    assert result["valid_loss"]["recent_slope"] == 1
    assert result["final_loss_gap"] == 4
    assert HPOFeedbackAnalyzer._curve_summary([{"eer": float("nan")}]) == {"points": 1}


def test_paired_rank_reversal_requires_three_pairs():
    s = study(); parents = trials(s, 3)
    children = [Trial("c"+p.trial_id, p.parameters, s.budgets[1], parent_trial_id=p.trial_id,
        rung=1, status="completed", metrics={"eer": 4-p.metrics["eer"]}) for p in parents]
    assert cross_budget_diagnostics(parents + children[:2], "eer")[0]["spearman"] is None
    result = cross_budget_diagnostics(parents + children, "eer")[0]
    assert result["spearman"] == -1
    assert result["rank_reversals"] == 3


def test_candidates_are_top_two_plus_last_and_selection_is_complete(tmp_path):
    for epoch, score in enumerate([.2, .1, .7, .8], 1):
        path = tmp_path / f"CKPT+{epoch}"; path.mkdir()
        (path / "CKPT.yaml").write_text(yaml.safe_dump({"ErrorRate": score,
            "training_observation": {"checkpoint_epoch": epoch}}), encoding="utf-8")
    rows = candidates(tmp_path)
    assert [r["epoch"] for r in rows] == [1, 2, 4]
    with pytest.raises(ValueError):
        choose(rows, "eer", "min")
    for r in rows:
        r.update(status="success", metrics={"eer": .3})
    assert choose(rows, "eer", "min")["epoch"] == 1
    rows[-1]["metrics"]["eer"] = .1
    assert choose(rows, "eer", "min")["epoch"] == 4


def test_concurrent_reservations_and_idempotent_settlement(tmp_path):
    now = [0.]
    ledger = CampaignResourceBudget(tmp_path / "budget.json", limits={"max_gpu_hours": 100/3600}, clock=lambda: now[0])
    kwargs = dict(experiment_id="e", trial_id="t", attempt=1, stage="training", gpu_count=1)
    assert ledger.start_stage(**kwargs, stage_id="a", available_slots=2) == 50
    assert ledger.start_stage(**kwargs, stage_id="b") == 50
    with pytest.raises(ResourceBudgetExhausted):
        ledger.start_stage(**kwargs, stage_id="c")
    now[0] = 10
    assert ledger.snapshot()["gpu_seconds"] == 20
    ledger.finish_stage(stage_id="a")
    ledger.finish_stage(stage_id="a")
    assert ledger.snapshot()["gpu_seconds"] == 20
    ledger.finish_stage(stage_id="b")
    assert ledger.data["gpu_seconds"] == 20
    ledger.close()


def test_signature_separates_protocols_and_bounded_tool_output():
    assert training_runtime_signature({}) != training_runtime_signature({"checkpoint_selection": "validation_topk"})
    payload = json.loads(_json({"trials": [{"id": str(i), "curve": list(range(1000))} for i in range(50)]}, 2000))
    assert payload["truncated"]
    assert "preview" not in payload


def test_candidate_evaluation_failure_recovers_without_repeating_successes(tmp_path, minimal_config, dataset_dir):
    from agent.utils import ExperimentTracker
    from agent.hpo import HPOService
    from agent.agents.hpo_execution import TrialExecutor
    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment = tracker.create_hpo_experiment(config_path=str(minimal_config), data_folder=str(dataset_dir))
    service = HPOService(tracker)
    s = service.create_study(experiment, SearchSpace([SearchParameter("lr", "categorical", choices=[.1])]),
        [Objective("eer", "min")], [TrialBudget("full", epochs=1)], strategy="grid_search", max_training_runs=1)
    trial = service.suggest_trials(s, 1)[0]
    service.record_trial(s, trial.trial_id, status="running")
    rows = []
    for epoch in (1, 2, 3):
        path = tmp_path / str(epoch); path.mkdir()
        rows.append({"path": str(path), "epoch": epoch, "observation": {"checkpoint_epoch": epoch}})
    executor = TrialExecutor(experiment, str(dataset_dir), tmp_path / "experiments", {},
        "speaker_verification", "ecapa_tdnn", "speechbrain", "speechbrain")
    calls = []
    def invoke(trial, attempt, tool, payload, stage, gpus):
        epoch = int(Path(payload["model_path"]).name)
        calls.append((attempt, epoch))
        assert payload["defer_commit"]
        if attempt == 1 and epoch == 2:
            return {"status": "failed", "error": "temporary connection error"}
        return {"status": "success", "metrics": {"validation": {"eer": 1/epoch}}}
    from pathlib import Path
    executor._invoke_stage = invoke
    train = {"extensions": {"speechbrain": {"checkpoint_candidates": rows}}}
    failed = executor._evaluate_candidates(trial, 1, train, 10, 1, 1, 1, {})
    assert failed["status"] == "failed"
    assert service.load_trial(experiment, trial.trial_id).status == "running"
    recovered = service.load_trial(experiment, trial.trial_id)
    recovered.cost["evaluation_only"] = True
    result = executor._evaluate_candidates(recovered, 2, {}, 0, 1, 1, 1, {})
    assert result["status"] == "success"
    assert calls == [(1, 1), (1, 2), (2, 2), (2, 3)]
    assert result["metrics"]["eer"] == pytest.approx(1/3)
    assert result["cost"]["training_observation"]["selected_checkpoint"]["checkpoint_epoch"] == 3
