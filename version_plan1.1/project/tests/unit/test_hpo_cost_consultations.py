import json
from types import SimpleNamespace

import pytest

from agent.hpo import HPOService, HPOScheduler, Objective, SearchSpace, SearchParameter, StrategyProposal, Trial, TrialBudget
from agent.hpo.consultations import request_response
from agent.hpo.costs import attach_ledger_costs, observed_cost
from agent.hpo.effects import summarize_decision_effect
from agent.hpo.feedback import HPOFeedbackAnalyzer
from agent.hpo.resources import CampaignResourceBudget
from agent.utils.experiment_tracker import ExperimentTracker


def test_ledger_includes_retries_checkpoints_and_filters_studies(tmp_path):
    ledger = CampaignResourceBudget(tmp_path / "ledger.json", limits={})
    try:
        for stage_id, exp, stage, seconds, gpus in [
            ("failed_train", "e", "training", 4, 2),
            ("retry", "e", "training", 6, 2),
            ("ckpt1", "e", "evaluation", 2, 1),
            ("ckpt2_retry", "e", "evaluation", 3, 1),
            ("confirmation", "other", "training", 99, 4),
        ]:
            ledger.start_stage(experiment_id=exp, trial_id="t", attempt=2, stage=stage,
                               gpu_count=gpus, stage_id=stage_id)
            ledger.finish_stage(stage_id=stage_id, seconds=seconds)
            ledger.finish_stage(stage_id=stage_id, seconds=seconds)
        tracker = SimpleNamespace(get_experiment=lambda _: {"extensions": {"optimization": {
            "study": {"resource_budget": ledger.snapshot()}}}})
        trial = Trial("t", {}, TrialBudget("full", epochs=1), status="completed", metrics={"eer": 1},
                      provenance={"decision_id": "d"}, cost={"training": {"duration_seconds": 999}})
        attach_ledger_costs(tracker, "e", [trial])
        costs = observed_cost([trial])
        assert costs["training_seconds"] == 10
        assert costs["evaluation_seconds"] == 5
        assert costs["gpu_hours"] == 25 / 3600
        assert costs["stage_count"] == 4
        assert not costs["cost_may_be_incomplete"]
        assert HPOFeedbackAnalyzer._cost_summary([trial])["stages"]["training"]["total_duration_seconds"] == 10
        assert summarize_decision_effect([trial], "d", Objective("eer", "min"))["realized_outcome"]["training_seconds"] == 10
    finally:
        ledger.close()


def test_missing_ledger_does_not_silently_use_stale_projection(tmp_path):
    tracker = SimpleNamespace(get_experiment=lambda _: {"extensions": {"optimization": {
        "resource_budget": {"ledger_path": str(tmp_path / "missing")}}}})
    trial = Trial("t", {}, TrialBudget("full"), cost={"ledger_cost": {"training_seconds": 999}})
    attach_ledger_costs(tracker, "e", [trial])
    assert observed_cost([trial])["cost_may_be_incomplete"]
    assert observed_cost([trial])["sources"] == ["ledger_unavailable"]


def test_consultation_retry_and_response_replay():
    record, snapshots, calls = {}, [], []
    def persist():
        snapshots.append(json.loads(json.dumps(record)))
    def request():
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("transport interrupted")
        return {"action": "keep_strategy"}
    response = request_response(record, persist, request)
    assert record["attempts"] == 2
    assert snapshots[0]["status"] == "requested"
    assert record["status"] == "response_saved"
    assert request_response(record, persist, lambda: pytest.fail("must replay")) == response


def test_interruption_before_response_can_resume_and_retry_is_bounded():
    record = {}
    def crash():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        request_response(record, lambda: None, crash)
    assert record["status"] == "requested"
    assert request_response(record, lambda: None, lambda: {"ok": True}) == {"ok": True}
    assert record["attempts"] == 2
    failed = {}
    def error():
        raise TimeoutError()
    assert request_response(failed, lambda: None, error) is None
    assert failed["attempts"] == 3
    assert request_response(failed, lambda: None, lambda: pytest.fail("exhausted")) is None


def test_saved_runtime_response_is_applied_once_after_resume(tmp_path, minimal_config, dataset_dir):
    tracker = ExperimentTracker(tmp_path / "experiments")
    exp = tracker.create_hpo_experiment(config_path=str(minimal_config), data_folder=str(dataset_dir))
    service = HPOService(tracker)
    study = service.create_study(exp, SearchSpace([SearchParameter("lr", "float", low=.001, high=.1)]),
        [Objective("eer", "min")], [TrialBudget("full", epochs=1)], max_training_runs=12, controller_mode="llm")
    study.history_context["training_runtime"] = {"decision_opportunities": True}
    for trial in service.suggest_trials(study, 6):
        service.record_trial(study, trial.trial_id, status="running")
        service.record_trial(study, trial.trial_id, status="completed", metrics={"eer": .1})
    calls = []
    def reviewer(*args):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("temporary")
        return StrategyProposal(action="keep_strategy")
    scheduler = HPOScheduler(service, lambda *args: {}, strategy_reviewer=reviewer)
    scheduler._study = study
    proposal = scheduler._make_strategy_proposal("interval_review")
    assert len(calls) == 2
    scheduler._study = service.load_study(exp)
    replay = scheduler._make_strategy_proposal("interval_review")
    assert replay.proposal_id == proposal.proposal_id
    assert len(calls) == 2
    first = service.review_strategy(scheduler.study, replay)
    restored = service.load_study(exp)
    before = len(restored.strategy_reviews)
    assert service.review_strategy(restored, replay)["decision"]["decision_id"] == first["decision"]["decision_id"]
    assert len(restored.strategy_reviews) == before
    assert next(iter(restored.scheduler_state["decision_opportunities"].values()))["status"] in {"applied", "rejected"}
