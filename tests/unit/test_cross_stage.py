from copy import deepcopy

import pytest

from agent.hpo import (
    HPOScheduler,
    HPOService,
    Objective,
    SearchParameter,
    SearchSpace,
    TrialBudget,
    cross_stage,
)
from agent.utils import ExperimentTracker


@pytest.fixture
def setup_cross_stage(tmp_path, minimal_config, dataset_dir):
    tracker = ExperimentTracker(tmp_path / "hpo")
    eid = tracker.create_hpo_experiment(
        config_path=str(minimal_config), data_folder=str(dataset_dir)
    )
    service = HPOService(tracker)
    study = service.create_study(
        eid,
        SearchSpace(
            [
                SearchParameter("lr", "float", low=0.0005, high=0.003),
                SearchParameter("margin", "float", low=0.15, high=0.3),
            ]
        ),
        [Objective("eer", "min")],
        [
            TrialBudget("screen", epochs=1, data_fraction=1.0),
            TrialBudget("promote", epochs=3, data_fraction=1.0),
            TrialBudget("full", epochs=6, data_fraction=1.0),
        ],
        strategy="tpe",
        sampler_strategy="tpe",
        pruner_strategy="successive_halving",
        controller_mode="llm",
        max_trials=13,
        max_training_runs=13,
        initial_trial_count=9,
        promotion_limits=[3, 1],
        candidate_batch_size=3,
        cross_stage_probes_enabled=True,
    )
    return service, study


def keep():
    return dict(
        action="keep_search",
        hypothesis=None,
        anchor_trial_id=None,
        challenger_trial_id=None,
        evidence_trial_ids=[],
        minimum_late_advantage=None,
        region=None,
        reason="insufficient_evidence",
        confidence=0.0,
    )


def initial_trials(service, study):
    ts = service.suggest_trials(study, 3)
    for index, t in enumerate(ts):
        finish(service, study, t, 0.10 + 0.02 * index)
    return ts


def finish(service, study, trial, value):
    service.record_trial(study, trial.trial_id, status="running")
    service.record_trial(
        study, trial.trial_id, status="completed", metrics={"eer": value}
    )


def pair_proposal(ts):
    return {
        **keep(),
        "action": "paired_promotion",
        "hypothesis": "The initially worse configuration will win at the next budget.",
        "anchor_trial_id": ts[0].trial_id,
        "challenger_trial_id": ts[1].trial_id,
        "evidence_trial_ids": [ts[0].trial_id, ts[1].trial_id],
        "minimum_late_advantage": 0.001,
        "reason": "Compare both configurations before narrowing the search.",
        "confidence": 0.8,
    }


def region_proposal(study, trials):
    outcome = cross_stage.pair_outcome(study, trials)
    winner = next(
        t for t in trials if t.trial_id == outcome["target_winner_trial_id"]
    )
    p = study.search_space.parameters[0]
    quarter = (p.high - p.low) / 4
    bounds = (
        {"low": p.low + quarter, "high": p.high}
        if winner.parameters[p.name] >= p.low + quarter
        else {"low": p.low, "high": p.high - quarter}
    )
    return {
        **keep(),
        "action": "focus_region",
        "region": {p.name: bounds},
        "evidence_trial_ids": outcome["source_trial_ids"]
        + outcome["child_trial_ids"],
        "reason": "One broad local batch near the observed later-budget winner.",
        "confidence": 0.8,
    }


def test_scheduler_pairs_early_then_focuses_one_batch_and_preserves_allocation(
    setup_cross_stage,
):
    service, study = setup_cross_stage
    space = deepcopy(study.search_space.to_dict())
    calls, executions = [], []

    def review(current, feedback):
        phase = feedback["opportunity"]["phase"]
        calls.append(phase)
        trials = service.list_trials(current.experiment_id)
        if phase == "paired_promotion":
            return pair_proposal(sorted(trials, key=lambda t: t.metrics["eer"]))
        return region_proposal(current, trials)

    def execute(t, attempt):
        executions.append(t.rung)
        pair = study.scheduler_state.get("cross_stage_pair") or {}
        if t.rung == 0:
            value = 0.10 + 0.02 * sum(r == 0 for r in executions[:-1])
        elif t.trial_id in pair.get("child_trial_ids", []):
            value = 0.09 if t.trial_id == pair["child_trial_ids"][0] else 0.07
        else:
            value = 0.065 if t.rung == 2 else 0.11
        return {"status": "success", "metrics": {"eer": value}}

    result = HPOScheduler(service, execute, cross_stage_reviewer=review).run(
        study
    )
    assert result.study.status == "completed", result.errors
    assert calls == ["paired_promotion", "region_review"]
    assert executions == [0] * 3 + [1] * 2 + [0] * 6 + [1, 2]
    assert [sum(t.rung == r for t in result.trials) for r in range(3)] == [
        9,
        3,
        1,
    ]
    assert service.training_runs_used(study) == 13
    assert study.search_space.to_dict() == space
    focused = [
        t
        for t in result.trials
        if t.rung == 0 and t.provenance.get("region_id")
    ]
    assert len(focused) == 3
    bounds = study.scheduler_state["cross_stage_focus"]["bounds"]["lr"]
    assert all(
        bounds["low"] <= t.parameters["lr"] <= bounds["high"] for t in focused
    )
    ordered = [
        service.load_trial(study.experiment_id, tid) for tid in study.trial_ids
    ]
    assert all(
        not t.provenance.get("region_id")
        for t in [t for t in ordered if t.rung == 0][-3:]
    )
    outcome = cross_stage.pair_outcome(study, result.trials)
    assert outcome["status"] == "supported_observation"
    assert outcome["late_challenger_advantage"] == pytest.approx(0.02)
    assert not outcome["causal_claim"] and not outcome["full_budget_comparison"]


@pytest.mark.parametrize(
    "bad",
    [
        "invented",
        "same_id",
        "winner_not_anchor",
        "low_confidence",
        "missing_evidence",
        "nonpositive_threshold",
        "budget_change",
    ],
)
def test_bad_pair_is_rejected_without_consuming_training_slots(
    setup_cross_stage, bad
):
    service, study = setup_cross_stage
    ts = initial_trials(service, study)
    p = pair_proposal(ts)
    if bad == "invented":
        p["challenger_trial_id"] = "invented"
    elif bad == "same_id":
        p["challenger_trial_id"] = p["anchor_trial_id"]
    elif bad == "winner_not_anchor":
        p["anchor_trial_id"], p["challenger_trial_id"] = (
            p["challenger_trial_id"],
            p["anchor_trial_id"],
        )
    elif bad == "low_confidence":
        p["confidence"] = 0.2
    elif bad == "missing_evidence":
        p["evidence_trial_ids"] = []
    elif bad == "nonpositive_threshold":
        p["minimum_late_advantage"] = 0
    elif bad == "budget_change":
        p["max_training_runs"] = 999
    assert not cross_stage.review(service, study, p)["accepted"]
    assert len(service.list_trials(study.experiment_id)) == 3
    assert not study.scheduler_state.get("cross_stage_pair")
    assert not cross_stage.opportunity(
        study, service.list_trials(study.experiment_id)
    )["eligible"]


def test_partial_pair_commit_recovers_without_duplicate_children(
    setup_cross_stage, monkeypatch
):
    service, study = setup_cross_stage
    ts = initial_trials(service, study)
    original = service.save_trial
    children = []

    def interrupt(eid, t):
        if t.rung == 1:
            children.append(t.trial_id)
            if len(children) == 2:
                raise KeyboardInterrupt()
        return original(eid, t)

    monkeypatch.setattr(service, "save_trial", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cross_stage.review(service, study, pair_proposal(ts))
    monkeypatch.setattr(service, "save_trial", original)
    restored = service.load_study(study.experiment_id)
    cross_stage.ensure_pair_trials(service, restored)
    cross_stage.ensure_pair_trials(service, restored)
    assert len(service.list_trials(study.experiment_id)) == 5
    assert len(restored.scheduler_state["cross_stage_reviews"]) == 1
    assert (
        len(
            {
                t.parent_trial_id
                for t in service.list_trials(study.experiment_id)
                if t.rung == 1
            }
        )
        == 2
    )


def test_failed_arm_never_supports_region_focus(setup_cross_stage):
    service, study = setup_cross_stage
    ts = initial_trials(service, study)
    cross_stage.review(service, study, pair_proposal(ts))
    pair = study.scheduler_state["cross_stage_pair"]
    a, b = [
        service.load_trial(study.experiment_id, tid)
        for tid in pair["child_trial_ids"]
    ]
    finish(service, study, a, 0.05)
    service.record_trial(study, b.trial_id, status="running")
    service.record_trial(
        study, b.trial_id, status="failed", stop_reason="invalid result"
    )
    assert (
        cross_stage.pair_outcome(
            study, service.list_trials(study.experiment_id)
        )["status"]
        == "inconclusive"
    )
    assert not cross_stage.opportunity(
        study, service.list_trials(study.experiment_id)
    )["eligible"]


@pytest.mark.parametrize(
    "bad", ["narrow", "outside", "missing_pair", "wrong_winner"]
)
def test_region_requires_broad_bounds_and_complete_evidence(
    setup_cross_stage, bad
):
    service, study = setup_cross_stage
    ts = initial_trials(service, study)
    cross_stage.review(service, study, pair_proposal(ts))
    for i, tid in enumerate(
        study.scheduler_state["cross_stage_pair"]["child_trial_ids"]
    ):
        finish(
            service,
            study,
            service.load_trial(study.experiment_id, tid),
            0.09 - 0.02 * i,
        )
    trials = service.list_trials(study.experiment_id)
    p = region_proposal(study, trials)
    winner = next(
        t
        for t in trials
        if t.trial_id
        == cross_stage.pair_outcome(study, trials)["target_winner_trial_id"]
    )
    if bad == "narrow":
        p["region"]["lr"] = {
            "low": winner.parameters["lr"] - 1e-8,
            "high": winner.parameters["lr"] + 1e-8,
        }
    elif bad == "outside":
        p["region"]["lr"] = {"low": 0.0, "high": 1.0}
    elif bad == "missing_pair":
        p["evidence_trial_ids"] = p["evidence_trial_ids"][:2]
    elif bad == "wrong_winner":
        p["region"]["lr"] = {
            "low": 0.0005,
            "high": winner.parameters["lr"] - 1e-8,
        }
    assert not cross_stage.review(service, study, p)["accepted"]
    assert not study.scheduler_state.get("cross_stage_focus")


def test_bad_advice_falls_back_to_the_original_tpe_and_promotion_budget(
    setup_cross_stage,
):
    service, study = setup_cross_stage
    calls = []

    def review(current, feedback):
        calls.append(feedback["opportunity"]["phase"])
        return {"advice_error": "malformed response"}

    result = HPOScheduler(
        service,
        lambda t, a: {
            "status": "success",
            "metrics": {"eer": t.parameters["lr"]},
        },
        cross_stage_reviewer=review,
    ).run(study)
    assert result.study.status == "completed", result.errors
    assert calls == ["paired_promotion"]
    assert len(result.trials) == 13
    assert [sum(t.rung == r for t in result.trials) for r in range(3)] == [
        9,
        3,
        1,
    ]
    assert not any(t.provenance.get("region_id") for t in result.trials)


def test_pair_outcome_handles_maximization_and_invalid_lineage(
    setup_cross_stage,
):
    service, study = setup_cross_stage
    ts = initial_trials(service, study)
    study.objectives[0].mode = "max"
    p = pair_proposal([ts[2], ts[1]])
    assert cross_stage.review(service, study, p)["accepted"]
    for i, tid in enumerate(
        study.scheduler_state["cross_stage_pair"]["child_trial_ids"]
    ):
        finish(
            service,
            study,
            service.load_trial(study.experiment_id, tid),
            0.2 + 0.05 * i,
        )
    trials = service.list_trials(study.experiment_id)
    assert (
        cross_stage.pair_outcome(study, trials)["status"]
        == "supported_observation"
    )
    child = next(t for t in trials if t.rung == 1)
    child.parameters["lr"] = 0.999
    assert cross_stage.pair_outcome(study, trials)["status"] == "inconclusive"
    assert (
        child.trial_id
        in cross_stage.evidence(study, trials)["invalid_lineage_trial_ids"]
    )


def test_rejected_region_samples_fall_back_to_global_tpe(
    setup_cross_stage, monkeypatch
):
    from agent.hpo.strategies import STRATEGIES

    service, study = setup_cross_stage
    ts = initial_trials(service, study)
    cross_stage.review(service, study, pair_proposal(ts))
    for i, tid in enumerate(
        study.scheduler_state["cross_stage_pair"]["child_trial_ids"]
    ):
        finish(
            service,
            study,
            service.load_trial(study.experiment_id, tid),
            0.09 - 0.02 * i,
        )
    cross_stage.review(
        service,
        study,
        region_proposal(study, service.list_trials(study.experiment_id)),
    )
    strategy = STRATEGIES.get("tpe")
    original = strategy.suggest
    calls = []

    def suggest(space, count, **kwargs):
        calls.append(space.to_dict())
        if len(calls) == 1:
            return [{"lr": -1.0, "margin": 0.2}]
        return original(space, count, **kwargs)

    monkeypatch.setattr(strategy, "suggest", suggest)
    new = service.suggest_trials(study, 3)
    assert len(new) == 3 and len(calls) == 2
    assert calls[1] == study.search_space.to_dict()
    assert not any(t.provenance.get("region_id") for t in new)
    assert study.scheduler_state["cross_stage_focus"]["consumed"]


def test_interrupted_consultation_is_not_repeated_on_resume(setup_cross_stage):
    service, study = setup_cross_stage
    calls = []

    def interrupt(current, feedback):
        calls.append(1)
        raise KeyboardInterrupt()

    def executor(t, a):
        return {"status": "success", "metrics": {"eer": t.parameters["lr"]}}

    with pytest.raises(KeyboardInterrupt):
        HPOScheduler(service, executor, cross_stage_reviewer=interrupt).run(
            study
        )
    restored = service.load_study(study.experiment_id)
    assert (
        restored.scheduler_state["cross_stage_pending_review"]["phase"]
        == "paired_promotion"
    )
    result = HPOScheduler(
        service, executor, cross_stage_reviewer=interrupt
    ).run(restored, resume=True)
    assert result.study.status == "completed", result.errors
    assert len(calls) == 1 and len(result.trials) == 13
    assert not restored.scheduler_state.get("cross_stage_pair")
    persisted = service.load_study(study.experiment_id)
    assert not persisted.scheduler_state.get("cross_stage_pending_review")
    assert len(persisted.scheduler_state["cross_stage_reviews"]) == 1
    assert not persisted.scheduler_state["cross_stage_reviews"][0]["accepted"]
