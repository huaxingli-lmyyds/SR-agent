from copy import deepcopy

import pytest

from agent.hpo import (
    CampaignPolicy,
    HPOFeedbackAnalyzer,
    OptimizationCampaign,
    campaign_control,
)
from agent.hpo.reporting import campaign_from_dict
from tests.unit import test_cross_stage as cross_tests

setup_cross_stage = cross_tests.setup_cross_stage
initial_trials = cross_tests.initial_trials


def campaign_for(study, total=39):
    campaign = OptimizationCampaign(
        study.objectives[0],
        max_studies=3,
        patience=3,
        max_total_training_runs=total,
        control_state={
            "original_search_space": deepcopy(study.search_space.to_dict()),
            "per_study_limit": 13,
            "initial_trial_count": 9,
            "promotion_limits": [3, 1],
        },
    )
    CampaignPolicy().record_study(
        campaign,
        experiment_id=study.experiment_id,
        study_id=study.study_id,
        best_value=0.1,
        training_runs=13,
    )
    return campaign


def test_menu_preserves_cross_stage_slots_and_confirmation(setup_cross_stage):
    service, study = setup_cross_stage
    campaign = campaign_for(study)
    plans = campaign_control.options(service, study, campaign)
    assert plans["continue"]["max_training_runs"] == 13
    assert plans["compact"]["max_training_runs"] == 11
    assert plans["compact"]["initial_trial_count"] == 7
    assert plans["compact"]["promotion_limits"] == [3, 1]
    assert all(p["sampler"] == "tpe" for p in plans.values())
    assert (
        campaign_control.options(service, study, campaign_for(study, total=23))
        == {}
    )
    restored = campaign_from_dict(campaign.to_dict(), study.objectives[0])
    assert restored.control_state == campaign.control_state


@pytest.mark.parametrize(
    "bad", ["unknown_plan", "early_stop", "invented_study", "budget_override"]
)
def test_campaign_response_is_bounded_by_menu_and_observed_evidence(
    setup_cross_stage, bad
):
    service, study = setup_cross_stage
    campaign = campaign_for(study)
    evidence = campaign_control.context(
        service,
        study,
        campaign,
        campaign_control.options(service, study, campaign),
        {},
    )
    value = {
        "plan_id": "compact",
        "evidence_study_ids": [study.study_id],
        "reason": "retain confirmation at lower cost",
    }
    campaign_control.validate_response(value, evidence)
    if bad == "unknown_plan":
        value["plan_id"] = "invented"
    elif bad == "early_stop":
        value["plan_id"] = "stop"
    elif bad == "invented_study":
        value["evidence_study_ids"] = ["invented"]
    else:
        value["max_training_runs"] = 999
    with pytest.raises(ValueError):
        campaign_control.validate_response(value, evidence)


def test_global_option_restores_original_domain(setup_cross_stage):
    service, study = setup_cross_stage
    campaign = campaign_for(study)
    study.search_space.parameters[0].low += 0.0001
    plans = campaign_control.options(service, study, campaign)
    assert (
        plans["global_explore"]["search_space"]
        == campaign.control_state["original_search_space"]
    )
    assert (
        plans["continue"]["search_space"]
        != plans["global_explore"]["search_space"]
    )


def test_complete_bracket_cost_screens_unaffordable_plan(
    setup_cross_stage, monkeypatch
):
    service, study = setup_cross_stage
    campaign = campaign_for(study)
    trials = initial_trials(service, study)
    for rung, trial in enumerate(trials):
        trial.rung = rung
        trial.status = "completed"
        trial.budget = study.budgets[rung]
        trial.cost.update(
            attempt_training_seconds=10,
            attempt_evaluation_seconds=2,
            attempt_training_gpu_seconds=3600,
            attempt_evaluation_gpu_seconds=0,
            ledger_cost={"source": "campaign_ledger", "training_seconds": 10, "evaluation_seconds": 2,
                         "training_gpu_seconds": 3600, "evaluation_gpu_seconds": 0,
                         "cost_may_be_incomplete": False, "stage_count": 2},
        )
    monkeypatch.setattr(service, "list_trials", lambda _: trials)
    plans = campaign_control.options(service, study, campaign)
    evidence = campaign_control.context(
        service, study, campaign, plans, {"gpu_hours_remaining": 12}
    )
    assert "continue" not in plans and "compact" in plans
    assert (
        evidence["budget_screened_plans"]["continue"]["estimated_cost"][
            "gpu_hours"
        ]
        == 13
    )
    assert campaign_control.rule_choice(evidence) == "compact"


def test_actual_cost_feedback_uses_current_executor_fields(setup_cross_stage):
    service, study = setup_cross_stage
    trials = initial_trials(service, study)
    for trial, seconds in zip(trials, [10, float("nan"), True], strict=True):
        trial.cost.update(
            attempt_training_seconds=seconds, attempt_evaluation_seconds=2
        )
    cost = HPOFeedbackAnalyzer().analyze(study, trials)["cost_summary"]
    assert cost["stages"]["training"]["total_duration_seconds"] == 10
    assert cost["stages"]["training"]["observations"] == 1
    assert cost["stages"]["evaluation"]["total_duration_seconds"] == 6
    assert cost["cost_may_be_incomplete"]  # Missing GPU accounting is not free compute.
