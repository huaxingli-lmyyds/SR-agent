import json
from types import SimpleNamespace
import pytest
from agent.hpo.confirmation import reserve, summarize
from agent.hpo.contracts import Objective, OptimizationCampaign, TrialBudget
from agent.hpo.execution_options import normalize_options, resume_options
from agent.agents.hpo_preview import preview_plan


def test_confirmation_preserves_explicit_search_and_total_cap():
    campaign = OptimizationCampaign(Objective("eer", "min"), max_total_training_runs=13,
                                    control_state={"per_study_limit": 13})
    reserve(campaign, {"confirmation_enabled": True}, {"initial_trial_count": 9}, [TrialBudget("full", epochs=1)])
    assert campaign.control_state["confirmation"]["status"] == "skipped"
    preview = preview_plan({"strategy": "successive_halving", "runtime_options": {"confirmation_enabled": True}},
                           {"max_training_runs": 13, "max_studies": 1})
    assert preview["search_total_cap"] + preview["confirmation"]["reserved_runs"] == 13
    assert sum(preview["promotion_limits"]) + preview["initial_candidates"] <= preview["search_runs_per_study_cap"]


def test_confirmation_incomplete_never_ranks():
    groups = [{"experiment_id": "a", "parameters": {}, "observations": [
        {"status": "completed", "metric": .2, "seed": 1, "trial_id": "t"}]}]
    assert summarize(groups, [1, 2, 3], Objective("eer", "min"))["confirmed_best"] is None
    complete = summarize(groups, [1], Objective("eer", "min"))
    assert complete["confirmed_best"]["mean"] == .2
    assert complete["groups"][0]["std"] is None


@pytest.mark.parametrize("options", [
    {"max_concurrent_trials": 2}, {"gpu_pool": [0, 0]}, {"gpu_pool": [0], "gpus_per_trial": 2},
    {"gpu_pool": [0], "ddp_devices": [0]}, {"max_concurrent_trials": True}, {"confirmation_seeds": 0}])
def test_invalid_execution_options_fail_before_training(options):
    with pytest.raises(ValueError):
        normalize_options(options)


def test_resume_inherits_but_rejects_changed_protocol():
    saved = {"max_concurrent_trials": 2, "gpu_pool": [0, 1], "gpus_per_trial": 1}
    assert resume_options({}, saved)["gpu_pool"] == [0, 1]
    with pytest.raises(ValueError):
        resume_options({"gpu_pool": [2, 3]}, saved)


def test_continuation_rejects_changed_semantics_before_execution(tmp_path):
    from agent.runners.continuation_state import bind_continuation
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"signature": {"world_size": 1, "train_manifest": "a"}}))
    params = {}
    bind_continuation(params, {"enabled": True, "manifest": str(manifest)},
                      {"world_size": 2, "train_manifest": "b"})
    assert not params["hpo_continuation"]["enabled"]
    assert params["hpo_continuation"]["reason"] == "prepared_training_semantics_changed"


def test_comparison_arms_share_budget_and_isolate_protocol():
    from scripts.experiments.run_hpo_optimization_comparison import plans
    request = {"context": {"strategy": "successive_halving"},
               "budget": {"max_training_runs": 13, "max_gpu_hours": 2}}
    arms = plans(request)
    assert set(arms) == {"legacy", "p0", "full"}
    assert all(arm["request"]["budget"] == request["budget"] for arm in arms.values())
    assert arms["legacy"]["preview"]["checkpoint_evaluations_per_trial_max"] == 1
    assert arms["p0"]["preview"]["checkpoint_evaluations_per_trial_max"] == 3
    assert arms["full"]["preview"]["confirmation"]["reserved_runs"] == 6
    assert "runtime_options" not in request["context"]
