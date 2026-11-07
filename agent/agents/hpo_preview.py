"""No-execution preview using the same confirmation reservation rules."""
from agent.hpo.contracts import Objective, OptimizationCampaign, TrialBudget
from agent.hpo.confirmation import reserve
from agent.hpo.execution_options import normalize_options


def preview_plan(context, budget):
    from agent.agents.hpo_agent import HPOAgent
    options = normalize_options(context.get("runtime_options") or {})
    strategy = context.get("pruner") or context.get("strategy", "auto")
    budgets = HPOAgent._build_budgets(context.get("budgets"), strategy)
    if strategy != "successive_halving":
        budgets = [budgets[-1]]
    limit = int(budget.get("max_training_runs", 10))
    studies = int(budget.get("max_studies", 1))
    campaign = OptimizationCampaign(Objective(context.get("primary_metric", "eer"), context.get("metric_mode", "min")),
        max_studies=studies, max_total_training_runs=budget.get("max_total_training_runs") or limit*studies,
        control_state={"per_study_limit": limit})
    reserve(campaign, options, budget, budgets, seed=context.get("hpo_seed", 0),
            special_probes=bool(context.get("hypothesis_probes") or context.get("cross_stage_probes")))
    confirmation = campaign.control_state.get("confirmation", {"status": "disabled", "reserved_runs": 0})
    search_limit = min(limit, campaign.max_total_training_runs-confirmation["reserved_runs"])
    initial = budget.get("initial_trial_count") or HPOAgent._default_initial_trial_count(strategy, budgets, search_limit)
    promotions = (budget.get("promotion_limits") or HPOAgent._default_promotion_limits(initial, budgets, budget.get("reduction_factor", 3))) if strategy == "successive_halving" else []
    return {"dry_run": True, "search_runs_per_study_cap": search_limit,
        "search_total_cap": campaign.max_total_training_runs-confirmation["reserved_runs"],
        "initial_candidates": initial, "promotion_limits": promotions,
        "budgets": [b.to_dict() for b in budgets], "confirmation": confirmation,
        "max_concurrent_trials": options.get("max_concurrent_trials", 1),
        "gpu_pool": options.get("gpu_pool"), "gpus_per_trial": options.get("gpus_per_trial", 1),
        "checkpoint_evaluations_per_trial_max": 3 if options.get("checkpoint_selection") == "validation_topk" else 1,
        "promotion_training": options.get("promotion_training", "restart"),
        "continuation_requires": ["identical_actual_data", "identical_parameters_and_seed", "compatible_runtime", "complete_per_rank_state"],
        "resource_limits": {k:budget.get(k) for k in ("max_gpu_hours", "max_wall_seconds")}}
