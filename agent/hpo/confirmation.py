"""Budgeted, restart-safe confirmation that never feeds the search sampler."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import statistics

from agent.core.metrics import is_finite_metric
from .contracts import SearchParameter, SearchSpace
from .history import budget_key
from .records import _write_json_atomic


def reserve(campaign, options, budget, budgets, *, seed=0, special_probes=False):
    if not options.get("confirmation_enabled"):
        return
    count, repeats = options.get("confirmation_top_k", 2), options.get("confirmation_seeds", 3)
    requested = count * repeats
    total = campaign.max_total_training_runs
    initial = budget.get("initial_trial_count")
    explicit = initial is not None or budget.get("promotion_limits") is not None
    minimum = len(budgets) * max(1, int(budget.get("min_completed_per_rung", 1)))
    if explicit or special_probes:
        minimum = campaign.control_state["per_study_limit"] * campaign.max_studies
    eligible = total is not None and total >= minimum + requested
    rng = random.Random(seed)
    seeds = []
    while len(seeds) < repeats:
        value = rng.randrange(1, 2**31)
        if value != seed and value not in seeds:
            seeds.append(value)
    campaign.control_state["confirmation"] = {
        "status": "reserved" if eligible else "skipped", "reserved_runs": requested if eligible else 0,
        "requested_runs": requested, "top_k": count, "seeds": seeds,
        "reason": None if eligible else "total_budget_cannot_preserve_search_and_confirmation",
        "studies": [], "search_best": None, "confirmed_best": None,
    }
    campaign.control_state["phase"] = "search"


def select_candidates(trials, objective, count):
    groups = defaultdict(list)
    for experiment, trial in trials:
        if trial.status in {"completed", "promoted"} and is_finite_metric(trial.metrics.get(objective.metric)):
            groups[budget_key(trial.budget.to_dict())].append((experiment, trial))
    sign = 1 if objective.mode == "min" else -1
    fallback = []
    for key in sorted(groups, key=lambda k: (k[0] or 0, k[1]), reverse=True):
        unique = {}
        for exp, trial in sorted(groups[key], key=lambda item: (sign*item[1].metrics[objective.metric], item[1].trial_id)):
            signature = json.dumps({k:v for k,v in trial.parameters.items() if k != "seed"}, sort_keys=True)
            unique.setdefault(signature, (exp, trial))
        items = list(unique.values())
        if len(items) >= count:
            return items[:count]
        if len(items) > len(fallback):
            fallback = items
    return fallback[:count]


def summarize(groups, seeds, objective):
    if not groups or any(len(g["observations"]) != len(seeds) or {o["seed"] for o in g["observations"]} != set(seeds) or any(
        o["status"] not in {"completed", "promoted"} or not is_finite_metric(o.get("metric"))
        for o in g["observations"]) for g in groups):
        return {"status": "incomplete", "confirmed_best": None, "groups": groups}
    rows = []
    for group in groups:
        values = [o["metric"] for o in group["observations"]]
        rows.append({**group, "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values), "sample_count": len(values)})
    rows.sort(key=lambda g: ((1 if objective.mode == "min" else -1)*g["mean"], g["experiment_id"]))
    winner = rows[0]
    model = min(winner["observations"], key=lambda o: (1 if objective.mode == "min" else -1)*o["metric"])
    differences = []
    if len(rows) > 1:
        a = {o["seed"]: o["metric"] for o in rows[0]["observations"]}
        b = {o["seed"]: o["metric"] for o in rows[1]["observations"]}
        differences = [{"seed": seed, "winner_minus_runner_up": a[seed]-b[seed]} for seed in seeds]
    return {"status": "completed", "groups": rows, "paired_differences": differences,
        "evidence": "limited_seed_replication; no statistical_significance_claim",
        "confirmed_best": {"experiment_id": winner["experiment_id"], "parameters": winner["parameters"],
            "mean": winner["mean"], "trial_id": model["trial_id"], "selection_basis": "configuration_mean;artifact_best_observed_seed"}}


def run_confirmation(agent, tracker, service, campaign, search_best, search_experiment, options, data_folder):
    from .scheduler import HPOScheduler
    from .policies import RetryPolicy
    state = campaign.control_state.get("confirmation")
    if not state or state["status"] == "skipped":
        return search_best, search_experiment
    manifest_path = Path(agent.experiments_dir) / "campaign_resources" / f"{campaign.campaign_id}.confirmation.json"
    if manifest_path.exists():
        state = json.loads(manifest_path.read_text(encoding="utf-8"))
        campaign.control_state["confirmation"] = state
    def persist():
        _write_json_atomic(manifest_path, state)
        for summary in campaign.study_summaries:
            tracker.update_hpo_experiment(summary["experiment_id"], extensions={"optimization": {"campaign": campaign.to_dict()}})
    state["search_best"] = state.get("search_best") or {"experiment_id": search_experiment,
        "trial_id": search_best.trial_id, "parameters": search_best.parameters, "metrics": search_best.metrics}
    if state["status"] == "completed":
        winner = state["confirmed_best"]
        return service.load_trial(winner["experiment_id"], winner["trial_id"]), winner["experiment_id"]
    campaign.control_state["phase"] = "confirmation"
    if not state.get("candidates"):
        all_trials = [(s["experiment_id"], t) for s in campaign.study_summaries for t in service.list_trials(s["experiment_id"])]
        used = {t.parameters.get("seed") for _,t in all_trials}
        while any(seed in used for seed in state["seeds"]):
            state["seeds"] = [seed+1 for seed in state["seeds"]]
        selected = select_candidates(all_trials, campaign.objective, state["top_k"])
        state["candidates"] = [{"source_experiment": exp, "source_trial_id": t.trial_id, "parameters": {k:v for k,v in t.parameters.items() if k != "seed"},
            "experiment_id": f"confirmation_{campaign.campaign_id}_{i}"} for i,(exp,t) in enumerate(selected)]
        state["reserved_runs"] = len(selected)*len(state["seeds"])
    state["status"] = "running"; persist()
    groups = []
    try:
        for candidate in state["candidates"]:
            experiment = candidate["experiment_id"]
            source = tracker.get_experiment(candidate["source_experiment"])
            source_study = service.load_study(candidate["source_experiment"])
            runtime = {**options, "promotion_training": "restart", "managed_execution": True}
            source_trial = service.load_trial(candidate["source_experiment"], candidate["source_trial_id"])
            observed = source_trial.cost.get("managed_training_result", {}).get("extensions", {}).get("speechbrain", {}).get("runtime", {})
            if observed.get("training_manifests"):
                runtime["frozen_data"] = observed["training_manifests"]
            execution = {**source["execution"], "training_runtime_options": runtime}
            if tracker.get_experiment(experiment) is None:
                tracker.create_hpo_experiment(config_path=str(tracker.get_config_snapshot(candidate["source_experiment"])),
                    data_folder=data_folder, task=source["task"], model=source["model"], execution=execution,
                    requested_id=experiment, description="Independent seed confirmation; excluded from search history")
            try:
                study = service.load_study(experiment)
                resume = True
            except FileNotFoundError:
                parameters = [SearchParameter(k, "categorical", choices=[v]) for k,v in candidate["parameters"].items()]
                parameters.append(SearchParameter("seed", "categorical", choices=state["seeds"]))
                study = service.create_study(experiment, SearchSpace(parameters), [campaign.objective], [source_study.budgets[-1]],
                    strategy="grid_search", pruner_strategy="none", controller_mode="fixed",
                    max_training_runs=len(state["seeds"]), initial_trial_count=len(state["seeds"]),
                    candidate_batch_size=len(state["seeds"]))
                resume = False
            if experiment not in state["studies"]:
                state["studies"].append(experiment); persist()
            scheduled = HPOScheduler(service, agent._trial_executor(experiment, data_folder, runtime),
                retry_policy=RetryPolicy(max_retries=0), resource_budget=agent._resource_budget).run(study, resume=resume)
            guard = scheduled.study.scheduler_state.get("failure_guard") or {}
            if guard.get("status") == "open":
                from .policies import SystemFailureBlocked
                raise SystemFailureBlocked("system failure during confirmation", guard)
            observations = [{"seed": t.parameters["seed"], "trial_id": t.trial_id, "status": t.status,
                             "metric": t.metrics.get(campaign.objective.metric)} for t in scheduled.trials]
            groups.append({"experiment_id": experiment, "parameters": candidate["parameters"], "observations": observations})
            state["training_runs"] = sum(service.training_runs_used(service.load_study(e)) for e in state["studies"])
            persist()
        state.update(summarize(groups, state["seeds"], campaign.objective))
        campaign.control_state["phase"] = "completed"
        state["reserved_runs"] = 0
        persist()
    except BaseException:
        state["status"] = "interrupted"; persist()
        raise
    if state.get("confirmed_best"):
        best = state["confirmed_best"]
        return service.load_trial(best["experiment_id"], best["trial_id"]), best["experiment_id"]
    return search_best, search_experiment
