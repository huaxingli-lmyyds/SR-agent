"""Versioned execution options shared by CLI, persistence and resume."""
KEYS = ("managed_execution", "max_concurrent_trials", "gpu_pool", "gpus_per_trial",
        "promotion_training", "confirmation_enabled", "confirmation_top_k", "confirmation_seeds")
DEFAULTS = dict(managed_execution=False, max_concurrent_trials=1, gpu_pool=None,
                gpus_per_trial=1, promotion_training="restart", confirmation_enabled=False,
                confirmation_top_k=2, confirmation_seeds=3)


def managed_enabled(options):
    return bool(options.get("managed_execution") or options.get("gpu_pool")
                or options.get("max_concurrent_trials", 1) > 1
                or options.get("promotion_training") == "resume_if_compatible"
                or options.get("confirmation_enabled"))


def normalize_options(options):
    from agent.runners.stage_pool import validate_options
    return validate_options(options)


def resume_options(requested, persisted):
    requested = normalize_options(requested)
    for key in KEYS:
        if key in requested and requested[key] != persisted.get(key, DEFAULTS[key]):
            raise ValueError(f"resume {key} conflicts with persisted execution protocol")
    return {**requested, **{k:v for k,v in persisted.items() if k in KEYS}}
