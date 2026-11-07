"""Shared, deterministic eligibility for bounded runtime consultations."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from agent.core.metrics import is_finite_metric
from . import hypothesis_probes
from .history import budget_key


def enabled(study):
    return bool(study.history_context.get("training_runtime", {}).get("decision_opportunities"))


@dataclass(frozen=True)
class DecisionOpportunity:
    opportunity_id: str
    eligible: bool
    allowed_actions: tuple
    evidence_count: int
    remaining_candidates: int
    reason: str

    def to_dict(self):
        return asdict(self)


def analyze(study, trials, *, minimum=6, remaining_runs=None):
    initial = [t for t in trials if t.rung == 0]
    limit = (study.initial_trial_count if study.pruner_strategy == "successive_halving"
             else study.max_training_runs or study.max_trials) or 0
    remaining = max(0, limit - len(initial) - len(study.pending_candidate_proposals))
    if remaining_runs is not None:
        remaining = min(remaining, max(0, remaining_runs))
    observed = [t for t in initial if t.status in {"completed", "promoted"}
                and budget_key(t.budget.to_dict()) == budget_key(study.budgets[0].to_dict())
                and is_finite_metric(t.metrics.get(study.objectives[0].metric))]
    terminal = [t for t in trials if t.status in {"completed", "promoted", "failed", "stopped"}]
    failures = sum(t.status == "failed" for t in terminal)
    reason, actions = "insufficient_evidence", ()
    probe = hypothesis_probes.opportunity(study, trials)
    if study.controller_mode != "llm":
        reason = "controller_not_llm"
    elif study.scheduler_state.get("cross_stage_policy"):
        reason = "dedicated_cross_stage_review"
    elif any(t.status in {"suggested", "running"} for t in trials):
        reason = "batch_in_progress"
    elif not remaining:
        reason = "no_remaining_candidates"
    elif probe.get("enabled"):
        reason = probe["reason"]
        if probe["eligible"]:
            actions = ("candidate_proposals", "hypotheses", "requested_sampler")
    elif len(observed) >= minimum or (failures >= 2 and failures / max(len(terminal), 1) >= .5):
        reason = "ready"
        actions = ("requested_sampler", "requested_strategy", "sampler_config", "search_space", "candidate_proposals", "hypotheses")
    elif len(observed) >= 3:
        reason = "bounded_probe_only"
        actions = ("candidate_proposals", "hypotheses", "requested_sampler")
    identity = json.dumps([study.study_id, sorted(t.trial_id for t in terminal), remaining, actions])
    return DecisionOpportunity(sha256(identity.encode()).hexdigest()[:24], bool(actions),
                               actions, len(observed), remaining, reason)
