"""Execution-only support for the five-group benchmark (no import in dry-run)."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

from agent.agents.hpo_agent import HPOAgent
from agent.hpo import StrategyProposal


ALLOWED_SAMPLERS = {"random_search", "tpe", "adaptive_search"}
LOCKED_FIELDS = (
    "budgets",
    "max_training_runs",
    "initial_trial_count",
    "promotion_limits",
    "reduction_factor",
    "requested_pruner",
)


def within_envelope(proposed: dict, envelope: dict) -> bool:
    """Allow narrowing/re-expansion within the original domain, never new domains."""
    try:
        original = {p["name"]: p for p in envelope["parameters"]}
        params = proposed["parameters"]
        if len(params) != len(original) or {p["name"] for p in params} != set(
            original
        ):
            return False
        if proposed.get("constraints", []) != envelope.get("constraints", []):
            return False
        for p in params:
            base = original[p["name"]]
            if any(
                p.get(k, default) != base.get(k, default)
                for k, default in (
                    ("parameter_type", None),
                    ("scale", "linear"),
                    ("condition", {}),
                )
            ):
                return False
            if base["parameter_type"] == "categorical":
                if not p.get("choices") or any(
                    not any(
                        type(v) is type(b) and v == b for b in base["choices"]
                    )
                    for v in p["choices"]
                ):
                    return False
            else:
                low, high = p["low"], p["high"]
                if any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    for v in (low, high)
                ):
                    return False
                if not base["low"] <= low <= high <= base["high"]:
                    return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def constrain_proposal(
    proposal: StrategyProposal | None, envelope: dict
) -> StrategyProposal:
    """An explicit keep response avoids the service's implicit rule controller."""
    if proposal is None:
        return StrategyProposal(
            action="keep_strategy", reason_codes=["benchmark_no_proposal"]
        )
    value = proposal.to_dict()
    blocked = []
    for name in LOCKED_FIELDS:
        if value.get(name) is not None:
            blocked.append(name)
            value[name] = None
    for name in ("requested_sampler", "requested_strategy"):
        if value.get(name) is not None and value[name] not in ALLOWED_SAMPLERS:
            blocked.append(name)
            value[name] = None
    if value.get("candidate_proposals"):
        blocked.append("candidate_proposals")
    value["candidate_proposals"] = []
    if value.get("search_space") is not None and not within_envelope(
        value["search_space"], envelope
    ):
        blocked.append("search_space")
        value["search_space"] = None
    value["evidence"] = {
        **value.get("evidence", {}),
        "benchmark_blocked_fields": blocked,
    }
    return StrategyProposal.from_dict(value, preserve_audit_fields=True)


class BenchmarkHPOAgent(HPOAgent):
    """Use the real HPO workflow, with benchmark-only execution restrictions."""

    def __init__(
        self,
        *,
        variant: str,
        envelope: dict,
        event_sink,
        **kwargs,
    ):
        super().__init__(enable_llm_advisor=variant == "system", **kwargs)
        self.variant = variant
        self.envelope = envelope
        self.event_sink = event_sink

    def _planning_proposal(self, *args, **kwargs):
        raw = super()._planning_proposal(*args, **kwargs)
        restricted = constrain_proposal(raw, self.envelope)
        self.event_sink(
            "proposals",
            {
                "phase": "planning",
                "raw": raw.to_dict(),
                "restricted": restricted.to_dict(),
            },
        )
        return restricted

    def _runtime_strategy_reviewer(self, request, campaign):
        if self.variant != "system":
            return lambda study, feedback: StrategyProposal(
                action="keep_strategy",
                reason_codes=["benchmark_fixed_baseline"],
            )
        reviewer = super()._runtime_strategy_reviewer(request, campaign)

        def review(study, feedback):
            raw = reviewer(study, feedback)
            restricted = constrain_proposal(raw, self.envelope)
            self.event_sink(
                "proposals",
                {
                    "phase": "runtime",
                    "study_id": study.study_id,
                    "raw": raw.to_dict() if raw else None,
                    "restricted": restricted.to_dict(),
                },
            )
            return restricted

        return review

    def _invoke_strategy_model(self, prompt: str, **objective: Any) -> Any:
        started = perf_counter()
        event = {"prompt": prompt, "allowed_samplers": sorted(ALLOWED_SAMPLERS)}
        try:
            response = super()._invoke_strategy_model(
                prompt
                + "\nBenchmark restrictions: only random_search, tpe, adaptive_search; "
                "no agent candidates; budgets, allocation and pruner are fixed. "
                "Search-space edits must stay within the original bounds.",
                **objective,
            )
            event.update(
                {
                    "status": "success",
                    "response": self._extract_message_content(response),
                    "final_response_usage": getattr(
                        response, "usage_metadata", None
                    ),
                }
            )
            return response
        except Exception as exc:
            event.update(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )
            raise
        finally:
            event["duration_seconds"] = perf_counter() - started
            self.event_sink("advisor", event)

    def _trial_executor(
        self,
        experiment_id,
        data_folder,
        runtime_options=None,
        execution_context=None,
    ):
        execute = super()._trial_executor(
            experiment_id, data_folder, runtime_options, execution_context
        )

        def measured(trial, attempt):
            # Persist ownership before execution so --resume can reopen the same Study.
            event = {
                "experiment_id": experiment_id,
                "trial_id": trial.trial_id,
                "parent_trial_id": trial.parent_trial_id,
                "rung": trial.rung,
                "attempt": attempt,
                "parameters": trial.parameters,
                "budget": trial.budget.to_dict(),
                "search_phase": trial.search_phase,
            }
            self.event_sink("attempts", {**event, "event": "started"})
            started = perf_counter()
            try:
                result = execute(trial, attempt)
                event.update(
                    {
                        "status": result.get("status"),
                        "error": result.get("error"),
                    }
                )
                event["attempt_cost"] = result.get("cost") or {}
                return result
            except BaseException as exc:
                event.update(
                    {
                        "status": "interrupted"
                        if isinstance(exc, KeyboardInterrupt)
                        else "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise
            finally:
                event["duration_seconds"] = perf_counter() - started
                self.event_sink("attempts", {**event, "event": "finished"})

        return measured
