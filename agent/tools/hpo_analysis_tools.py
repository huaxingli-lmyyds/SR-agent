"""Read-only, study-aware tools exposed only to the HPO advisor."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import tool

from agent.core.metrics import is_finite_metric
from agent.hpo.feedback import HPOFeedbackAnalyzer
from agent.hpo.campaign import study_confirmation_signature
from agent.hpo.service import HPOService
from agent.hpo.protocol import METRIC_PROTOCOL_ID
from agent.utils import ExperimentTracker


def _json(value: Any, max_chars: int = 16000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return json.dumps(
        {"truncated": True, "preview": text[: max_chars - 100]},
        ensure_ascii=False,
    )


def build_hpo_analysis_tools(
    tracker: ExperimentTracker,
    parameter_validator: Optional[Callable[[Dict[str, Any]], None]] = None,
    objective_metric: str = "eer",
    objective_mode: str = "min",
    confirmation_signature: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Bind read-only tools to the current experiment store and model validator."""
    service = HPOService(tracker, parameter_validator=parameter_validator)

    @tool("inspect_hpo_study")
    def inspect_hpo_study(experiment_id: str) -> str:
        """Inspect one HPO Study, its trial/rung state, feedback, failures, and cost."""
        try:
            study = service.load_study(experiment_id)
            trials = service.list_trials(experiment_id)
            feedback = HPOFeedbackAnalyzer().analyze(study, trials)
            return _json({
                "study": {
                    "study_id": study.study_id,
                    "experiment_id": study.experiment_id,
                    "status": study.status,
                    "controller_mode": study.controller_mode,
                    "sampler": study.candidate_strategy or study.sampler_strategy,
                    "sampler_config": study.sampler_config,
                    "search_phases": study.search_phases,
                    "pruner": study.pruner_strategy,
                    "search_space": study.search_space.to_dict(),
                    "budgets": [item.to_dict() for item in study.budgets],
                    "max_training_runs": study.max_training_runs,
                    "candidate_batch_size": study.candidate_batch_size,
                    "pending_agent_candidate_count": len(study.pending_candidate_proposals),
                    "next_study_proposal": study.next_study_proposal,
                    "objective": study.objectives[0].to_dict(),
                    "metric_protocol": study.history_context.get("metric_protocol"),
                    "recent_strategy_reviews": [
                        {
                            "trigger": review.get("trigger"),
                            "scope": review.get("scope"),
                            "controller_mode": review.get("controller_mode"),
                            "decision": review.get("decision"),
                            "realized_outcome": review.get("realized_outcome") or {},
                            "effect_estimate": review.get("effect_estimate") or {},
                        }
                        for review in (study.strategy_reviews or [])[-5:]
                    ],
                },
                "feedback": feedback,
            })
        except Exception as exc:
            return _json({"error": f"{type(exc).__name__}: {exc}"})

    @tool("compare_hpo_trials")
    def compare_hpo_trials(
        experiment_id: str,
        trial_ids: Optional[List[str]] = None,
        metric: Optional[str] = None,
    ) -> str:
        """Compare selected Trials using parameters, fidelity, metrics, cost, and provenance."""
        try:
            study = service.load_study(experiment_id)
            authoritative = study.objectives[0].metric
            if metric is not None and metric != authoritative:
                return _json({
                    "error": "requested metric conflicts with Study objective",
                    "requested_metric": metric,
                    "authoritative_metric": authoritative,
                })
            metric = authoritative
            selected = set((trial_ids or [])[:30])
            trials = [
                item for item in service.list_trials(experiment_id)
                if not selected or item.trial_id in selected
            ]
            rows = [{
                "trial_id": item.trial_id,
                "status": item.status,
                "rung": item.rung,
                "budget": item.budget.to_dict(),
                "parameters": item.parameters,
                "metric": item.metrics.get(metric),
                "metrics": item.metrics,
                "cost": item.cost,
                "candidate_source": item.candidate_source,
                "search_phase": item.search_phase,
                "proposal_id": item.proposal_id,
                "hypothesis_id": item.hypothesis_id,
                "parent_trial_id": item.parent_trial_id,
                "provenance": item.provenance,
            } for item in trials[:30]]
            return _json({"metric": metric, "trials": rows})
        except Exception as exc:
            return _json({"error": f"{type(exc).__name__}: {exc}"})

    @tool("validate_hpo_candidate")
    def validate_hpo_candidate(experiment_id: str, parameters: Dict[str, Any]) -> str:
        """Validate one candidate without creating a Trial or changing Study state."""
        try:
            study = service.load_study(experiment_id)
            normalized = service.validate_candidate_parameters(study, parameters)
            existing = {
                json.dumps(item.parameters, sort_keys=True, default=str)
                for item in service.list_trials(experiment_id)
            }
            existing.update(
                json.dumps(item.get("parameters") or {}, sort_keys=True, default=str)
                for item in study.pending_candidate_proposals
                if isinstance(item, dict)
            )
            existing.update(
                json.dumps(item.get("parameters") or {}, sort_keys=True, default=str)
                for item in study.warm_start_trials
                if isinstance(item, dict)
            )
            duplicate = json.dumps(normalized, sort_keys=True, default=str) in existing
            return _json({"valid": not duplicate, "duplicate": duplicate, "parameters": normalized})
        except Exception as exc:
            return _json({"valid": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool("list_comparable_hpo_experiments")
    def list_comparable_hpo_experiments(
        model_family: Optional[str] = None,
        dataset: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """List only experiments matching the bound frozen confirmation protocol."""
        required_signature = dict(confirmation_signature or {})
        if not required_signature:
            return _json({
                "error": "a frozen confirmation_signature is required for history comparison"
            })
        rows = []
        for record in tracker.list_experiments(
            limit=max(int(limit) * 20, 20),
            experiment_type="hpo",
        ):
            model = record.get("model") or {}
            task = record.get("task") or {}
            if record.get("status") != "success":
                continue
            if model_family and model.get("family") != model_family:
                continue
            if dataset and task.get("dataset") != dataset:
                continue
            if task.get("metric_protocol") != METRIC_PROTOCOL_ID:
                continue
            if (
                task.get("primary_metric") != objective_metric
                or task.get("metric_mode") != objective_mode
            ):
                continue
            optimization = (record.get("extensions") or {}).get("optimization") or {}
            candidate_signature = {}
            experiment_id = str(record.get("experiment_id") or "")
            try:
                candidate_signature = study_confirmation_signature(
                    service.load_study(experiment_id)
                )
            except Exception:
                candidate_signature = dict(
                    (optimization.get("campaign") or {}).get("confirmation_signature")
                    or {}
                )
            if candidate_signature != required_signature:
                continue
            best = (record.get("metrics") or {}).get("best") or {}
            primary_value = best.get("primary_value", best.get(objective_metric))
            if not is_finite_metric(primary_value):
                continue
            rows.append({
                "experiment_id": experiment_id,
                "status": record.get("status"),
                "task": task,
                "model": model,
                "best": best,
                "study": optimization.get("study"),
                "campaign": optimization.get("campaign"),
                "confirmation_signature": candidate_signature,
            })
            if len(rows) >= max(min(int(limit), 10), 1):
                break
        return _json(rows)

    return [
        inspect_hpo_study,
        compare_hpo_trials,
        validate_hpo_candidate,
        list_comparable_hpo_experiments,
    ]


__all__ = ["build_hpo_analysis_tools"]
