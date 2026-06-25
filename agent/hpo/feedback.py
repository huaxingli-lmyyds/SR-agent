"""Deterministic feedback analysis for Study strategy reviews."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from .contracts import HPOStudy, StrategyProposal, Trial


class HPOFeedbackAnalyzer:
    def analyze(self, study: HPOStudy, trials: List[Trial]) -> Dict[str, Any]:
        objective = study.objectives[0]
        completed = [
            trial for trial in trials
            if trial.status in {"completed", "promoted"}
            and isinstance(trial.metrics.get(objective.metric), (int, float))
        ]
        failures = Counter(
            str(trial.cost.get("failure_category") or "unknown")
            for trial in trials if trial.status == "failed"
        )
        boundary_hits: List[Dict[str, Any]] = []
        if completed:
            reverse = objective.mode == "max"
            best = sorted(completed, key=lambda item: item.metrics[objective.metric], reverse=reverse)[0]
            for parameter in study.search_space.parameters:
                value = best.parameters.get(parameter.name)
                if not isinstance(value, (int, float)) or parameter.low is None or parameter.high is None:
                    continue
                span = float(parameter.high) - float(parameter.low)
                if span <= 0:
                    continue
                ratio = (float(value) - float(parameter.low)) / span
                if ratio <= 0.05:
                    boundary_hits.append({"parameter": parameter.name, "edge": "low", "value": value})
                elif ratio >= 0.95:
                    boundary_hits.append({"parameter": parameter.name, "edge": "high", "value": value})
        failure_total = sum(failures.values())
        terminal_count = len([
            trial for trial in trials
            if trial.status in {"completed", "promoted", "stopped", "failed"}
        ])
        best_metric = None
        if completed:
            values = [item.metrics[objective.metric] for item in completed]
            best_metric = min(values) if objective.mode == "min" else max(values)
        return {
            "completed_trials": len(completed),
            "failed_trials": failure_total,
            "failure_clusters": dict(failures),
            "dominant_failure": failures.most_common(1)[0][0] if failures else None,
            "failure_rate": failure_total / terminal_count if terminal_count else 0.0,
            "boundary_hits": boundary_hits,
            "best_metric": best_metric,
        }

    def propose(
        self,
        study: HPOStudy,
        feedback: Dict[str, Any],
        available_strategies: Optional[List[str]] = None,
    ) -> StrategyProposal:
        """Create a conservative deterministic proposal when no LLM reviewer is enabled."""
        strategy = study.candidate_strategy or study.strategy
        reasons: List[str] = []
        search_space = None
        if feedback["failure_rate"] >= 0.5:
            strategy = "random_search"
            reasons.append("high_failure_rate_explore_safely")
        elif feedback["completed_trials"] >= 2 and strategy != "tpe":
            strategy = (
                "tpe"
                if feedback["completed_trials"] >= 5 and "tpe" in (available_strategies or [])
                else "adaptive_search"
            )
            reasons.append("completed_history_available")
        if feedback["boundary_hits"]:
            parameters = [parameter.to_dict() for parameter in study.search_space.parameters]
            for hit in feedback["boundary_hits"]:
                parameter = next(item for item in parameters if item["name"] == hit["parameter"])
                low, high = parameter.get("low"), parameter.get("high")
                if low is None or high is None:
                    continue
                span = high - low
                if hit["edge"] == "low":
                    parameter["low"] = max(low - span * 0.25, 1e-12) if parameter.get("scale") == "log" else low - span * 0.25
                else:
                    parameter["high"] = high + span * 0.25
            search_space = {"parameters": parameters, "constraints": study.search_space.constraints}
            reasons.append("best_trial_at_search_boundary")
        return StrategyProposal(
            action="refine_search_space" if search_space else "switch_strategy",
            requested_strategy=strategy,
            search_space=search_space,
            reason_codes=reasons or ["keep_current_candidate_generation"],
            evidence=feedback,
            confidence=1.0,
        )
