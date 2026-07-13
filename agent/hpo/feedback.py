"""Deterministic feedback analysis for Study strategy reviews."""

from __future__ import annotations

from collections import Counter, defaultdict
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
        best_parameters = None
        ranked_trials: List[Dict[str, Any]] = []
        if completed:
            reverse = objective.mode == "max"
            ranked = sorted(completed, key=lambda item: item.metrics[objective.metric], reverse=reverse)
            best_metric = ranked[0].metrics[objective.metric]
            best_parameters = dict(ranked[0].parameters)
            ranked_trials = [self._trial_summary(trial, objective.metric) for trial in ranked[:5]]
        return {
            "completed_trials": len(completed),
            "failed_trials": failure_total,
            "terminal_trials": terminal_count,
            "failure_clusters": dict(failures),
            "dominant_failure": failures.most_common(1)[0][0] if failures else None,
            "failure_rate": failure_total / terminal_count if terminal_count else 0.0,
            "boundary_hits": boundary_hits,
            "best_metric": best_metric,
            "best_parameters": best_parameters,
            "ranked_trials": ranked_trials,
            "rung_summaries": self._rung_summaries(trials, objective.metric, objective.mode),
            "parameter_observations": self._parameter_observations(completed, objective.metric, objective.mode),
            "failed_trial_examples": self._failed_trial_examples(trials),
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
                    parameter["low"] = max(low / 3.0, 1e-12) if parameter.get("scale") == "log" else low - span * 0.1
                else:
                    parameter["high"] = high * 3.0 if parameter.get("scale") == "log" else high + span * 0.1
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

    @staticmethod
    def _trial_summary(trial: Trial, primary_metric: str) -> Dict[str, Any]:
        return {
            "trial_id": trial.trial_id,
            "rung": trial.rung,
            "stage": trial.budget.stage,
            "status": trial.status,
            "parameters": dict(trial.parameters),
            "primary_metric": trial.metrics.get(primary_metric),
            "training": {
                key: trial.metrics[key]
                for key in ("valid_error_rate", "final_valid_loss", "final_train_loss", "final_epoch")
                if key in trial.metrics
            },
            "evaluation": {
                key: trial.metrics[key]
                for key in ("eer", "min_dcf")
                if key in trial.metrics
            },
            "curve": HPOFeedbackAnalyzer._curve_summary(trial.intermediate_metrics),
        }

    @staticmethod
    def _rung_summaries(trials: List[Trial], metric: str, mode: str) -> List[Dict[str, Any]]:
        by_rung: Dict[int, List[Trial]] = defaultdict(list)
        for trial in trials:
            by_rung[int(trial.rung)].append(trial)
        summaries: List[Dict[str, Any]] = []
        for rung in sorted(by_rung):
            items = by_rung[rung]
            metric_values = [float(trial.metrics[metric]) for trial in items if isinstance(trial.metrics.get(metric), (int, float))]
            best_value = None
            if metric_values:
                best_value = min(metric_values) if mode == "min" else max(metric_values)
            summaries.append({
                "rung": rung,
                "stage": items[0].budget.stage if items else None,
                "total": len(items),
                "completed": len([trial for trial in items if trial.status in {"completed", "promoted"}]),
                "failed": len([trial for trial in items if trial.status == "failed"]),
                "active": len([trial for trial in items if trial.status in {"suggested", "running"}]),
                "best_metric": best_value,
            })
        return summaries

    @staticmethod
    def _parameter_observations(trials: List[Trial], metric: str, mode: str) -> Dict[str, Any]:
        observations: Dict[str, Any] = {}
        if not trials:
            return observations
        reverse = mode == "max"
        ranked = sorted(trials, key=lambda trial: trial.metrics[metric], reverse=reverse)
        top = ranked[: max(1, min(3, len(ranked)))]
        for name in sorted({key for trial in trials for key in trial.parameters}):
            values = [trial.parameters.get(name) for trial in trials if name in trial.parameters]
            top_values = [trial.parameters.get(name) for trial in top if name in trial.parameters]
            numeric = [float(value) for value in values if isinstance(value, (int, float))]
            observations[name] = {
                "top_values": top_values,
                "unique_values": sorted(set(values)) if all(isinstance(value, (int, float, str)) for value in values) else list({str(value) for value in values}),
            }
            if numeric:
                observations[name].update({
                    "min": min(numeric),
                    "max": max(numeric),
                    "top_min": min(float(value) for value in top_values if isinstance(value, (int, float))),
                    "top_max": max(float(value) for value in top_values if isinstance(value, (int, float))),
                })
        return observations

    @staticmethod
    def _curve_summary(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not history:
            return {}
        keys = ("train_loss", "valid_loss", "valid_error_rate", "eer")
        summary: Dict[str, Any] = {"points": len(history)}
        for key in keys:
            values = [float(item[key]) for item in history if isinstance(item.get(key), (int, float))]
            if len(values) >= 2:
                summary[key] = {
                    "first": values[0],
                    "last": values[-1],
                    "delta": values[-1] - values[0],
                    "trend": "down" if values[-1] < values[0] else "up" if values[-1] > values[0] else "flat",
                }
        return summary

    @staticmethod
    def _failed_trial_examples(trials: List[Trial]) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        for trial in trials:
            if trial.status != "failed":
                continue
            examples.append({
                "trial_id": trial.trial_id,
                "rung": trial.rung,
                "stage": trial.budget.stage,
                "parameters": dict(trial.parameters),
                "failure_category": trial.cost.get("failure_category") or "unknown",
                "stop_reason": str(trial.stop_reason or "")[:300],
            })
            if len(examples) >= 3:
                break
        return examples
