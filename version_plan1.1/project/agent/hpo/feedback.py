"""Deterministic feedback analysis for Study strategy reviews."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log, sqrt
from .opportunities import enabled as opportunities_enabled
from typing import Any, Dict, List, Optional

from agent.core.metrics import is_finite_metric
from . import cross_stage, hypothesis_probes
from .protocol import METRIC_PROTOCOL_ID, METRIC_UNITS
from .contracts import HPOStudy, StrategyProposal, Trial


class HPOFeedbackAnalyzer:
    def analyze(self, study: HPOStudy, trials: List[Trial]) -> Dict[str, Any]:
        objective = study.objectives[0]
        completed = [
            trial for trial in trials
            if trial.status in {"completed", "promoted"}
            and is_finite_metric(trial.metrics.get(objective.metric))
        ]
        comparable = list(completed)
        comparable_rung = None
        active_pruner = study.pruner_strategy or (
            "successive_halving" if study.strategy == "successive_halving" else "none"
        )
        if comparable and active_pruner == "successive_halving":
            comparable_rung = 0 if opportunities_enabled(study) else max(trial.rung for trial in comparable)
            comparable = [trial for trial in comparable if trial.rung == comparable_rung]
        failures = Counter(
            str(trial.cost.get("failure_category") or "unknown")
            for trial in trials if trial.status == "failed"
        )
        boundary_hits: List[Dict[str, Any]] = []
        if comparable:
            reverse = objective.mode == "max"
            best = sorted(comparable, key=lambda item: item.metrics[objective.metric], reverse=reverse)[0]
            for parameter in study.search_space.parameters:
                value = best.parameters.get(parameter.name)
                if not isinstance(value, (int, float)) or parameter.low is None or parameter.high is None:
                    continue
                low, high, observed = float(parameter.low), float(parameter.high), float(value)
                if parameter.scale == "log":
                    if min(low, high, observed) <= 0:
                        continue
                    low, high, observed = log(low), log(high), log(observed)
                span = high - low
                if span <= 0:
                    continue
                ratio = (observed - low) / span
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
        if comparable:
            reverse = objective.mode == "max"
            ranked = sorted(comparable, key=lambda item: item.metrics[objective.metric], reverse=reverse)
            best_metric = ranked[0].metrics[objective.metric]
            best_parameters = dict(ranked[0].parameters)
            ranked_trials = [self._trial_summary(trial, objective.metric) for trial in ranked[:5]]
        return {
            "hypothesis_probe": hypothesis_probes.opportunity(study, trials),
            "cross_stage": {"opportunity": cross_stage.opportunity(study, trials),
                            "evidence": cross_stage.evidence(study, trials)} if cross_stage.enabled(study) else {"opportunity": {"enabled": False}},
            "objective": objective.to_dict(),
            "metric_protocol": study.history_context.get(
                "metric_protocol", METRIC_PROTOCOL_ID
            ),
            "metric_units": study.history_context.get(
                "metric_units", dict(METRIC_UNITS)
            ),
            "completed_trials": len(completed),
            "failed_trials": failure_total,
            "terminal_trials": terminal_count,
            "failure_clusters": dict(failures),
            "dominant_failure": failures.most_common(1)[0][0] if failures else None,
            "failure_rate": failure_total / terminal_count if terminal_count else 0.0,
            "boundary_hits": boundary_hits,
            "best_metric": best_metric,
            "best_parameters": best_parameters,
            "comparable_rung": comparable_rung,
            "ranked_trials": ranked_trials,
            "rung_summaries": self._rung_summaries(trials, objective.metric, objective.mode),
            "search_phase_summaries": self._search_phase_summaries(
                study, trials, objective.metric, objective.mode
            ),
            "hypothesis_summaries": self._hypothesis_summaries(
                trials, objective.metric, objective.mode
            ),
            "parameter_observations": self._parameter_observations(comparable, objective.metric, objective.mode),
            "failed_trial_examples": self._failed_trial_examples(trials),
            "cost_summary": self._cost_summary(trials),
            "cross_budget_diagnostics": cross_budget_diagnostics(trials, objective.metric),
        }

    def propose(
        self,
        study: HPOStudy,
        feedback: Dict[str, Any],
        available_strategies: Optional[List[str]] = None,
    ) -> StrategyProposal:
        """Create a conservative deterministic proposal when no LLM reviewer is enabled."""
        strategy = study.candidate_strategy or study.sampler_strategy or study.strategy
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
            requested_sampler=strategy,
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
            "candidate_source": trial.candidate_source,
            "search_phase": trial.search_phase,
            "proposal_id": trial.proposal_id,
            "hypothesis_id": trial.hypothesis_id,
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
            "training_observation": trial.cost.get("training_observation", {}),
            "checkpoint_evaluations": trial.cost.get("checkpoint_evaluations", []),
        }

    @staticmethod
    def _rung_summaries(trials: List[Trial], metric: str, mode: str) -> List[Dict[str, Any]]:
        by_rung: Dict[int, List[Trial]] = defaultdict(list)
        for trial in trials:
            by_rung[int(trial.rung)].append(trial)
        summaries: List[Dict[str, Any]] = []
        for rung in sorted(by_rung):
            items = by_rung[rung]
            metric_values = [
                float(trial.metrics[metric]) for trial in items
                if trial.status in {"completed", "promoted"}
                and is_finite_metric(trial.metrics.get(metric))
            ]
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
    def _search_phase_summaries(
        study: HPOStudy,
        trials: List[Trial],
        metric: str,
        mode: str,
    ) -> List[Dict[str, Any]]:
        grouped: Dict[tuple[int, int], List[Trial]] = defaultdict(list)
        for trial in trials:
            grouped[(int(trial.search_phase), int(trial.rung))].append(trial)
        phases = {
            int(item.get("phase_index", index)): item
            for index, item in enumerate(study.search_phases)
            if isinstance(item, dict)
        }
        return [
            HPOFeedbackAnalyzer._group_summary(
                items,
                metric,
                mode,
                {
                    "search_phase": phase,
                    "sampler": phases.get(phase, {}).get("sampler"),
                    "sampler_config": phases.get(phase, {}).get("sampler_config", {}),
                    "rung": rung,
                },
            )
            for (phase, rung), items in sorted(grouped.items())
        ]

    @staticmethod
    def _hypothesis_summaries(
        trials: List[Trial],
        metric: str,
        mode: str,
    ) -> List[Dict[str, Any]]:
        grouped: Dict[tuple[str, int], List[Trial]] = defaultdict(list)
        for trial in trials:
            if trial.hypothesis_id:
                grouped[(str(trial.hypothesis_id), int(trial.rung))].append(trial)
        return [
            HPOFeedbackAnalyzer._group_summary(
                items,
                metric,
                mode,
                {"hypothesis_id": hypothesis_id, "rung": rung},
            )
            for (hypothesis_id, rung), items in sorted(grouped.items())
        ]

    @staticmethod
    def _group_summary(
        trials: List[Trial],
        metric: str,
        mode: str,
        identity: Dict[str, Any],
    ) -> Dict[str, Any]:
        values = [
            float(trial.metrics[metric]) for trial in trials
            if trial.status in {"completed", "promoted"}
            and is_finite_metric(trial.metrics.get(metric))
        ]
        return {
            **identity,
            "total": len(trials),
            "completed": len(values),
            "failed": len([trial for trial in trials if trial.status == "failed"]),
            "best_metric": (
                min(values) if mode == "min" else max(values)
            ) if values else None,
            "average_metric": round(sum(values) / len(values), 8) if values else None,
        }

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
            numeric = [
                float(value) for value in values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            numeric_top = [
                float(value) for value in top_values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            unique_values = []
            for value in values:
                if not any(value == existing for existing in unique_values):
                    unique_values.append(value)
            observations[name] = {
                "top_values": top_values,
                "unique_values": sorted(unique_values, key=lambda value: (type(value).__name__, repr(value))),
            }
            if numeric:
                observations[name].update({
                    "min": min(numeric),
                    "max": max(numeric),
                })
                if numeric_top:
                    observations[name].update({
                        "top_min": min(numeric_top),
                        "top_max": max(numeric_top),
                    })
        return observations

    @staticmethod
    def _curve_summary(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not history:
            return {}
        keys = ("train_loss", "valid_loss", "valid_error_rate", "eer")
        summary: Dict[str, Any] = {"points": len(history)}
        for key in keys:
            values = [float(item[key]) for item in history if is_finite_metric(item.get(key))]
            if len(values) >= 2:
                summary[key] = {
                    "first": values[0],
                    "last": values[-1],
                    "delta": values[-1] - values[0],
                    "trend": "down" if values[-1] < values[0] else "up" if values[-1] > values[0] else "flat",
                }
            if values:
                window = values[-5:]
                n = len(window)
                mean = sum(window) / n
                center = (n - 1) / 2
                denominator = sum((i - center) ** 2 for i in range(n))
                summary.setdefault(key, {}).update({
                    "recent_points": n,
                    "recent_slope": sum((i-center)*(v-mean) for i,v in enumerate(window)) / denominator if denominator else None,
                    "recent_std": sqrt(sum((v-mean)**2 for v in window) / n),
                    "best_value": min(values),
                    "best_point_index": values.index(min(values)),
                })
        paired = [item for item in history if is_finite_metric(item.get("train_loss")) and is_finite_metric(item.get("valid_loss"))]
        if paired:
            summary["final_loss_gap"] = paired[-1]["valid_loss"] - paired[-1]["train_loss"]
        return summary

    @staticmethod
    def _cost_summary(trials: List[Trial]) -> Dict[str, Any]:
        from .costs import observed_cost
        result = observed_cost(trials)
        result["observed_trial_count"] = len(trials)
        result["stages"] = {}
        for stage in ("training", "evaluation"):
            values = [observed_cost([t])[f"{stage}_seconds"] for t in trials
                      if ((t.cost.get("ledger_cost") or {}).get("stage_count", 0) > 0
                          or ("ledger_cost" not in t.cost and (
                              is_finite_metric((t.cost.get(stage) or {}).get("duration_seconds"))
                              or is_finite_metric(t.cost.get(f"attempt_{stage}_seconds")))))]
            result["stages"][stage] = {"observations": len(values),
                "total_duration_seconds": sum(values),
                "average_duration_seconds": sum(values) / len(values) if values else None,
                "max_duration_seconds": max(values) if values else None}
        return result
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


def cross_budget_diagnostics(trials, metric):
    """Paired observations only; selection-biased correlations are not causal."""
    by_id = {t.trial_id: t for t in trials}
    groups = defaultdict(list)
    for child in trials:
        parent = by_id.get(child.parent_trial_id)
        if (parent is None or child.rung != parent.rung + 1 or child.parameters != parent.parameters
            or child.status not in {"completed", "promoted"} or parent.status not in {"completed", "promoted"}
            or not all(is_finite_metric(t.metrics.get(metric)) for t in (parent, child))):
            continue
        groups[(parent.rung, child.rung)].append({"parent_trial_id": parent.trial_id,
            "child_trial_id": child.trial_id, "early": parent.metrics[metric], "late": child.metrics[metric]})
    def ranks(values):
        ordered = sorted(values)
        return [sum(i+1 for i,x in enumerate(ordered) if x == v) / ordered.count(v) for v in values]
    result = []
    for (early, late), pairs in sorted(groups.items()):
        x, y = ranks([p["early"] for p in pairs]), ranks([p["late"] for p in pairs])
        n = len(pairs); mx, my = sum(x)/n, sum(y)/n
        denom = sqrt(sum((v-mx)**2 for v in x) * sum((v-my)**2 for v in y))
        correlation = sum((a-mx)*(b-my) for a,b in zip(x,y))/denom if n >= 3 and denom else None
        reversals = sum((x[i]-x[j])*(y[i]-y[j]) < 0 for i in range(n) for j in range(i))
        result.append({"from_rung": early, "to_rung": late, "pair_count": n, "pairs": pairs,
            "spearman": correlation, "rank_reversals": reversals,
            "selection_bias": "promoted_candidates_only", "causal_claim": False})
    return result
