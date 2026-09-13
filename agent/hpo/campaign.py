"""Cross-Study optimization campaign stopping policy."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from agent.core.metrics import is_finite_metric, require_finite_metric
from .contracts import HPOStudy, OptimizationCampaign
from .history import budget_key


def study_confirmation_signature(study: HPOStudy) -> Dict[str, Any]:
    """Return the immutable protocol used to compare completed Studies."""
    context = study.history_context or {}
    objective = study.objectives[0]
    final_budget = budget_key(study.budgets[-1].to_dict())
    return {
        "objective": objective.to_dict(),
        "final_budget": {
            "epochs": final_budget[0],
            "data_fraction": final_budget[1],
            "max_duration_seconds": final_budget[2],
        },
        "metric_protocol": context.get("metric_protocol"),
        "metric_units": context.get("metric_units"),
        "dataset": context.get("dataset"),
        "dataset_id": context.get("dataset_id"),
        "dataset_version": context.get("dataset_version"),
        "task_type": context.get("task_type"),
        "model_family": context.get("model_family"),
        "implementation": context.get("implementation"),
        "runner": context.get("runner"),
        "config_sha256": context.get("config_sha256"),
        "verification_config_sha256": context.get("verification_config_sha256"),
        "validation_pairs_sha256": context.get("validation_pairs_sha256"),
        "training_exclusion_pairs_sha256": context.get(
            "training_exclusion_pairs_sha256"
        ),
    }


class CampaignPolicy:
    def record_study(
        self,
        campaign: OptimizationCampaign,
        *,
        experiment_id: str,
        study_id: str,
        best_value: float,
        training_runs: int,
        confirmation_signature: Optional[Dict[str, Any]] = None,
    ) -> OptimizationCampaign:
        require_finite_metric({campaign.objective.metric: best_value}, campaign.objective.metric)
        signature = dict(confirmation_signature or {})
        if campaign.confirmation_signature:
            if signature != campaign.confirmation_signature:
                raise ValueError(
                    "Study confirmation protocol differs from the frozen Campaign protocol"
                )
        elif signature:
            campaign.confirmation_signature = signature
        previous = campaign.best_value
        if not is_finite_metric(previous):
            previous = None
        improvement = None if previous is None else (
            previous - best_value if campaign.objective.mode == "min" else best_value - previous
        )
        is_better = previous is None or (best_value < previous if campaign.objective.mode == "min" else best_value > previous)
        if is_better:
            campaign.best_value = best_value
            campaign.best_experiment_id = experiment_id
        campaign.study_summaries.append({
            "experiment_id": experiment_id,
            "study_id": study_id,
            "best_value": best_value,
            "training_runs": training_runs,
            "confirmation_signature": signature,
            "improvement": improvement,
            "improved": bool(is_better and (improvement is None or improvement >= campaign.min_improvement)),
        })
        campaign.updated_at = datetime.now().isoformat()
        return campaign

    def should_continue(self, campaign: OptimizationCampaign) -> bool:
        if self._target_reached(campaign):
            return self._stop(campaign, "target_reached")
        if len(campaign.study_summaries) >= campaign.max_studies:
            return self._stop(campaign, "max_studies_reached")
        total_runs = sum(item["training_runs"] for item in campaign.study_summaries)
        if campaign.max_total_training_runs is not None and total_runs >= campaign.max_total_training_runs:
            return self._stop(campaign, "max_total_training_runs_reached")
        recent = campaign.study_summaries[-campaign.patience:]
        if len(recent) >= campaign.patience and not any(item["improved"] for item in recent):
            return self._stop(campaign, "patience_exhausted")
        return True

    @staticmethod
    def remaining_runs(campaign: OptimizationCampaign) -> Optional[int]:
        if campaign.max_total_training_runs is None:
            return None
        used = sum(item["training_runs"] for item in campaign.study_summaries)
        return max(campaign.max_total_training_runs - used, 0)

    @staticmethod
    def _target_reached(campaign: OptimizationCampaign) -> bool:
        if not is_finite_metric(campaign.target_value) or not is_finite_metric(campaign.best_value):
            return False
        return (
            campaign.best_value <= campaign.target_value
            if campaign.objective.mode == "min"
            else campaign.best_value >= campaign.target_value
        )

    @staticmethod
    def _stop(campaign: OptimizationCampaign, reason: str) -> bool:
        campaign.status = "completed"
        campaign.stop_reason = reason
        campaign.updated_at = datetime.now().isoformat()
        return False


__all__ = ["CampaignPolicy", "study_confirmation_signature"]
