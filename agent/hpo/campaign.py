"""Cross-Study optimization campaign stopping policy."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .contracts import OptimizationCampaign


class CampaignPolicy:
    def record_study(
        self,
        campaign: OptimizationCampaign,
        *,
        experiment_id: str,
        study_id: str,
        best_value: float,
        training_runs: int,
    ) -> OptimizationCampaign:
        previous = campaign.best_value
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
        if campaign.target_value is None or campaign.best_value is None:
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

