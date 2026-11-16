"""Persisted HPO record codecs, including legacy Study migration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .contracts import (
    HPOStudy,
    Objective,
    SearchParameter,
    SearchSpace,
    Trial,
    TrialBudget,
)


def search_space_from_dict(data: dict[str, Any]) -> SearchSpace:
    return SearchSpace(
        parameters=[
            SearchParameter(**item) for item in data.get("parameters") or []
        ],
        constraints=data.get("constraints") or [],
    )


def study_from_dict(data: dict[str, Any]) -> HPOStudy:
    study = HPOStudy(
        study_id=data["study_id"],
        experiment_id=data["experiment_id"],
        strategy=data["strategy"],
        search_space=search_space_from_dict(data["search_space"]),
        objectives=[Objective(**item) for item in data.get("objectives") or []],
        budgets=[TrialBudget(**item) for item in data.get("budgets") or []],
        sampler_strategy=data.get("sampler_strategy"),
        pruner_strategy=data.get("pruner_strategy"),
        candidate_strategy=data.get("candidate_strategy"),
        controller_mode=data.get("controller_mode", "auto"),
        reduction_factor=data.get("reduction_factor", 3),
        max_trials=data.get("max_trials"),
        initial_trial_count=data.get("initial_trial_count"),
        promotion_limits=data.get("promotion_limits") or [],
        max_training_runs=data.get("max_training_runs"),
        min_completed_per_rung=data.get("min_completed_per_rung", 1),
        candidate_batch_size=data.get("candidate_batch_size"),
        sampler_config=data.get("sampler_config") or {},
        search_phases=data.get("search_phases") or [],
        hypotheses=data.get("hypotheses") or [],
        pending_candidate_proposals=data.get("pending_candidate_proposals")
        or [],
        candidate_proposal_reviews=data.get("candidate_proposal_reviews") or [],
        candidate_generation_reviews=data.get("candidate_generation_reviews")
        or [],
        scheduler_state=data.get("scheduler_state") or {},
        constraints=data.get("constraints") or [],
        strategy_reviews=data.get("strategy_reviews") or [],
        next_study_proposal=data.get("next_study_proposal"),
        warm_start_trials=data.get("warm_start_trials") or [],
        history_context=data.get("history_context") or {},
        trial_ids=data.get("trial_ids") or [],
        best_trial_id=data.get("best_trial_id"),
        status=data.get("status", "created"),
        stop_reason=data.get("stop_reason"),
        random_seed=data.get("random_seed", 0),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )
    if not study.search_phases:
        sampler = (
            study.candidate_strategy
            or study.sampler_strategy
            or (
                "random_search"
                if study.strategy == "successive_halving"
                else study.strategy
            )
        )
        study.search_phases = [
            {
                "phase_index": 0,
                "sampler": sampler,
                "sampler_config": dict(study.sampler_config),
                "start_trial_index": 0,
                "end_trial_index": None,
                "trigger": "legacy_study_migration",
                "proposal_id": None,
                "decision_id": None,
                "reason_codes": [],
                "fallback_sampler": None,
                "fallback_sampler_config": {},
                "created_at": study.created_at,
            }
        ]
    return study


def trial_from_dict(data: dict[str, Any]) -> Trial:
    return Trial(
        trial_id=data["trial_id"],
        parameters=data.get("parameters") or {},
        budget=TrialBudget(**data["budget"]),
        status=data.get("status", "suggested"),
        parent_trial_id=data.get("parent_trial_id"),
        rung=data.get("rung", 0),
        metrics=data.get("metrics") or {},
        intermediate_metrics=data.get("intermediate_metrics") or [],
        cost=data.get("cost") or {},
        artifacts=data.get("artifacts") or [],
        stop_reason=data.get("stop_reason"),
        candidate_source=str(data.get("candidate_source") or "optimizer"),
        search_phase=int(data.get("search_phase") or 0),
        proposal_id=(
            str(data["proposal_id"])
            if data.get("proposal_id") is not None
            else None
        ),
        hypothesis_id=(
            str(data["hypothesis_id"])
            if data.get("hypothesis_id") is not None
            else None
        ),
        provenance=data.get("provenance") or {},
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
