"""Model-agnostic hyperparameter optimization contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any, ClassVar, Dict, List, Optional
from uuid import uuid4


def _record_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def _now() -> str:
    return datetime.now().isoformat()


@dataclass
class SearchParameter:
    name: str
    parameter_type: str
    low: Optional[float] = None
    high: Optional[float] = None
    choices: List[Any] = field(default_factory=list)
    scale: str = "linear"
    condition: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchSpace:
    parameters: List[SearchParameter]
    constraints: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "constraints": self.constraints,
        }


@dataclass
class Objective:
    metric: str
    mode: str = "min"
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrialBudget:
    stage: str
    epochs: Optional[int] = None
    data_fraction: Optional[float] = None
    max_duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyProposal:
    """Structured LLM proposal. It is advisory until approved by the service layer."""

    LLM_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "action",
        "requested_strategy",
        "requested_sampler",
        "sampler_config",
        "requested_pruner",
        "search_space",
        "budgets",
        "max_training_runs",
        "initial_trial_count",
        "promotion_limits",
        "reduction_factor",
        "hypotheses",
        "candidate_proposals",
        "reason_codes",
        "evidence",
        "expected_effect",
        "confidence",
    })

    action: str
    requested_strategy: Optional[str] = None
    requested_sampler: Optional[str] = None
    sampler_config: Dict[str, Any] = field(default_factory=dict)
    requested_pruner: Optional[str] = None
    search_space: Optional[Dict[str, Any]] = None
    budgets: Optional[List[Dict[str, Any]]] = None
    max_training_runs: Optional[int] = None
    initial_trial_count: Optional[int] = None
    promotion_limits: Optional[List[int]] = None
    reduction_factor: Optional[int] = None
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    candidate_proposals: List[Dict[str, Any]] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    expected_effect: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    proposal_id: str = field(default_factory=lambda: _record_id("proposal"))
    created_at: str = field(default_factory=_now)

    @classmethod
    def from_dict(
        cls,
        value: Dict[str, Any],
        *,
        preserve_audit_fields: bool = False,
        strict_llm: bool = False,
    ) -> "StrategyProposal":
        if not isinstance(value, dict):
            raise ValueError("strategy proposal must be a JSON object")
        if strict_llm:
            cls._validate_llm_payload(value)
        proposal = cls(
            action=str(value.get("action") or ""),
            requested_strategy=value.get("requested_strategy"),
            requested_sampler=value.get("requested_sampler"),
            sampler_config=dict(value.get("sampler_config") or {}),
            requested_pruner=value.get("requested_pruner"),
            search_space=value.get("search_space"),
            budgets=value.get("budgets"),
            max_training_runs=value.get("max_training_runs"),
            initial_trial_count=value.get("initial_trial_count"),
            promotion_limits=value.get("promotion_limits"),
            reduction_factor=value.get("reduction_factor"),
            hypotheses=[
                dict(item) for item in (value.get("hypotheses") or [])
                if isinstance(item, dict)
            ],
            candidate_proposals=[
                dict(item) for item in (value.get("candidate_proposals") or [])
                if isinstance(item, dict)
            ],
            reason_codes=list(value.get("reason_codes") or []),
            evidence=dict(value.get("evidence") or {}),
            expected_effect=dict(value.get("expected_effect") or {}),
            confidence=value.get("confidence"),
        )
        # IDs and timestamps supplied by an LLM are untrusted.  Internal policy
        # transformations may opt in to preserving already-issued audit fields.
        if preserve_audit_fields and value.get("proposal_id"):
            proposal.proposal_id = str(value["proposal_id"])
        if preserve_audit_fields and value.get("created_at"):
            proposal.created_at = str(value["created_at"])
        return proposal

    @classmethod
    def _validate_llm_payload(cls, value: Dict[str, Any]) -> None:
        """Reject malformed model output before it reaches policy validation."""
        fields = set(value)
        missing = sorted(cls.LLM_FIELDS - fields)
        unknown = sorted(fields - cls.LLM_FIELDS)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing fields: " + ", ".join(missing))
            if unknown:
                details.append("unknown fields: " + ", ".join(unknown))
            raise ValueError("invalid proposal structure (" + "; ".join(details) + ")")
        if not isinstance(value["action"], str) or not value["action"].strip():
            raise ValueError("proposal action must be a nonempty string")
        if value["requested_strategy"] is not None:
            raise ValueError("requested_strategy is legacy and must be null")
        for name in ("requested_sampler", "requested_pruner"):
            if value[name] is not None and not isinstance(value[name], str):
                raise ValueError(f"{name} must be a string or null")
        for name in ("sampler_config", "evidence", "expected_effect"):
            if not isinstance(value[name], dict):
                raise ValueError(f"{name} must be an object")
        if value["search_space"] is not None and not isinstance(
            value["search_space"], dict
        ):
            raise ValueError("search_space must be an object or null")
        if value["budgets"] is not None and (
            not isinstance(value["budgets"], list)
            or any(not isinstance(item, dict) for item in value["budgets"])
        ):
            raise ValueError("budgets must be a list of objects or null")
        for name in (
            "max_training_runs",
            "initial_trial_count",
            "reduction_factor",
        ):
            if value[name] is not None and type(value[name]) is not int:
                raise ValueError(f"{name} must be an integer or null")
        if value["promotion_limits"] is not None and (
            not isinstance(value["promotion_limits"], list)
            or any(type(item) is not int for item in value["promotion_limits"])
        ):
            raise ValueError("promotion_limits must be a list of integers or null")
        for name in ("hypotheses", "candidate_proposals"):
            if not isinstance(value[name], list) or any(
                not isinstance(item, dict) for item in value[name]
            ):
                raise ValueError(f"{name} must be a list of objects")
        if not isinstance(value["reason_codes"], list) or any(
            not isinstance(item, str) or not item
            for item in value["reason_codes"]
        ):
            raise ValueError("reason_codes must be a list of nonempty strings")
        confidence = value["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            raise ValueError("confidence must be a finite number in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyDecisionRecord:
    """Audit record of the deterministic decision applied before Study creation."""

    decision: str
    adopted_strategy: str
    adopted_search_space: Dict[str, Any]
    adopted_budgets: List[Dict[str, Any]]
    adopted_max_training_runs: int
    adopted_sampler: Optional[str] = None
    adopted_sampler_config: Dict[str, Any] = field(default_factory=dict)
    adopted_pruner: str = "none"
    adopted_initial_trial_count: Optional[int] = None
    adopted_promotion_limits: List[int] = field(default_factory=list)
    adopted_reduction_factor: int = 3
    proposal_id: Optional[str] = None
    proposal: Optional[Dict[str, Any]] = None
    accepted_fields: List[str] = field(default_factory=list)
    rejected_fields: List[Dict[str, Any]] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    scope: str = "study_planning"
    applied: bool = False
    effective_from_trial_index: Optional[int] = None
    affected_trial_ids: List[str] = field(default_factory=list)
    realized_outcome: Dict[str, Any] = field(default_factory=dict)
    effect_estimate: Dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(default_factory=lambda: _record_id("decision"))
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trial:
    trial_id: str
    parameters: Dict[str, Any]
    budget: TrialBudget
    status: str = "suggested"
    parent_trial_id: Optional[str] = None
    rung: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    intermediate_metrics: List[Dict[str, Any]] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None
    candidate_source: str = "optimizer"
    search_phase: int = 0
    proposal_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["budget"] = self.budget.to_dict()
        return data


@dataclass
class HPOStudy:
    study_id: str
    experiment_id: str
    strategy: str
    search_space: SearchSpace
    objectives: List[Objective]
    budgets: List[TrialBudget]
    sampler_strategy: Optional[str] = None
    pruner_strategy: Optional[str] = None
    candidate_strategy: Optional[str] = None
    controller_mode: str = "auto"
    reduction_factor: int = 3
    max_trials: Optional[int] = None
    initial_trial_count: Optional[int] = None
    promotion_limits: List[int] = field(default_factory=list)
    max_training_runs: Optional[int] = None
    min_completed_per_rung: int = 1
    candidate_batch_size: Optional[int] = None
    sampler_config: Dict[str, Any] = field(default_factory=dict)
    search_phases: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    pending_candidate_proposals: List[Dict[str, Any]] = field(default_factory=list)
    candidate_proposal_reviews: List[Dict[str, Any]] = field(default_factory=list)
    candidate_generation_reviews: List[Dict[str, Any]] = field(default_factory=list)
    scheduler_state: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    strategy_reviews: List[Dict[str, Any]] = field(default_factory=list)
    next_study_proposal: Optional[Dict[str, Any]] = None
    warm_start_trials: List[Dict[str, Any]] = field(default_factory=list)
    history_context: Dict[str, Any] = field(default_factory=dict)
    trial_ids: List[str] = field(default_factory=list)
    best_trial_id: Optional[str] = None
    status: str = "created"
    stop_reason: Optional[str] = None
    random_seed: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "search_space": self.search_space.to_dict(),
            "objectives": [objective.to_dict() for objective in self.objectives],
            "budgets": [budget.to_dict() for budget in self.budgets],
        }


@dataclass
class OptimizationCampaign:
    objective: Objective
    target_value: Optional[float] = None
    max_studies: int = 1
    patience: int = 1
    min_improvement: float = 0.0
    max_total_training_runs: Optional[int] = None
    confirmation_signature: Dict[str, Any] = field(default_factory=dict)
    campaign_id: str = field(default_factory=lambda: _record_id("campaign"))
    study_summaries: List[Dict[str, Any]] = field(default_factory=list)
    best_value: Optional[float] = None
    best_experiment_id: Optional[str] = None
    status: str = "running"
    stop_reason: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["objective"] = self.objective.to_dict()
        return data
