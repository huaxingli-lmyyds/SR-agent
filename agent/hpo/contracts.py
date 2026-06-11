"""Model-agnostic hyperparameter optimization contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


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
    reduction_factor: int = 3
    max_trials: Optional[int] = None
    constraints: List[Dict[str, Any]] = field(default_factory=list)
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
