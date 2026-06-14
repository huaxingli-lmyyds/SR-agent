"""Model-agnostic HPO domain APIs."""

from importlib import import_module

from .contracts import HPOStudy, Objective, SearchParameter, SearchSpace, Trial, TrialBudget
from .policies import EarlyStoppingPolicy, FailureDecision, FailurePolicy, HPOPlanningPolicy, RetryPolicy, StopDecision
from .service import HPOService, search_space_from_dict
from .strategies import (
    AdaptiveSearchStrategy,
    CandidateStrategy,
    CandidateStrategyRegistry,
    GridSearchStrategy,
    RandomSearchStrategy,
    SuccessiveHalvingStrategy,
)

_WORKFLOW_EXPORTS = {
    "DecisionPolicy": ("agent.hpo.scheduler", "DecisionPolicy"),
    "HPOScheduler": ("agent.hpo.scheduler", "HPOScheduler"),
    "HPOGraphState": ("agent.hpo.scheduler", "HPOGraphState"),
    "SchedulerResult": ("agent.hpo.scheduler", "SchedulerResult"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _WORKFLOW_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "SearchParameter",
    "SearchSpace",
    "Objective",
    "TrialBudget",
    "Trial",
    "HPOStudy",
    "RandomSearchStrategy",
    "GridSearchStrategy",
    "AdaptiveSearchStrategy",
    "CandidateStrategy",
    "CandidateStrategyRegistry",
    "SuccessiveHalvingStrategy",
    "EarlyStoppingPolicy",
    "StopDecision",
    "FailureDecision",
    "FailurePolicy",
    "RetryPolicy",
    "HPOPlanningPolicy",
    "DecisionPolicy",
    "HPOScheduler",
    "HPOGraphState",
    "SchedulerResult",
    "HPOService",
    "search_space_from_dict",
]
