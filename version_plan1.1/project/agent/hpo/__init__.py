"""Model-agnostic HPO domain APIs."""

from importlib import import_module

from .contracts import (
    HPOStudy,
    Objective,
    OptimizationCampaign,
    SearchParameter,
    SearchSpace,
    StrategyDecisionRecord,
    StrategyProposal,
    Trial,
    TrialBudget,
)
from .policies import (
    EarlyStoppingPolicy,
    EvidenceGate,
    EvidenceGateResult,
    FailureDecision,
    FailurePolicy,
    HPOPlanningPolicy,
    OptimizationPlanDecisionPolicy,
    RetryPolicy,
    StopDecision,
    StrategyDecisionPolicy,
)
from .service import HPOService, search_space_from_dict
from .campaign import CampaignPolicy, study_confirmation_signature
from .feedback import HPOFeedbackAnalyzer
from .strategies import (
    AdaptiveSearchStrategy,
    AgentProposalStrategy,
    CandidateStrategy,
    CandidateStrategyRegistry,
    GridSearchStrategy,
    OptunaTPEStrategy,
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
    "OptimizationCampaign",
    "StrategyProposal",
    "StrategyDecisionRecord",
    "RandomSearchStrategy",
    "GridSearchStrategy",
    "AdaptiveSearchStrategy",
    "AgentProposalStrategy",
    "OptunaTPEStrategy",
    "CandidateStrategy",
    "CandidateStrategyRegistry",
    "SuccessiveHalvingStrategy",
    "EarlyStoppingPolicy",
    "EvidenceGate",
    "EvidenceGateResult",
    "StopDecision",
    "FailureDecision",
    "FailurePolicy",
    "RetryPolicy",
    "HPOPlanningPolicy",
    "OptimizationPlanDecisionPolicy",
    "StrategyDecisionPolicy",
    "DecisionPolicy",
    "HPOScheduler",
    "HPOGraphState",
    "SchedulerResult",
    "HPOService",
    "CampaignPolicy",
    "study_confirmation_signature",
    "HPOFeedbackAnalyzer",
    "search_space_from_dict",
]
