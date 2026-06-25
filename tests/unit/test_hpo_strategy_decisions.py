from agent.hpo import (
    HPOService,
    Objective,
    SearchParameter,
    SearchSpace,
    StrategyDecisionPolicy,
    StrategyProposal,
    TrialBudget,
)


def _review(proposal: StrategyProposal | None):
    service = HPOService()
    space = SearchSpace([SearchParameter("lr", "float", low=1e-5, high=1e-2, scale="log")])
    return StrategyDecisionPolicy().review(
        proposal,
        base_strategy="random_search",
        base_search_space=space,
        base_budgets=[TrialBudget("full", epochs=20, data_fraction=1.0)],
        hard_max_training_runs=6,
        objectives=[Objective("eer", "min")],
        available_strategies=service.available_strategies(),
        validate_plan=service.validate_study_plan,
    )


def test_valid_structured_proposal_is_applied() -> None:
    decision = _review(StrategyProposal(
        action="adjust_budget",
        requested_strategy="adaptive_search",
        budgets=[{"stage": "full", "epochs": 12, "data_fraction": 0.75}],
        max_training_runs=4,
        reason_codes=["reduce_cost"],
        confidence=0.8,
    ))

    assert decision.decision == "approved"
    assert decision.adopted_strategy == "adaptive_search"
    assert decision.adopted_budgets[0]["epochs"] == 12
    assert decision.adopted_max_training_runs == 4
    assert set(decision.accepted_fields) == {"requested_strategy", "budgets", "max_training_runs"}


def test_invalid_fields_are_rejected_and_valid_fields_are_kept() -> None:
    decision = _review(StrategyProposal(
        action="switch_strategy",
        requested_strategy="not_registered",
        max_training_runs=3,
    ))

    assert decision.decision == "approved_with_changes"
    assert decision.adopted_strategy == "random_search"
    assert decision.adopted_max_training_runs == 3
    assert decision.rejected_fields[0]["field"] == "requested_strategy"


def test_proposal_cannot_raise_hard_training_run_limit() -> None:
    decision = _review(StrategyProposal(
        action="adjust_budget",
        max_training_runs=100,
    ))

    assert decision.decision == "rejected"
    assert decision.adopted_max_training_runs == 6
    assert decision.rejected_fields[0]["field"] == "max_training_runs"


def test_invalid_search_space_falls_back_to_base_plan() -> None:
    decision = _review(StrategyProposal(
        action="refine_search_space",
        search_space={"parameters": [], "constraints": []},
    ))

    assert decision.decision == "rejected"
    assert decision.adopted_search_space["parameters"][0]["name"] == "lr"
    assert decision.rejected_fields[0]["field"] == "search_space"


def test_missing_proposal_records_base_plan_decision() -> None:
    decision = _review(None)

    assert decision.decision == "no_proposal"
    assert decision.proposal is None
    assert decision.reason_codes == ["base_plan_adopted"]


def test_interdependent_strategy_and_search_space_are_approved_together() -> None:
    service = HPOService()
    decision = StrategyDecisionPolicy().review(
        StrategyProposal(
            action="switch_strategy",
            requested_strategy="random_search",
            search_space={
                "parameters": [{"name": "lr", "parameter_type": "float", "low": 1e-5, "high": 1e-2}],
                "constraints": [],
            },
        ),
        base_strategy="grid_search",
        base_search_space=SearchSpace([SearchParameter("batch_size", "categorical", choices=[16, 32])]),
        base_budgets=[TrialBudget("full", epochs=1)],
        hard_max_training_runs=3,
        objectives=[Objective("eer", "min")],
        available_strategies=service.available_strategies(),
        validate_plan=service.validate_study_plan,
    )

    assert decision.decision == "approved"
    assert decision.adopted_strategy == "random_search"
    assert set(decision.accepted_fields) == {"requested_strategy", "search_space"}
