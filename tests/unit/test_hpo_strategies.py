from agent.hpo import (
    AdaptiveSearchStrategy,
    GridSearchStrategy,
    HPOPlanningPolicy,
    Objective,
    OptunaTPEStrategy,
    SearchParameter,
    SearchSpace,
    Trial,
    TrialBudget,
)
from agent.hpo.strategies import CandidateStrategyRegistry


def test_grid_search_enumerates_unique_combinations() -> None:
    space = SearchSpace([
        SearchParameter("model_family", "categorical", choices=["ecapa", "resnet"]),
        SearchParameter("batch_size", "categorical", choices=[16, 32]),
    ])

    suggestions = GridSearchStrategy().suggest(space, 10)

    assert len(suggestions) == 4
    assert len({tuple(sorted(item.items())) for item in suggestions}) == 4


def test_adaptive_search_moves_around_best_completed_trial() -> None:
    space = SearchSpace([SearchParameter("lr", "float", low=0.0, high=1.0)])
    history = [
        Trial("trial_a", {"lr": 0.8}, TrialBudget("full"), status="completed", metrics={"eer": 0.2}),
        Trial("trial_b", {"lr": 0.4}, TrialBudget("full"), status="completed", metrics={"eer": 0.1}),
    ]

    suggestions = AdaptiveSearchStrategy().suggest(
        space,
        2,
        existing=[trial.parameters for trial in history],
        history=history,
        objective=Objective("eer", "min"),
    )

    assert suggestions
    assert all(item["lr"] != 0.4 for item in suggestions)


def test_auto_planning_selects_strategy_deterministically() -> None:
    policy = HPOPlanningPolicy()
    grid = SearchSpace([SearchParameter("batch_size", "categorical", choices=[16, 32])])
    continuous = SearchSpace([SearchParameter("lr", "float", low=1e-5, high=1e-2)])
    available = ["random_search", "grid_search", "adaptive_search", "tpe", "successive_halving"]

    assert policy.select_strategy("auto", grid, [TrialBudget("full")], 2, available) == "grid_search"
    assert policy.select_strategy("auto", continuous, [TrialBudget("full")], 3, available) == "random_search"
    assert policy.select_strategy("auto", continuous, [TrialBudget("full")], 5, available) == "adaptive_search"
    assert policy.select_strategy("auto", continuous, [TrialBudget("full")], 8, available) == "tpe"
    assert policy.select_strategy(
        "auto",
        continuous,
        [TrialBudget("small"), TrialBudget("large")],
        5,
        available,
    ) == "successive_halving"


def test_optuna_tpe_uses_completed_history_and_returns_unique_candidates() -> None:
    import pytest

    pytest.importorskip("optuna")
    space = SearchSpace([
        SearchParameter("lr", "float", low=1e-5, high=1e-2, scale="log"),
        SearchParameter("batch_size", "categorical", choices=[16, 32, 64]),
    ])
    history = [
        Trial(
            f"trial_{index}",
            {"lr": 1e-4 * (index + 1), "batch_size": [16, 32, 64][index % 3]},
            TrialBudget("full"),
            status="completed",
            metrics={"eer": 0.2 - index * 0.01},
        )
        for index in range(6)
    ]

    suggestions = OptunaTPEStrategy().suggest(
        space,
        3,
        seed=7,
        existing=[trial.parameters for trial in history],
        history=history,
        objective=Objective("eer", "min"),
    )

    assert len(suggestions) == 3
    assert len({tuple(sorted(item.items())) for item in suggestions}) == 3


def test_registry_hides_unavailable_optional_strategy(monkeypatch) -> None:
    registry = CandidateStrategyRegistry()
    strategy = OptunaTPEStrategy()
    registry.register(strategy)
    monkeypatch.setattr(strategy, "is_available", lambda: False)

    assert "tpe" not in registry.names()
    with __import__("pytest").raises(ValueError, match="optional dependency"):
        registry.get("tpe")
