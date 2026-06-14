from agent.hpo import (
    AdaptiveSearchStrategy,
    GridSearchStrategy,
    HPOPlanningPolicy,
    Objective,
    SearchParameter,
    SearchSpace,
    Trial,
    TrialBudget,
)


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
    available = ["random_search", "grid_search", "adaptive_search", "successive_halving"]

    assert policy.select_strategy("auto", grid, [TrialBudget("full")], 2, available) == "grid_search"
    assert policy.select_strategy("auto", continuous, [TrialBudget("full")], 3, available) == "random_search"
    assert policy.select_strategy("auto", continuous, [TrialBudget("full")], 5, available) == "adaptive_search"
    assert policy.select_strategy(
        "auto",
        continuous,
        [TrialBudget("small"), TrialBudget("large")],
        5,
        available,
    ) == "successive_halving"
