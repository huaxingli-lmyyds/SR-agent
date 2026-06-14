"""Candidate generation and promotion strategies."""

from __future__ import annotations

import math
import random
from itertools import product
from typing import Any, Dict, List, Optional, Protocol

from .contracts import Objective, SearchParameter, SearchSpace, Trial


class CandidateStrategy(Protocol):
    strategy_name: str

    def validate(self, search_space: SearchSpace) -> None:
        ...

    def suggest(
        self,
        search_space: SearchSpace,
        count: int,
        *,
        seed: int = 0,
        existing: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Trial]] = None,
        objective: Optional[Objective] = None,
    ) -> List[Dict[str, Any]]:
        ...


class RandomSearchStrategy:
    strategy_name = "random_search"

    @staticmethod
    def validate(search_space: SearchSpace) -> None:
        for parameter in search_space.parameters:
            _validate_sampled_parameter(parameter)

    def suggest(
        self,
        search_space: SearchSpace,
        count: int,
        *,
        seed: int = 0,
        existing: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Trial]] = None,
        objective: Optional[Objective] = None,
    ) -> List[Dict[str, Any]]:
        rng = random.Random(seed)
        seen = {_signature(item) for item in (existing or [])}
        suggestions: List[Dict[str, Any]] = []
        attempts = 0
        while len(suggestions) < count and attempts < max(100, count * 50):
            attempts += 1
            candidate: Dict[str, Any] = {}
            for parameter in search_space.parameters:
                if _condition_matches(parameter.condition, candidate):
                    candidate[parameter.name] = _sample_parameter(parameter, rng)
            signature = _signature(candidate)
            if signature in seen or not _constraints_match(candidate, search_space.constraints):
                continue
            seen.add(signature)
            suggestions.append(candidate)
        return suggestions


class GridSearchStrategy:
    strategy_name = "grid_search"

    def validate(self, search_space: SearchSpace) -> None:
        for parameter in search_space.parameters:
            _grid_values(parameter)

    def suggest(
        self,
        search_space: SearchSpace,
        count: int,
        *,
        seed: int = 0,
        existing: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Trial]] = None,
        objective: Optional[Objective] = None,
    ) -> List[Dict[str, Any]]:
        seen = {_signature(item) for item in (existing or [])}
        names = [parameter.name for parameter in search_space.parameters]
        values = [_grid_values(parameter) for parameter in search_space.parameters]
        suggestions = []
        for combination in product(*values):
            candidate = dict(zip(names, combination))
            if _signature(candidate) in seen or not _constraints_match(candidate, search_space.constraints):
                continue
            suggestions.append(candidate)
            if len(suggestions) >= count:
                break
        return suggestions


class AdaptiveSearchStrategy:
    """Deterministically explore around the best completed trial."""

    strategy_name = "adaptive_search"

    @staticmethod
    def validate(search_space: SearchSpace) -> None:
        for parameter in search_space.parameters:
            _validate_sampled_parameter(parameter)

    def suggest(
        self,
        search_space: SearchSpace,
        count: int,
        *,
        seed: int = 0,
        existing: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Trial]] = None,
        objective: Optional[Objective] = None,
    ) -> List[Dict[str, Any]]:
        completed = [
            trial for trial in (history or [])
            if objective
            and isinstance(trial.metrics.get(objective.metric), (int, float))
        ]
        if not completed:
            return RandomSearchStrategy().suggest(
                search_space,
                count,
                seed=seed,
                existing=existing,
            )
        reverse = objective.mode == "max"
        completed.sort(key=lambda trial: trial.metrics[objective.metric], reverse=reverse)
        best = completed[0].parameters
        rng = random.Random(seed)
        seen = {_signature(item) for item in (existing or [])}
        suggestions = []
        attempts = 0
        while len(suggestions) < count and attempts < max(100, count * 50):
            attempts += 1
            candidate = dict(best)
            parameter = search_space.parameters[attempts % len(search_space.parameters)]
            if parameter.parameter_type == "categorical":
                options = [item for item in parameter.choices if item != candidate.get(parameter.name)]
                if options:
                    candidate[parameter.name] = options[(attempts + seed) % len(options)]
            else:
                if parameter.low is None or parameter.high is None:
                    continue
                span = parameter.high - parameter.low
                direction = -1 if attempts % 2 else 1
                value = float(candidate.get(parameter.name, parameter.low)) + direction * span * 0.1
                value = min(max(value, parameter.low), parameter.high)
                candidate[parameter.name] = int(round(value)) if parameter.parameter_type == "int" else value
            signature = _signature(candidate)
            if signature in seen or not _constraints_match(candidate, search_space.constraints):
                continue
            seen.add(signature)
            suggestions.append(candidate)
        if len(suggestions) < count:
            suggestions.extend(RandomSearchStrategy().suggest(
                search_space,
                count - len(suggestions),
                seed=rng.randint(0, 2**31 - 1),
                existing=[*(existing or []), *suggestions],
            ))
        return suggestions


class CandidateStrategyRegistry:
    def __init__(self) -> None:
        self._strategies: Dict[str, CandidateStrategy] = {}

    def register(self, strategy: CandidateStrategy) -> None:
        self._strategies[strategy.strategy_name] = strategy

    def get(self, name: str) -> CandidateStrategy:
        candidate_name = "random_search" if name == "successive_halving" else name
        try:
            return self._strategies[candidate_name]
        except KeyError as exc:
            available = ", ".join(sorted([*self._strategies, "successive_halving"]))
            raise ValueError(f"unsupported HPO strategy: {name}; available: {available}") from exc

    def names(self) -> List[str]:
        return sorted([*self._strategies, "successive_halving"])


STRATEGIES = CandidateStrategyRegistry()
STRATEGIES.register(RandomSearchStrategy())
STRATEGIES.register(GridSearchStrategy())
STRATEGIES.register(AdaptiveSearchStrategy())


class SuccessiveHalvingStrategy:
    strategy_name = "successive_halving"

    def promote(
        self,
        trials: List[Trial],
        objective: Objective,
        reduction_factor: int = 3,
        *,
        rung: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Trial]:
        eligible = [
            trial for trial in trials
            if trial.status == "completed"
            and objective.metric in trial.metrics
            and (rung is None or trial.rung == rung)
        ]
        reverse = objective.mode == "max"
        eligible.sort(key=lambda trial: trial.metrics[objective.metric], reverse=reverse)
        keep = max(1, math.ceil(len(eligible) / max(reduction_factor, 2)))
        if limit is not None:
            keep = min(keep, max(limit, 0))
        return eligible[:keep]


def _sample_parameter(parameter: SearchParameter, rng: random.Random) -> Any:
    if parameter.parameter_type == "categorical":
        if not parameter.choices:
            raise ValueError(f"choices are required for {parameter.name}")
        return rng.choice(parameter.choices)
    if parameter.low is None or parameter.high is None:
        raise ValueError(f"low/high are required for {parameter.name}")
    if parameter.scale == "log":
        value = math.exp(rng.uniform(math.log(parameter.low), math.log(parameter.high)))
    else:
        value = rng.uniform(parameter.low, parameter.high)
    if parameter.parameter_type == "int":
        return int(round(value))
    if parameter.parameter_type == "float":
        return float(value)
    raise ValueError(f"unsupported parameter type: {parameter.parameter_type}")


def _validate_sampled_parameter(parameter: SearchParameter) -> None:
    if parameter.parameter_type == "categorical":
        if not parameter.choices:
            raise ValueError(f"choices are required for {parameter.name}")
        return
    if parameter.parameter_type not in {"int", "float"}:
        raise ValueError(f"unsupported parameter type: {parameter.parameter_type}")
    if parameter.low is None or parameter.high is None:
        raise ValueError(f"low/high are required for {parameter.name}")
    if parameter.low > parameter.high:
        raise ValueError(f"low cannot exceed high for {parameter.name}")
    if parameter.scale == "log" and parameter.low <= 0:
        raise ValueError(f"log-scale low must be positive for {parameter.name}")


def _grid_values(parameter: SearchParameter) -> List[Any]:
    if parameter.choices:
        return list(parameter.choices)
    if parameter.parameter_type == "int" and parameter.low is not None and parameter.high is not None:
        return list(range(int(parameter.low), int(parameter.high) + 1))
    raise ValueError(
        f"grid_search requires choices or an integer low/high range for {parameter.name}"
    )


def _condition_matches(condition: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    return not condition or all(candidate.get(key) == value for key, value in condition.items())


def _constraints_match(candidate: Dict[str, Any], constraints: List[Dict[str, Any]]) -> bool:
    for constraint in constraints:
        parameter = constraint.get("parameter")
        operator = constraint.get("operator")
        value = constraint.get("value")
        current = candidate.get(parameter)
        if current is None:
            continue
        if operator == "lte" and not current <= value:
            return False
        if operator == "gte" and not current >= value:
            return False
        if operator == "eq" and not current == value:
            return False
        if operator == "in" and not current in value:
            return False
    return True


def _signature(candidate: Dict[str, Any]) -> tuple:
    return tuple(sorted(candidate.items()))
