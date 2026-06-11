"""Candidate generation and promotion strategies."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

from .contracts import Objective, SearchParameter, SearchSpace, Trial


class RandomSearchStrategy:
    strategy_name = "random_search"

    def suggest(
        self,
        search_space: SearchSpace,
        count: int,
        *,
        seed: int = 0,
        existing: List[Dict[str, Any]] | None = None,
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


class SuccessiveHalvingStrategy:
    strategy_name = "successive_halving"

    def promote(
        self,
        trials: List[Trial],
        objective: Objective,
        reduction_factor: int = 3,
    ) -> List[Trial]:
        eligible = [
            trial for trial in trials
            if trial.status == "completed"
            and objective.metric in trial.metrics
        ]
        reverse = objective.mode == "max"
        eligible.sort(key=lambda trial: trial.metrics[objective.metric], reverse=reverse)
        keep = max(1, math.ceil(len(eligible) / max(reduction_factor, 2)))
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
