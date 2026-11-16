"""Candidate identity, dependency ordering and constraint validation."""

from __future__ import annotations

import json
from typing import Any

from .contracts import SearchParameter, SearchSpace


def _candidate_signature(parameters: dict[str, Any]) -> str:
    return json.dumps(
        parameters, sort_keys=True, ensure_ascii=False, default=str
    )


def _condition_matches(
    condition: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    return not condition or all(
        candidate.get(key) == value for key, value in condition.items()
    )


def _ordered_search_parameters(
    search_space: SearchSpace,
) -> list[SearchParameter]:
    """Resolve conditional parameters without relying on declaration order."""
    remaining = list(search_space.parameters)
    resolved = set()
    ordered: list[SearchParameter] = []
    while remaining:
        ready = [
            parameter
            for parameter in remaining
            if set(parameter.condition).issubset(resolved)
        ]
        if not ready:
            cycle = ", ".join(parameter.name for parameter in remaining)
            raise ValueError(
                f"cyclic or unknown search-space conditions: {cycle}"
            )
        for parameter in ready:
            remaining.remove(parameter)
            resolved.add(parameter.name)
            ordered.append(parameter)
    return ordered


def _constraints_match(
    candidate: dict[str, Any], constraints: list[dict[str, Any]]
) -> bool:
    for constraint in constraints:
        parameter = constraint.get("parameter")
        operator = constraint.get("operator")
        expected = constraint.get("value")
        current = candidate.get(parameter)
        if current is None:
            continue
        try:
            if operator == "lte":
                matches = current <= expected
            elif operator == "gte":
                matches = current >= expected
            elif operator == "eq":
                matches = current == expected
            elif operator == "in":
                matches = current in expected
            else:
                return False
            if not matches:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _validate_candidate_value(parameter: SearchParameter, value: Any) -> Any:
    if parameter.parameter_type == "categorical":
        if value not in parameter.choices:
            raise ValueError(
                f"candidate value for {parameter.name} is not one of the allowed choices"
            )
        return value
    if parameter.parameter_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"candidate value for {parameter.name} must be an integer"
            )
        normalized: Any = int(value)
    elif parameter.parameter_type == "float":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(
                f"candidate value for {parameter.name} must be numeric"
            )
        normalized = float(value)
    else:
        raise ValueError(
            f"unsupported parameter type: {parameter.parameter_type}"
        )
    if parameter.low is None or parameter.high is None:
        raise ValueError(f"low/high are required for {parameter.name}")
    if normalized < parameter.low or normalized > parameter.high:
        raise ValueError(
            f"candidate value for {parameter.name} is outside [{parameter.low}, {parameter.high}]"
        )
    return normalized
