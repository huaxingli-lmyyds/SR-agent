"""Candidate generation and promotion strategies."""

from __future__ import annotations

import math
import random
from itertools import product
from typing import Any, Dict, List, Optional, Protocol

from agent.core.metrics import is_finite_metric
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
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ...


class AgentProposalStrategy:
    """Use validated candidates proposed by the HPO agent as one standalone sampler."""

    strategy_name = "agent_proposal"

    @staticmethod
    def validate(search_space: SearchSpace) -> None:
        _ordered_parameters(search_space)
        for parameter in search_space.parameters:
            _validate_sampled_parameter(parameter)

    def suggest(
        self,
        search_space: SearchSpace,
        count: int,
        *,
        proposed_candidates: Optional[List[Dict[str, Any]]] = None,
        seed: int = 0,
        existing: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Trial]] = None,
        objective: Optional[Objective] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if count <= 0:
            return []
        seen = {_signature(item) for item in (existing or [])}
        suggestions: List[Dict[str, Any]] = []
        for item in proposed_candidates or []:
            parameters = dict(item.get("parameters") or {})
            signature = _signature(parameters)
            if not parameters or signature in seen:
                continue
            seen.add(signature)
            suggestions.append(parameters)
            if len(suggestions) >= count:
                break
        return suggestions[:count]

class RandomSearchStrategy:
    strategy_name = "random_search"

    @staticmethod
    def validate(search_space: SearchSpace) -> None:
        _ordered_parameters(search_space)
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
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        rng = random.Random(seed)
        seen = {_signature(item) for item in (existing or [])}
        suggestions: List[Dict[str, Any]] = []
        attempts = 0
        while len(suggestions) < count and attempts < max(100, count * 50):
            attempts += 1
            candidate: Dict[str, Any] = {}
            for parameter in _ordered_parameters(search_space):
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
        _ordered_parameters(search_space)
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
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        seen = {_signature(item) for item in (existing or [])}
        parameters = _ordered_parameters(search_space)
        names = [parameter.name for parameter in parameters]
        values = [_grid_values(parameter) for parameter in parameters]
        suggestions = []
        for combination in product(*values):
            raw = dict(zip(names, combination))
            candidate: Dict[str, Any] = {}
            for parameter in parameters:
                if _condition_matches(parameter.condition, candidate):
                    candidate[parameter.name] = raw[parameter.name]
            if _signature(candidate) in seen or not _constraints_match(candidate, search_space.constraints):
                continue
            seen.add(_signature(candidate))
            suggestions.append(candidate)
            if len(suggestions) >= count:
                break
        return suggestions


class AdaptiveSearchStrategy:
    """Deterministically explore around the best completed trial."""

    strategy_name = "adaptive_search"

    @staticmethod
    def validate(search_space: SearchSpace) -> None:
        _ordered_parameters(search_space)
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
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        completed = [
            trial for trial in (history or [])
            if trial.status in {"completed", "promoted"}
            and objective
            and is_finite_metric(trial.metrics.get(objective.metric))
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
        parameters = _ordered_parameters(search_space)
        while len(suggestions) < count and attempts < max(100, count * 50):
            attempts += 1
            candidate = dict(best)
            parameter = parameters[attempts % len(parameters)]
            if parameter.parameter_type == "categorical":
                value = _nearby_categorical_value(parameter, candidate.get(parameter.name), attempts + seed)
                if value is not None:
                    candidate[parameter.name] = value
            else:
                value = _nearby_numeric_value(parameter, candidate.get(parameter.name), attempts)
                if value is None:
                    continue
                candidate[parameter.name] = value
            candidate = _normalize_conditional_candidate(search_space, candidate, rng)
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


def _nearby_categorical_value(parameter: SearchParameter, current: Any, offset: int) -> Any:
    choices = list(parameter.choices or [])
    if len(choices) <= 1:
        return None
    try:
        ordered = sorted(choices)
    except TypeError:
        ordered = choices
    if current not in ordered:
        return ordered[offset % len(ordered)]
    index = ordered.index(current)
    step = -1 if offset % 2 else 1
    next_index = min(max(index + step, 0), len(ordered) - 1)
    if next_index == index:
        next_index = min(max(index - step, 0), len(ordered) - 1)
    return ordered[next_index] if next_index != index else None


def _nearby_numeric_value(parameter: SearchParameter, current: Any, offset: int) -> Any:
    if parameter.low is None or parameter.high is None:
        return None
    low = float(parameter.low)
    high = float(parameter.high)
    if high < low:
        return None
    value = float(current if current is not None else low)
    direction = -1 if offset % 2 else 1
    if parameter.scale == "log" and low > 0 and high > 0:
        ratio = high / low
        factor = ratio ** 0.15
        value = value / factor if direction < 0 else value * factor
    else:
        value = value + direction * (high - low) * 0.1
    value = min(max(value, low), high)
    return int(round(value)) if parameter.parameter_type == "int" else value


class OptunaTPEStrategy:
    """Use Optuna TPE for candidate sampling while keeping lifecycle state in HPOService."""

    strategy_name = "tpe"

    @staticmethod
    def is_available() -> bool:
        try:
            import optuna  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def validate(search_space: SearchSpace) -> None:
        _ordered_parameters(search_space)
        for parameter in search_space.parameters:
            _validate_sampled_parameter(parameter)
        _import_optuna()

    def suggest(
        self,
        search_space: SearchSpace,
        count: int,
        *,
        seed: int = 0,
        existing: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Trial]] = None,
        objective: Optional[Objective] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        optuna = _import_optuna()
        sampler_config = dict(config or {})
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            multivariate=bool(sampler_config.get("multivariate", False)),
            n_startup_trials=int(sampler_config.get("n_startup_trials", 3)),
        )
        study = optuna.create_study(
            direction="maximize" if objective and objective.mode == "max" else "minimize",
            sampler=sampler,
        )
        for item in history or []:
            if (
                item.status not in {"completed", "promoted"}
                or not objective
                or not is_finite_metric(item.metrics.get(objective.metric))
            ):
                continue
            params, distributions = _optuna_history(item.parameters, search_space, optuna)
            if not params:
                continue
            try:
                study.add_trial(optuna.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    value=float(item.metrics[objective.metric]),
                ))
            except ValueError:
                # Revised search boundaries can make older Trial parameters incompatible.
                continue

        seen = {_signature(item) for item in (existing or [])}
        suggestions: List[Dict[str, Any]] = []
        attempts = 0
        while len(suggestions) < count and attempts < max(100, count * 50):
            attempts += 1
            trial = study.ask()
            candidate: Dict[str, Any] = {}
            for parameter in _ordered_parameters(search_space):
                if _condition_matches(parameter.condition, candidate):
                    candidate[parameter.name] = _optuna_suggest(trial, parameter)
            signature = _signature(candidate)
            if signature in seen or not _constraints_match(candidate, search_space.constraints):
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                continue
            seen.add(signature)
            suggestions.append(candidate)
        return suggestions


class CandidateStrategyRegistry:
    def __init__(self) -> None:
        self._strategies: Dict[str, CandidateStrategy] = {}

    def register(self, strategy: CandidateStrategy) -> None:
        self._strategies[strategy.strategy_name] = strategy

    def get(self, name: str) -> CandidateStrategy:
        candidate_name = "random_search" if name == "successive_halving" else name
        try:
            strategy = self._strategies[candidate_name]
        except KeyError as exc:
            available = ", ".join(sorted([*self._strategies, "successive_halving"]))
            raise ValueError(f"unsupported HPO strategy: {name}; available: {available}") from exc
        if not self._is_available(strategy):
            raise ValueError(f"HPO strategy is unavailable because an optional dependency is missing: {name}")
        return strategy

    def names(self) -> List[str]:
        available = [
            name for name, strategy in self._strategies.items()
            if self._is_available(strategy)
        ]
        return sorted([*available, "successive_halving"])

    @staticmethod
    def _is_available(strategy: CandidateStrategy) -> bool:
        check = getattr(strategy, "is_available", None)
        return bool(check()) if callable(check) else True


STRATEGIES = CandidateStrategyRegistry()
STRATEGIES.register(RandomSearchStrategy())
STRATEGIES.register(GridSearchStrategy())
STRATEGIES.register(AdaptiveSearchStrategy())
STRATEGIES.register(AgentProposalStrategy())
STRATEGIES.register(OptunaTPEStrategy())


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
            if trial.status in {"completed", "promoted"}
            and is_finite_metric(trial.metrics.get(objective.metric))
            and (rung is None or trial.rung == rung)
        ]
        reverse = objective.mode == "max"
        eligible.sort(key=lambda trial: trial.metrics[objective.metric], reverse=reverse)
        keep = max(1, math.ceil(len(eligible) / max(reduction_factor, 2)))
        # Rank the original cohort, including already committed promotions.
        # Retrying a partial batch must not shrink its quota or promote losers.
        pending = [trial for trial in eligible[:keep] if trial.status == "completed"]
        if limit is not None:
            pending = pending[:max(limit, 0)]
        return pending


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


def _import_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "the tpe strategy requires Optuna; install the project dependencies or run: pip install optuna"
        ) from exc
    return optuna


def _optuna_suggest(trial: Any, parameter: SearchParameter) -> Any:
    if parameter.parameter_type == "categorical":
        return trial.suggest_categorical(parameter.name, parameter.choices)
    if parameter.parameter_type == "int":
        return trial.suggest_int(
            parameter.name,
            int(parameter.low),
            int(parameter.high),
            log=parameter.scale == "log",
        )
    return trial.suggest_float(
        parameter.name,
        float(parameter.low),
        float(parameter.high),
        log=parameter.scale == "log",
    )


def _optuna_history(parameters: Dict[str, Any], search_space: SearchSpace, optuna: Any) -> tuple:
    params: Dict[str, Any] = {}
    distributions: Dict[str, Any] = {}
    for parameter in _ordered_parameters(search_space):
        if parameter.name not in parameters or not _condition_matches(parameter.condition, parameters):
            continue
        params[parameter.name] = parameters[parameter.name]
        if parameter.parameter_type == "categorical":
            distributions[parameter.name] = optuna.distributions.CategoricalDistribution(parameter.choices)
        elif parameter.parameter_type == "int":
            distributions[parameter.name] = optuna.distributions.IntDistribution(
                int(parameter.low),
                int(parameter.high),
                log=parameter.scale == "log",
            )
        else:
            distributions[parameter.name] = optuna.distributions.FloatDistribution(
                float(parameter.low),
                float(parameter.high),
                log=parameter.scale == "log",
            )
    return params, distributions


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


def _ordered_parameters(search_space: SearchSpace) -> List[SearchParameter]:
    """Return a stable dependency order for conditional parameters."""
    names = [parameter.name for parameter in search_space.parameters]
    if len(names) != len(set(names)):
        raise ValueError("search space parameter names must be unique")
    known = set(names)
    for parameter in search_space.parameters:
        unknown = sorted(set(parameter.condition) - known)
        if unknown:
            raise ValueError(
                f"condition for {parameter.name} references unknown parameters: "
                f"{', '.join(unknown)}"
            )
    remaining = list(search_space.parameters)
    resolved = set()
    ordered: List[SearchParameter] = []
    while remaining:
        ready = [
            parameter for parameter in remaining
            if set(parameter.condition).issubset(resolved)
        ]
        if not ready:
            cycle = ", ".join(parameter.name for parameter in remaining)
            raise ValueError(f"cyclic search-space conditions: {cycle}")
        for parameter in ready:
            remaining.remove(parameter)
            resolved.add(parameter.name)
            ordered.append(parameter)
    return ordered


def _normalize_conditional_candidate(
    search_space: SearchSpace,
    values: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    candidate: Dict[str, Any] = {}
    for parameter in _ordered_parameters(search_space):
        if not _condition_matches(parameter.condition, candidate):
            continue
        value = values.get(parameter.name)
        if parameter.parameter_type == "categorical":
            candidate[parameter.name] = (
                value if value in parameter.choices else _sample_parameter(parameter, rng)
            )
            continue
        if isinstance(value, (int, float)) and parameter.low is not None and parameter.high is not None:
            numeric = min(max(float(value), float(parameter.low)), float(parameter.high))
            candidate[parameter.name] = int(round(numeric)) if parameter.parameter_type == "int" else numeric
        else:
            candidate[parameter.name] = _sample_parameter(parameter, rng)
    return candidate


def _constraints_match(candidate: Dict[str, Any], constraints: List[Dict[str, Any]]) -> bool:
    for constraint in constraints:
        parameter = constraint.get("parameter")
        operator = constraint.get("operator")
        value = constraint.get("value")
        current = candidate.get(parameter)
        if current is None:
            continue
        try:
            if operator == "lte":
                matches = current <= value
            elif operator == "gte":
                matches = current >= value
            elif operator == "eq":
                matches = current == value
            elif operator == "in":
                matches = current in value
            else:
                return False
            if not matches:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _signature(candidate: Dict[str, Any]) -> tuple:
    return tuple(sorted(candidate.items()))
