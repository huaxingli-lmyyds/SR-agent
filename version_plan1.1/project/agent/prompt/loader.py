"""Load JSON prompt templates from the agent prompt directory."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

_PROMPT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def load_prompt_template(name: str) -> Dict[str, Any]:
    """Return a prompt template by name without caller-specific context."""
    template_path = _PROMPT_DIR / f"{name}.json"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    with template_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(
            f"Prompt template must be a JSON object: {template_path}"
        )
    return value


def render_prompt(name: str, **fields: Any) -> str:
    """Render a JSON prompt by merging runtime fields into a named template."""
    payload = copy.deepcopy(load_prompt_template(name))
    payload.update(fields)
    if name == "hpo_strategy_proposal":
        _specialize_hpo(payload)
    elif name == "hpo_cross_stage_proposal":
        phase = payload["context"]["phase"]
        profile = payload.pop("stage_profiles")[phase]
        payload["response_contract"].update(phase=phase, allowed_actions=profile["allowed_actions"])
        payload["rules"] += profile["rules"]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _specialize_hpo(payload: Dict[str, Any]) -> None:
    """Send only rules and examples that apply to this decision stage."""
    context = payload.get("context") or {}
    probe = (context.get("feedback") or {}).get("hypothesis_probe") or {}
    phase = (
        "hypothesis_probe"
        if probe.get("enabled")
        else context.get("phase", "runtime_review")
    )
    profiles = payload.pop("stage_profiles")
    profile = profiles.get(phase, profiles["runtime_review"])
    constraints = context.get("execution_constraints") or {}
    locked = set(profile["locked_fields"]) | set(
        constraints.get("locked_fields") or []
    )
    allowed = list(context.get("available_samplers") or [])
    if "allowed_samplers" in constraints:
        allowed = [
            name for name in allowed if name in constraints["allowed_samplers"]
        ]
    if constraints.get("agent_proposal_allowed") is False:
        allowed = [name for name in allowed if name != "agent_proposal"]
        locked.update({"hypotheses", "candidate_proposals"})
    if phase == "hypothesis_probe":
        allowed = [name for name in allowed if name == "agent_proposal"]
    contract = payload["response_contract"]
    contract.update(
        phase=phase,
        allowed_actions=profile["allowed_actions"],
        allowed_samplers=allowed,
        required_keys=list(payload["schema"]),
        locked_values={
            key: copy.deepcopy(payload["schema"][key])
            for key in sorted(locked)
            if key in payload["schema"]
        },
    )
    if phase == "hypothesis_probe":
        contract["hypothesis_field_types"].update(
            profile["hypothesis_field_types"]
        )
        contract["maximum_candidates"] = min(
            3, max(0, int(probe.get("maximum_candidates", 0)))
        )
        contract["hypothesis_fields"] += [
            "anchor_trial_id",
            "probe_design",
            "tested_parameters",
            "falsification_criterion",
        ]
    payload["task"] = profile["task"]
    payload["rules"] += profile["rules"]
    # Avoid teaching a fixed training budget or fabricated anchor via an example.
    # Nested field definitions remain available in the contract for actual probes.
    payload["examples"] = [
        example
        for example in payload["examples"]
        if example["name"] in profile["examples"]
        and example["name"]
        not in {"planning_successive_halving", "bounded_hypothesis_probe"}
        and all(
            example["output"].get(key) == value
            for key, value in contract["locked_values"].items()
        )
        and (
            example["output"].get("requested_sampler") is None
            or example["output"]["requested_sampler"] in allowed
        )
    ]
    # A no-change example must not pretend that any particular sample count was observed.
    for example in payload["examples"]:
        if example["output"]["action"] == "keep_strategy":
            example["output"]["evidence"] = {}
            example["output"]["confidence"] = 0.0
