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
        raise ValueError(f"Prompt template must be a JSON object: {template_path}")
    return value


def render_prompt(name: str, **fields: Any) -> str:
    """Render a JSON prompt by merging runtime fields into a named template."""
    payload = copy.deepcopy(load_prompt_template(name))
    payload.update(fields)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))