import json

import pytest

from agent.prompt import load_prompt_template, render_prompt


@pytest.mark.parametrize(
    ("name", "required_schema_keys"),
    [
        (
            "hpo_strategy_proposal",
            {"action", "requested_strategy", "search_space", "budgets", "max_training_runs"},
        ),
        ("data_processing_planning_advice", {"diagnostics", "suggested_operations", "notes"}),
        ("orchestration_coordination_advisor", {"diagnostics", "risks", "notes"}),
    ],
)
def test_prompt_templates_define_schema_and_json_rules(name, required_schema_keys) -> None:
    template = load_prompt_template(name)

    assert required_schema_keys <= set(template["schema"])
    assert "Return raw JSON only." in template["rules"]
    assert all(isinstance(rule, str) and rule for rule in template["rules"])


def test_render_prompt_merges_runtime_fields_without_mutating_template() -> None:
    prompt = render_prompt("hpo_strategy_proposal", context={"phase": "study_planning"})
    payload = json.loads(prompt)
    template = load_prompt_template("hpo_strategy_proposal")

    assert payload["schema"] == template["schema"]
    assert payload["rules"] == template["rules"]
    assert payload["context"] == {"phase": "study_planning"}
    rules = " ".join(payload["rules"])
    assert "cross_study_memory.local_search_anchor" in rules
    assert "resource_profile" in rules
    assert "9+3+1" in rules
    assert "```" not in prompt