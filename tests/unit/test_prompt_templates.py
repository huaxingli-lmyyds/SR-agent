import json

import pytest

from agent.hpo import StrategyProposal
from agent.prompt import load_prompt_template, render_prompt


@pytest.mark.parametrize(
    ("name", "required_schema_keys"),
    [
        (
            "hpo_strategy_proposal",
            {
                "action",
                "requested_strategy",
                "search_space",
                "budgets",
                "max_training_runs",
                "hypotheses",
                "candidate_proposals",
                "sampler_config",
            },
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

def test_hpo_prompt_examples_are_schema_exact_and_parseable() -> None:
    template = load_prompt_template("hpo_strategy_proposal")
    schema_keys = set(template["schema"])

    assert len(template["examples"]) >= 4
    for example in template["examples"]:
        output = example["output"]
        assert set(output) == schema_keys
        proposal = StrategyProposal.from_dict(output)
        assert proposal.action == output["action"]
        assert proposal.requested_strategy is None
        if output["search_space"] is not None:
            assert all(
                "parameter_type" in parameter and "type" not in parameter
                for parameter in output["search_space"]["parameters"]
            )

    halving = next(
        item["output"]
        for item in template["examples"]
        if item["name"] == "plan_resource_constrained_successive_halving"
    )
    assert halving["initial_trial_count"] + sum(halving["promotion_limits"]) <= halving["max_training_runs"]
    assert len(halving["promotion_limits"]) <= len(halving["budgets"]) - 1

    single = next(
        item["output"]
        for item in template["examples"]
        if item["name"] == "switch_to_single_fidelity_tpe"
    )
    assert single["requested_pruner"] == "none"
    assert len(single["budgets"]) == 1
    assert single["promotion_limits"] == []


def test_hpo_prompt_declares_gpu_hours_as_a_hard_ceiling() -> None:
    template = load_prompt_template("hpo_strategy_proposal")
    rules = " ".join(template["rules"])

    assert "declared_limits.gpu_hours" in rules
    assert "hard ceilings" in rules
    assert "GPU-hours mean GPU count multiplied by elapsed GPU time" in rules
    assert template["decision_priority"][0].startswith("Satisfy explicit GPU-hours")
