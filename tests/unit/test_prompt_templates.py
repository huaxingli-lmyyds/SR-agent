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
    assert payload["rules"] == template["rules"] + template["stage_profiles"]["study_planning"]["rules"]
    assert "stage_profiles" not in payload
    assert "stage_profiles" in template
    assert payload["context"] == {"phase": "study_planning"}
    rules = " ".join(payload["rules"])
    assert "cross_study_memory.local_search_anchor" in rules
    assert "resource_profile" in rules
    assert "9+3+1" in rules
    assert "```" not in prompt

def test_hpo_prompt_examples_are_schema_exact_and_parseable() -> None:
    template = load_prompt_template("hpo_strategy_proposal")
    schema_keys = set(template["schema"])

    assert schema_keys == set(StrategyProposal.LLM_FIELDS)
    assert len(template["examples"]) >= 4
    for example in template["examples"]:
        output = example["output"]
        assert set(output) == schema_keys
        proposal = StrategyProposal.from_dict(output, strict_llm=True)
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
        if item["name"] == "planning_successive_halving"
    )
    assert halving["initial_trial_count"] + sum(halving["promotion_limits"]) <= halving["max_training_runs"]
    assert len(halving["promotion_limits"]) <= len(halving["budgets"]) - 1

    single = next(
        item["output"]
        for item in template["examples"]
        if item["name"] == "runtime_switch_to_tpe"
    )
    assert single["requested_pruner"] is None
    assert single["budgets"] is None
    assert single["promotion_limits"] is None


def test_hpo_prompt_strict_parser_rejects_missing_extra_and_wrong_types() -> None:
    template = load_prompt_template("hpo_strategy_proposal")
    valid = dict(template["examples"][0]["output"])

    StrategyProposal.from_dict(valid, strict_llm=True)
    with pytest.raises(ValueError, match="missing fields"):
        StrategyProposal.from_dict(
            {key: value for key, value in valid.items() if key != "confidence"},
            strict_llm=True,
        )
    with pytest.raises(ValueError, match="unknown fields"):
        StrategyProposal.from_dict(
            {**valid, "explanation": "not allowed"}, strict_llm=True
        )
    with pytest.raises(ValueError, match="reason_codes"):
        StrategyProposal.from_dict(
            {**valid, "reason_codes": "not-a-list"}, strict_llm=True
        )


def test_hpo_prompt_declares_gpu_hours_as_a_hard_ceiling() -> None:
    template = load_prompt_template("hpo_strategy_proposal")
    rules = " ".join(template["rules"])

    assert "declared_limits.gpu_hours" in rules
    assert "hard ceilings" in rules
    assert "GPU-hours mean GPU count multiplied by elapsed GPU time" in rules
    assert template["decision_priority"][0].startswith("Satisfy explicit GPU-hours")


@pytest.mark.parametrize('phase,probe', [('study_planning',False),('runtime_review',False),('runtime_review',True)])
def test_stage_prompt_has_only_compatible_examples_and_contract(phase, probe):
    context = {'phase':phase, 'available_samplers':['tpe','random_search','agent_proposal'],
               'feedback':{'hypothesis_probe':{'enabled':probe,'eligible':True,'maximum_candidates':2}}}
    before = json.dumps(load_prompt_template('hpo_strategy_proposal'), sort_keys=True)
    payload = json.loads(render_prompt('hpo_strategy_proposal', context=context))
    contract = payload['response_contract']
    assert set(contract['required_keys']) == set(StrategyProposal.LLM_FIELDS)
    assert contract['phase'] == ('hypothesis_probe' if probe else phase)
    if phase == 'runtime_review':
        assert contract['locked_values']['budgets'] is None
        assert 'adjust_budget' not in contract['allowed_actions']
        assert 'The conservative baseline is 9+3+1.' not in ' '.join(payload['rules'])
    if probe:
        assert contract['locked_values']['search_space'] is None
        assert contract['locked_values']['sampler_config'] == {}
        assert contract['allowed_samplers'] == ['agent_proposal']
        assert contract['maximum_candidates'] == 2
        assert 'falsification_criterion' in contract['hypothesis_field_types']
        assert 'trial_observed' not in json.dumps(payload)
    for example in payload['examples']:
        output = example['output']
        StrategyProposal.from_dict(output, strict_llm=True)
        assert output['action'] in contract['allowed_actions']
        assert all(output[key] == value for key, value in contract['locked_values'].items())
    assert json.dumps(load_prompt_template('hpo_strategy_proposal'), sort_keys=True) == before


def test_benchmark_constraints_remove_inapplicable_sampler_examples():
    context = {'phase':'runtime_review', 'available_samplers':['tpe','agent_proposal'],
               'execution_constraints':{'allowed_samplers':['random_search'], 'agent_proposal_allowed':False,
                                        'locked_fields':['budgets','max_training_runs']}}
    payload = json.loads(render_prompt('hpo_strategy_proposal', context=context))
    assert payload['response_contract']['allowed_samplers'] == []
    assert payload['response_contract']['locked_values']['candidate_proposals'] == []
    assert all(example['output']['action'] == 'keep_strategy' for example in payload['examples'])


def test_non_hpo_prompts_distinguish_pre_execution_advice_from_results():
    for name, kind in [('data_processing_planning_advice','data_processing'),('orchestration_coordination_advisor','coordination')]:
        payload = json.loads(render_prompt(name, context={}))
        assert payload['response_contract']['kind'] == kind
        assert set(payload['response_contract']['required_keys']) == set(payload['schema'])
        assert 'before' in payload['task'] or 'pre-execution' in payload['task']
