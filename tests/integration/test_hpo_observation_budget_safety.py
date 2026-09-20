"""Raw-score, cross-Study fidelity and reserved promotion-budget regressions."""

import copy
import json

import pytest

pytest.importorskip("langgraph")

from agent.agents.hpo_agent import HPOAgent
from agent.core.metrics import InvalidMetricError
from agent.hpo import HPOScheduler, HPOService, Objective, RetryPolicy, SearchParameter, SearchSpace, StrategyProposal, TrialBudget
from agent.hpo.strategies import STRATEGIES
from agent.hpo.protocol import resolve_hpo_validation_protocol
from agent.runners import RUNNER_ADAPTERS
from agent.tools import evaluation_tools
from agent.utils import (
    MetricsCalculator,
    extract_scores_data,
    resolve_evaluation_metrics,
)
from agent.utils.experiment_tracker import ExperimentTracker
from tests.fakes import FakeRunnerAdapter


@pytest.fixture
def studies(tmp_path, minimal_config, dataset_dir):
    tracker = ExperimentTracker(tmp_path / "experiments")
    service = HPOService(tracker)
    pairs = tmp_path / "validation_pairs.txt"
    pairs.write_text(
        "1 id00001/session/a.wav id00001/session/b.wav\n"
        "0 id00001/session/a.wav id00002/session/c.wav\n",
        encoding="utf-8",
    )
    protocol = resolve_hpo_validation_protocol(
        {
            "verification_config": str(minimal_config),
            "validation_pairs": str(pairs),
            "training_exclusion_pairs": str(pairs),
        },
        require_explicit=True,
    )

    def create(**kwargs):
        dataset_version = kwargs.pop("dataset_version", "v1")
        eid = tracker.create_hpo_experiment(
            config_path=str(minimal_config), data_folder=str(dataset_dir),
            task={"type": "speaker_verification", "dataset": str(dataset_dir), "dataset_version": dataset_version},
            model={"family": "ecapa_tdnn", "implementation": "fake"},
            execution={
                "runner": "fake",
                "evaluation_config_path": protocol["verification_config"],
                "validation_pairs_path": protocol["validation_pairs"],
                "training_exclusion_pairs_path": protocol["training_exclusion_pairs"],
                **{
                    key: protocol[key]
                    for key in (
                        "verification_config_sha256",
                        "validation_pairs_sha256",
                        "training_exclusion_pairs_sha256",
                        "metric_protocol",
                    )
                },
            },
        )
        settings = {
            "search_space": SearchSpace([SearchParameter("lr", "categorical", choices=[0.1, 0.2, 0.3])]),
            "objectives": [Objective("eer", "min")], "budgets": [TrialBudget("screen", epochs=1)],
            "strategy": "grid_search", "candidate_batch_size": 1, "max_training_runs": 3,
        }
        settings.update(kwargs)
        return service.create_study(eid, **settings)

    create.protocol = protocol
    return service, create


def keep(*_):
    return StrategyProposal(action="keep_strategy")


def complete_trial(service, study):
    trial = service.suggest_trials(study, 1)[0]
    service.record_trial(study, trial.trial_id, status="running")
    return service.record_trial(study, trial.trial_id, status="completed", metrics={"eer": 0.2})


@pytest.mark.parametrize("scores", [
    [float("nan")], [float("inf")], [float("-inf")], [0.2, float("nan")],
    [True], [0.3, False], [], [[0.1]], ["0.2"], None,
])
@pytest.mark.parametrize("side", ["genuine", "impostor"])
def test_invalid_raw_scores_rejected_before_computation(scores, side):
    genuine, impostor = (scores, [0.1]) if side == "genuine" else ([0.9], scores)
    for calculate in (MetricsCalculator.calculate_eer, MetricsCalculator.calculate_min_dcf):
        with pytest.raises(InvalidMetricError):
            calculate(genuine, impostor)
    result = MetricsCalculator.compute_all_metrics(genuine, impostor)
    assert "error" in result
    assert "eer" not in result
    assert "min_dcf" not in result


def test_finite_arrays_and_true_perfect_scores_remain_valid():
    import numpy as np

    result = MetricsCalculator.compute_all_metrics(np.array([0.8, 0.9]), np.array([-0.2, 0.1]))
    assert result == {"eer": 0.0, "min_dcf": 0.0}


@pytest.mark.parametrize("text", [
    "a b 1 nan\nc d 0 nan\n", "a b 1 inf\nc d 0 -inf\n",
    "a b 1 0.8\nc d 0 0.1\ne f 1 nan\n",
    "a b 2 0.9\nc d 0 0.1\n", "broken line\n",
    "a b 1 0.8\n", "", "a b 1 not-a-number\nc d 0 0.2\n",
])
def test_invalid_score_file_fails_entire_trial_even_with_finite_runner_metrics(studies, tmp_path, monkeypatch, text):
    service, create = studies
    study = create(max_training_runs=1)
    scores = tmp_path / "scores.txt"
    scores.write_text(text, encoding="utf-8")
    runner = FakeRunnerAdapter()
    monkeypatch.setitem(RUNNER_ADAPTERS, "fake", runner)
    monkeypatch.setattr(runner, "run_evaluation", lambda *a, **k: {
        "status": "success", "metrics": {"eer": 0.3, "min_dcf": 0.2}, "scores_path": str(scores),
    })
    monkeypatch.setattr(evaluation_tools, "ExperimentTracker", lambda *_: service.tracker)
    monkeypatch.setattr(evaluation_tools, "get_experiment_artifact_dir", lambda *a, **k: tmp_path / "evaluation")

    def execute(trial, attempt):
        return json.loads(evaluation_tools.RunEvaluation.invoke({
            "experiment_id": study.experiment_id, "trial_id": trial.trial_id,
            "model_path": str(tmp_path / "fake.ckpt"),
        }))

    result = HPOScheduler(service, execute).run(study)
    assert result.study.status == "failed"
    assert result.study.best_trial_id is None
    assert result.trials[0].cost["failure_category"] == "invalid_metric"
    assert "invalid metric" in result.errors[0]
    assert not result.trials[0].metrics


@pytest.mark.parametrize("missing_file", [False, True])
def test_score_computation_error_cannot_fall_back_to_runner_metric(studies, tmp_path, monkeypatch, missing_file):
    service, create = studies
    study = create()
    trial = service.suggest_trials(study, 1)[0]
    service.record_trial(study, trial.trial_id, status="running")
    scores = tmp_path / "scores.txt"
    if not missing_file:
        scores.write_text("a b 1 0.8\nc d 0 0.1\n", encoding="utf-8")
    runner = FakeRunnerAdapter()
    monkeypatch.setitem(RUNNER_ADAPTERS, "fake", runner)
    monkeypatch.setattr(runner, "run_evaluation", lambda *a, **k: {
        "status": "success", "metrics": {"eer": 0.0}, "scores_path": str(scores),
    })
    monkeypatch.setattr(evaluation_tools, "ExperimentTracker", lambda *_: service.tracker)
    monkeypatch.setattr(evaluation_tools, "get_experiment_artifact_dir", lambda *a, **k: tmp_path / "evaluation")
    monkeypatch.setattr(MetricsCalculator, "compute_all_metrics", lambda *a: {"error": "calculation failed"})
    payload = json.loads(evaluation_tools.RunEvaluation.invoke({
        "experiment_id": study.experiment_id, "model_path": str(tmp_path / "fake.ckpt"),
        "trial_id": trial.trial_id,
    }))
    assert payload["status"] == "failed"
    assert "invalid metric" in payload["error"]


def test_valid_score_file_parses_all_pairs(tmp_path):
    path = tmp_path / "scores.txt"
    path.write_text("a b 1 0.8\n\nc d 0 0.2\n", encoding="utf-8")
    result = extract_scores_data(path)
    assert result["total_pairs"] == 2
    assert result["genuine_scores"] == [0.8]
    assert result["impostor_scores"] == [0.2]


def test_valid_scores_do_not_overwrite_complete_backend_metrics(
    tmp_path, monkeypatch
):
    path = tmp_path / "scores.txt"
    path.write_text("a b 1 0.8\nc d 0 0.2\n", encoding="utf-8")

    def unexpected_recalculation(*_):
        raise AssertionError("complete backend metrics must not be recalculated")

    monkeypatch.setattr(
        MetricsCalculator, "compute_all_metrics", unexpected_recalculation
    )
    metrics = resolve_evaluation_metrics(
        {
            "status": "success",
            "eer": 0.025,
            "min_dcf": 0.4,
            "scores_path": str(path),
        }
    )
    assert metrics == {"eer": 0.025, "min_dcf": 0.4}


def test_missing_backend_metric_uses_eer_ratio_and_scaled_min_dcf(
    tmp_path,
):
    path = tmp_path / "scores.txt"
    path.write_text(
        "a b 1 0.2\nc d 1 0.9\ne f 0 0.1\ng h 0 0.8\n",
        encoding="utf-8",
    )
    metrics = resolve_evaluation_metrics(
        {
            "status": "success",
            "eer": 0.075,
            "scores_path": str(path),
        }
    )
    assert metrics["eer"] == 0.075
    assert metrics["min_dcf"] == pytest.approx(0.5)


@pytest.mark.parametrize("change", ["epochs", "fraction", "duration", "dataset", "legacy", "malformed"])
@pytest.mark.parametrize("sampler", ["adaptive_search", "tpe"])
def test_incompatible_history_is_not_sent_to_sampler_or_dedup(studies, monkeypatch, change, sampler):
    if sampler == "tpe":
        pytest.importorskip("optuna")
    service, create = studies
    first = create(budgets=[TrialBudget("screen", epochs=1, data_fraction=0.1)])
    source = complete_trial(service, first)
    history = source.to_dict()
    budget = TrialBudget("screen", epochs=1, data_fraction=0.1)
    options = {}
    if change == "epochs":
        budget.epochs = 5
    elif change == "fraction":
        budget.data_fraction = 0.5
    elif change == "duration":
        budget.max_duration_seconds = 60
    elif change == "dataset":
        options["dataset_version"] = "v2"
    elif change == "legacy":
        history["provenance"] = {}
    else:
        history["budget"]["data_fraction"] = float("nan")
    study = create(strategy=sampler, budgets=[budget], warm_start_trials=[history], **options)
    captured = {}
    def suggest(space, count, **kwargs):
        captured.update(kwargs)
        return [dict(source.parameters)]
    monkeypatch.setattr(STRATEGIES.get(sampler), "suggest", suggest)
    trials = service.suggest_trials(study, 1)
    assert captured["history"] == []
    assert captured["existing"] == []
    assert trials[0].parameters == source.parameters
    assert len(study.warm_start_trials) == 1  # Still retained as planning evidence.
    assert study.candidate_generation_reviews[-1]["accepted_count"] == 0
    resumed = service.load_study(study.experiment_id)
    assert service.compatible_warm_start_trials(resumed) == []
    assert resumed.history_context == study.history_context


def test_compatible_history_ignores_stage_label_and_normalizes_full_data(studies):
    service, create = studies
    source_study = create(budgets=[TrialBudget("old_name", epochs=1, data_fraction=None)])
    source = complete_trial(service, source_study)
    study = create(budgets=[TrialBudget("new_name", epochs=1, data_fraction=1.0)], warm_start_trials=[source.to_dict()])
    assert [t.trial_id for t in service.compatible_warm_start_trials(study)] == [source.trial_id]
    assert service.suggest_trials(study, 1)[0].parameters != source.parameters


def test_agent_proposals_can_retest_parameters_from_incompatible_history(studies):
    service, create = studies
    source = complete_trial(service, create())
    study = create(
        strategy="agent_proposal", budgets=[TrialBudget("full", epochs=5)],
        warm_start_trials=[source.to_dict()], candidate_proposals=[{"parameters": source.parameters}],
    )
    assert study.sampler_strategy == "agent_proposal"
    assert len(study.pending_candidate_proposals) == 1
    assert service.suggest_trials(study, 1)[0].parameters == source.parameters


def test_campaign_history_keeps_same_parameters_at_distinct_budgets(studies):
    service, create = studies
    first = complete_trial(service, create())
    second = copy.deepcopy(first)
    second.trial_id = "later_trial"
    second.budget = TrialBudget("full", epochs=5)
    history = HPOAgent._merge_campaign_history([], [first, second], Objective("eer"), "none")
    assert len(history) == 2
    study = create(budgets=[second.budget], warm_start_trials=history)
    assert [t.trial_id for t in service.compatible_warm_start_trials(study)] == ["warm_later_trial"]


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("extra_budget", [0, 1])
def test_retry_preserves_future_initial_and_promotion_runs(studies, batch_size, extra_budget):
    service, create = studies
    study = create(
        pruner_strategy="successive_halving", initial_trial_count=2, promotion_limits=[1],
        budgets=[TrialBudget("screen", epochs=1), TrialBudget("confirm", epochs=5)],
        max_training_runs=3 + extra_budget, candidate_batch_size=batch_size,
    )
    calls = []
    def execute(trial, attempt):
        calls.append((trial.trial_id, attempt, trial.rung))
        if len(calls) == 1:
            return {"status": "failed", "error": "timeout"}
        return {"status": "success", "metrics": {"eer": 0.1}}
    result = HPOScheduler(service, execute, retry_policy=RetryPolicy(1), strategy_reviewer=keep).run(study)
    assert result.study.status == "completed"
    assert len(calls) == 3 + extra_budget
    assert service.training_runs_used(study) == 3 + extra_budget
    assert max(t.rung for t in result.trials) == 1
    assert result.study.scheduler_state["completion"]["confirmation_satisfied"]
    assert sum(attempt == 2 for _, attempt, _ in calls) == extra_budget
    if not extra_budget:
        failure = next(t for t in result.trials if t.status == "failed")
        assert failure.cost["retry_blocked_reason"] == "reserved_training_budget"


def test_direct_retry_and_resume_enforce_reserved_budget(studies):
    service, create = studies
    study = create(
        pruner_strategy="successive_halving", initial_trial_count=2, promotion_limits=[1],
        budgets=[TrialBudget("screen", epochs=1), TrialBudget("confirm", epochs=5)],
    )
    trial = service.suggest_trials(study, 1)[0]
    service.record_trial(study, trial.trial_id, status="failed", stop_reason="timeout")
    restored = service.load_study(study.experiment_id)
    service.prepare_resume(restored)
    assert service.reserved_training_runs(restored) == 2
    with pytest.raises(ValueError, match="reserved"):
        service.retry_trial(restored, trial.trial_id, "timeout")
    result = HPOScheduler(service, lambda *_: {"status": "success", "metrics": {"eer": 0.1}}, strategy_reviewer=keep).run(restored, resume=True)
    assert result.study.status == "completed"
    assert len(result.trials) == 3


def test_multi_rung_reservations_are_not_double_counted(studies):
    service, create = studies
    study = create(
        pruner_strategy="successive_halving", reduction_factor=2,
        initial_trial_count=2, promotion_limits=[1, 1], max_training_runs=5,
        budgets=[TrialBudget("screen", epochs=1), TrialBudget("promote", epochs=2), TrialBudget("confirm", epochs=5)],
    )
    trial = complete_trial(service, study)
    assert service.reserved_training_runs(study) == 3
    assert service.retry_budget_available(study)
    service.record_trial(study, trial.trial_id, status="failed")
    service.retry_trial(study, trial.trial_id, "timeout")
    assert service.reserved_training_runs(study) == 3
    assert not service.retry_budget_available(study)


@pytest.mark.parametrize("resume", [False, True])
def test_legacy_unconfirmed_study_cannot_report_completed(studies, resume):
    service, create = studies
    study = create(
        pruner_strategy="successive_halving", initial_trial_count=2, promotion_limits=[1],
        budgets=[TrialBudget("screen", epochs=1), TrialBudget("confirm", epochs=5)],
    )
    trial = complete_trial(service, study)
    complete_trial(service, study)
    trial.cost["retry_count"] = 1  # Legacy version consumed the confirmation slot.
    service.save_trial(study.experiment_id, trial)
    study.status = "completed"
    service._save_study(study)
    result = HPOScheduler(service, lambda *_: pytest.fail("no budget left"), strategy_reviewer=keep).run(study, resume=resume)
    assert result.study.status == "failed"
    assert any("required confirmation rung 1" in e for e in result.errors)
    assert result.study.scheduler_state["completion"] == {
        "required_rung": 1, "highest_completed_rung": 0, "confirmation_satisfied": False,
    }


def test_failed_confirmation_cannot_fall_back_to_screening_success(studies):
    service, create = studies
    study = create(
        pruner_strategy="successive_halving", initial_trial_count=2, promotion_limits=[1],
        budgets=[TrialBudget("screen", epochs=1), TrialBudget("confirm", epochs=5)],
    )
    def execute(trial, attempt):
        return {"status": "failed", "error": "timeout"} if trial.rung else {"status": "success", "metrics": {"eer": 0.1}}
    result = HPOScheduler(service, execute, strategy_reviewer=keep).run(study)
    assert result.study.status == "failed"
    assert any("required confirmation rung 1" in e for e in result.errors)
