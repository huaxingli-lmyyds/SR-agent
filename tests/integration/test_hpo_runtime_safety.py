"""Regression tests for candidate starvation and invalid objective observations."""

import json

import pytest

pytest.importorskip("langgraph")

from agent.core.metrics import InvalidMetricError, is_finite_metric
from agent.hpo import (
    AdaptiveSearchStrategy, CampaignPolicy, HPOFeedbackAnalyzer, HPOScheduler,
    HPOService, Objective, OptimizationCampaign, OptunaTPEStrategy, RetryPolicy,
    SearchParameter, SearchSpace, StrategyProposal, Trial, TrialBudget,
)
from agent.hpo.strategies import SuccessiveHalvingStrategy
from agent.hpo.protocol import resolve_hpo_validation_protocol
from agent.tasks.speaker_verification import SpeakerVerificationTaskAdapter
from agent.utils.experiment_tracker import ExperimentTracker


BAD_VALUES = [float("nan"), float("inf"), float("-inf"), True, False, None, "0.1"]


@pytest.fixture
def study_factory(tmp_path, minimal_config, dataset_dir):
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

    def create(**options):
        experiment_id = tracker.create_hpo_experiment(
            config_path=str(minimal_config), data_folder=str(dataset_dir),
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
            "strategy": "grid_search", "max_training_runs": 3,
            "candidate_batch_size": 1,
            "budgets": [TrialBudget("full", epochs=1)],
            "objectives": [Objective("eer", "min")],
            "search_space": SearchSpace([
                SearchParameter("lr", "categorical", choices=[0.1, 0.2, 0.3]),
            ]),
        }
        settings.update(options)
        return service, service.create_study(experiment_id, **settings)

    return create


def bad_review(kind):
    def review(*_):
        if kind == "exception":
            raise RuntimeError("LLM unavailable")
        if kind == "rejected_candidates":
            return StrategyProposal(
                action="keep_strategy", requested_sampler="agent_proposal",
                candidate_proposals=[{"parameters": {"lr": 99}}],
            )
        return StrategyProposal(action="invalid_proposal")
    return review


@pytest.mark.parametrize("pruner", ["none", "successive_halving"])
@pytest.mark.parametrize("kind", ["exception", "invalid", "rejected_candidates"])
def test_empty_agent_queue_falls_back_even_when_review_fails(study_factory, pruner, kind):
    options = {}
    if pruner == "successive_halving":
        options = {
            "budgets": [TrialBudget("screen", epochs=1), TrialBudget("full", epochs=3)],
            "initial_trial_count": 3, "promotion_limits": [1], "max_training_runs": 4,
        }
    service, study = study_factory(
        strategy="agent_proposal", pruner_strategy=pruner,
        fallback_sampler_strategy="grid_search",
        candidate_proposals=[{"parameters": {"lr": 0.1}}], **options,
    )
    result = HPOScheduler(
        service, lambda t, a: {"status": "success", "metrics": {"eer": t.parameters["lr"]}},
        strategy_reviewer=bad_review(kind),
    ).run(study)
    assert result.study.status == "completed"
    assert not result.errors
    assert len(result.trials) == (4 if pruner == "successive_halving" else 3)
    assert sum(t.candidate_source == "sampler:agent_proposal" for t in result.trials) == 1
    assert sum(t.candidate_source == "sampler:grid_search" for t in result.trials) == 2
    assert not result.study.pending_candidate_proposals
    assert result.study.strategy_reviews[0]["decision"]["decision"] == "rejected"
    assert "agent_queue_exhausted_fallback" in result.study.strategy_reviews[0]["decision"]["reason_codes"]


@pytest.mark.parametrize("pending", [[], [{"parameters": {"lr": 99}}], [{"parameters": {"lr": 0.1}}]])
def test_resume_with_already_reviewed_exhausted_or_stale_queue(study_factory, pending):
    service, study = study_factory(
        strategy="agent_proposal", fallback_sampler_strategy="grid_search",
        candidate_proposals=[{"parameters": {"lr": 0.1}}],
    )
    trial = service.suggest_trials(study, 1)[0]
    service.record_trial(study, trial.trial_id, status="running")
    service.record_trial(study, trial.trial_id, status="completed", metrics={"eer": 0.1})
    # Legacy checkpoint: the invalid review was committed but left no usable queue.
    study.pending_candidate_proposals = pending
    service.update_scheduler_state(study, reviewed_trial_ids=[trial.trial_id])
    result = HPOScheduler(
        service, lambda t, a: {"status": "success", "metrics": {"eer": 0.2}},
        strategy_reviewer=bad_review("invalid"),
    ).run(study, resume=True)
    assert result.study.status == "completed"
    assert len(result.trials) == 3
    assert result.study.candidate_generation_reviews[-1]["trigger"] == "agent_queue_exhausted"
    assert result.study.search_phases[-1]["sampler"] == "grid_search"


@pytest.mark.parametrize("pruner", ["none", "successive_halving"])
def test_exhausted_fallback_search_space_terminates(study_factory, pruner):
    service, study = study_factory(
        strategy="agent_proposal", fallback_sampler_strategy="grid_search",
        pruner_strategy=pruner,
        search_space=SearchSpace([SearchParameter("lr", "categorical", choices=[0.1])]),
        budgets=[TrialBudget("screen", epochs=1), TrialBudget("full", epochs=3)]
        if pruner == "successive_halving" else [TrialBudget("full", epochs=1)],
        initial_trial_count=3, promotion_limits=[1] if pruner == "successive_halving" else [],
        max_training_runs=4,
        candidate_proposals=[{"parameters": {"lr": 0.1}}],
    )
    result = HPOScheduler(
        service, lambda t, a: {"status": "success", "metrics": {"eer": 0.1}},
        strategy_reviewer=bad_review("exception"),
    ).run(study)
    assert result.study.status == "completed"
    assert sorted(t.rung for t in result.trials) == ([0, 1] if pruner == "successive_halving" else [0])


def test_valid_replenishment_keeps_agent_sampler(study_factory):
    service, study = study_factory(
        strategy="agent_proposal", candidate_proposals=[{"parameters": {"lr": 0.1}}],
    )
    def replenish(current, _feedback):
        count = len(service.list_trials(current.experiment_id))
        return StrategyProposal(
            action="keep_strategy", requested_sampler="agent_proposal",
            candidate_proposals=[{
                "parameters": {"lr": [0.1, 0.2, 0.3][count]},
                "confidence": 0.8,
            }] if count < 3 else [],
        )
    result = HPOScheduler(
        service, lambda t, a: {"status": "success", "metrics": {"eer": 0.1}},
        strategy_reviewer=replenish,
    ).run(study)
    assert result.study.status == "completed"
    assert len(result.trials) == 3
    assert all(t.candidate_source == "sampler:agent_proposal" for t in result.trials)


def test_fixed_scheduler_never_invokes_llm_reviewer(study_factory):
    service, study = study_factory(controller_mode="fixed")
    calls = []

    def reviewer(*args):
        calls.append(args)
        return StrategyProposal(
            action="switch_strategy", requested_sampler="adaptive_search",
            confidence=1.0,
        )

    result = HPOScheduler(
        service,
        lambda t, a: {"status": "success", "metrics": {"eer": 0.1}},
        strategy_reviewer=reviewer,
        review_interval_trials=1,
    ).run(study)

    assert result.study.status == "completed"
    assert calls == []
    assert all(t.candidate_source == "sampler:grid_search" for t in result.trials)


def test_stale_invalid_head_does_not_discard_valid_pending_candidates(study_factory):
    service, study = study_factory(
        strategy="agent_proposal", candidate_proposals=[{"parameters": {"lr": 0.2}}],
    )
    study.pending_candidate_proposals.insert(0, {"parameters": {"lr": 99}})
    trial = service.suggest_trials(study, 1)[0]
    assert trial.parameters == {"lr": 0.2}
    assert trial.candidate_source == "sampler:agent_proposal"
    assert (study.candidate_strategy or study.sampler_strategy) == "agent_proposal"


def test_invalid_self_referencing_fallback_uses_random(study_factory):
    service, study = study_factory(
        strategy="agent_proposal", candidate_proposals=[{"parameters": {"lr": 0.2}}],
    )
    study.pending_candidate_proposals = []
    study.search_phases[-1]["fallback_sampler"] = "agent_proposal"
    trial = service.suggest_trials(study, 1)[0]
    assert trial.candidate_source == "sampler:random_search"


@pytest.mark.parametrize("value", BAD_VALUES)
@pytest.mark.parametrize("metric", ["eer", "accuracy"])
def test_invalid_completed_metric_rejected_before_write(study_factory, value, metric):
    service, study = study_factory(objectives=[Objective(metric, "max")])
    trial = service.suggest_trials(study, 1)[0]
    service.record_trial(study, trial.trial_id, status="running", metrics={"train_loss": 0.5})
    before = service.load_trial(study.experiment_id, trial.trial_id).to_dict()
    with pytest.raises(InvalidMetricError, match="finite number"):
        service.record_trial(study, trial.trial_id, status="completed", metrics={metric: value})
    assert service.load_trial(study.experiment_id, trial.trial_id).to_dict() == before
    assert service.load_study(study.experiment_id).best_trial_id is None


@pytest.mark.parametrize("value", BAD_VALUES)
def test_scheduler_fails_invalid_trial_without_retry_and_continues(study_factory, value):
    service, study = study_factory(max_training_runs=2)
    calls = []
    def execute(trial, attempt):
        calls.append((trial.trial_id, attempt))
        return {"status": "success", "metrics": {"eer": value if len(calls) == 1 else 0.0}}
    result = HPOScheduler(
        service, execute, retry_policy=RetryPolicy(max_retries=3),
        strategy_reviewer=bad_review("invalid"),
    ).run(study)
    assert result.study.status == "completed"
    assert [attempt for _, attempt in calls] == [1, 1]
    failed = next(t for t in result.trials if t.status == "failed")
    assert failed.cost["failure_category"] == "invalid_metric"
    assert failed.cost["recoverable"] is False
    assert "eer" not in failed.metrics
    assert failed.provenance["invalid_primary_metric"]["metric"] == "eer"
    assert service.load_trial(study.experiment_id, result.study.best_trial_id).metrics["eer"] == 0.0
    assert "invalid_metric" in result.errors[0]
    json.dumps(result.study.to_dict(), allow_nan=False)
    json.dumps([t.to_dict() for t in result.trials], allow_nan=False)


@pytest.mark.parametrize("metrics", [{}, {"eer": float("nan")}, {"eer": float("inf")}])
def test_all_invalid_results_fail_study(study_factory, metrics):
    service, study = study_factory(max_training_runs=1)
    result = HPOScheduler(service, lambda *_: {"status": "success", "metrics": metrics}).run(study)
    assert result.study.status == "failed"
    assert result.study.best_trial_id is None
    assert result.trials[0].status == "failed"
    assert any("no completed trial with a valid primary metric" in e for e in result.errors)


@pytest.mark.parametrize("value", BAD_VALUES)
def test_task_adapter_and_campaign_reject_invalid_values(value):
    with pytest.raises(InvalidMetricError):
        SpeakerVerificationTaskAdapter().validate_metrics({"eer": value})
    if value is not None:  # Optional secondary metrics may be absent.
        with pytest.raises(InvalidMetricError):
            SpeakerVerificationTaskAdapter().validate_metrics({"eer": 0.1, "min_dcf": value})
    campaign = OptimizationCampaign(objective=Objective("eer", "min"))
    with pytest.raises(InvalidMetricError):
        CampaignPolicy().record_study(
            campaign, experiment_id="exp", study_id="study", best_value=value, training_runs=1,
        )
    assert campaign.study_summaries == []
    assert campaign.best_value is None


def test_legacy_invalid_best_is_not_a_completed_resume(study_factory):
    service, study = study_factory(max_training_runs=1)
    trial = service.suggest_trials(study, 1)[0]
    trial.status = "completed"
    trial.metrics = {"eer": float("nan")}
    service.save_trial(study.experiment_id, trial)
    study.status = "completed"
    study.best_trial_id = trial.trial_id
    service._save_study(study)
    assert service.completion_errors(study)
    result = HPOScheduler(service, lambda *_: pytest.fail("must not rerun legacy trial")).run(study, resume=True)
    assert result.study.status == "failed"
    assert result.study.best_trial_id is None
    best = service.tracker.get_experiment(study.experiment_id)["metrics"]["best"]
    assert best["primary_value"] is None
    assert best["trial_id"] is None


def test_completed_shortcut_rechecks_legacy_metrics(study_factory):
    service, study = study_factory(max_training_runs=1)
    trial = service.suggest_trials(study, 1)[0]
    trial.status = "completed"
    trial.metrics = {"eer": float("inf")}
    service.save_trial(study.experiment_id, trial)
    study.status = "completed"
    study.best_trial_id = trial.trial_id
    result = HPOScheduler(service, lambda *_: pytest.fail("should not run")).run(study)
    assert result.study.status == "failed"
    assert result.study.best_trial_id is None
    assert result.errors


def test_legacy_invalid_results_excluded_from_best_feedback_and_promotion(study_factory):
    service, study = study_factory(
        strategy="successive_halving", sampler_strategy="grid_search",
        search_space=SearchSpace([SearchParameter("lr", "categorical", choices=list(range(8)))]),
        budgets=[TrialBudget("screen", epochs=1), TrialBudget("full", epochs=3)],
        initial_trial_count=8, promotion_limits=[1], max_training_runs=9,
    )
    trials = service.suggest_trials(study, 8)
    for trial, value in zip(trials, [*BAD_VALUES, 0.2]):
        trial.status = "completed"
        trial.metrics = {"eer": value}
        service.save_trial(study.experiment_id, trial)
    service._refresh_study(study)
    assert study.best_trial_id == trials[-1].trial_id
    assert service.best_metric_value(study) == 0.2
    feedback = HPOFeedbackAnalyzer().analyze(study, trials)
    assert feedback["completed_trials"] == 1
    assert feedback["best_metric"] == 0.2
    json.dumps(feedback, allow_nan=False)
    assert SuccessiveHalvingStrategy().promote(trials, study.objectives[0]) == [trials[-1]]
    children = service.promote_trials(study)
    assert [t.parent_trial_id for t in children] == [trials[-1].trial_id]


def test_samplers_ignore_invalid_history(monkeypatch):
    space = SearchSpace([SearchParameter("lr", "float", low=0.0, high=1.0)])
    objective = Objective("eer", "min")
    valid = Trial("valid", {"lr": 0.5}, TrialBudget("full", epochs=1), status="completed", metrics={"eer": 0.2})
    invalid = [
        Trial(f"bad_{i}", {"lr": 0.1}, valid.budget, status="completed", metrics={"eer": v})
        for i, v in enumerate(BAD_VALUES)
    ]
    strategy = AdaptiveSearchStrategy()
    assert strategy.suggest(space, 2, history=[*invalid, valid], objective=objective) == strategy.suggest(
        space, 2, history=[valid], objective=objective,
    )
    optuna = pytest.importorskip("optuna")
    captured = []
    create_study = optuna.create_study
    def capture(**kwargs):
        study = create_study(**kwargs)
        captured.append(study)
        return study
    monkeypatch.setattr(optuna, "create_study", capture)
    OptunaTPEStrategy().suggest(space, 1, history=[*invalid, valid], objective=objective)
    observations = captured[0].get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    assert len(observations) == 1
    assert is_finite_metric(observations[0].value)


@pytest.mark.parametrize("raw", [
    {"status": "success", "metrics": {"eer": float("nan")}},
    {"status": "success", "metrics": {"eer": float("inf")}},
    {"status": "success", "metrics": {}},
    {"status": "failed", "error": "evaluation timeout"},
])
def test_evaluation_tool_rejects_bad_success_but_preserves_runner_error(
    study_factory, tmp_path, monkeypatch, raw,
):
    from agent.hpo.policies import FailurePolicy
    from agent.runners import RUNNER_ADAPTERS
    from agent.tools import evaluation_tools
    from tests.fakes import FakeRunnerAdapter

    service, study = study_factory(max_training_runs=1)
    runner = FakeRunnerAdapter()
    monkeypatch.setattr(runner, "run_evaluation", lambda *args, **kwargs: raw)
    monkeypatch.setitem(RUNNER_ADAPTERS, "fake", runner)
    monkeypatch.setattr(evaluation_tools, "ExperimentTracker", lambda *_: service.tracker)
    monkeypatch.setattr(evaluation_tools, "get_experiment_artifact_dir", lambda *a, **k: tmp_path / "evaluation")
    executed = []
    def execute(trial, attempt):
        payload = json.loads(evaluation_tools.RunEvaluation.invoke({
            "experiment_id": study.experiment_id, "trial_id": trial.trial_id,
            "model_path": str(tmp_path / "fake.ckpt"), "runner": "fake", "implementation": "fake",
            "task_type": "speaker_verification", "model_family": "ecapa_tdnn",
        }))
        executed.append(payload)
        return payload
    result = HPOScheduler(service, execute, retry_policy=RetryPolicy(max_retries=0)).run(study)
    assert result.study.status == "failed"
    assert result.study.best_trial_id is None
    assert executed[0]["status"] == "failed"
    if raw["status"] == "failed":
        assert executed[0]["error"] == "evaluation timeout"
        assert FailurePolicy().classify(executed[0]["error"]).recoverable
    else:
        assert "invalid metric" in executed[0]["error"]
        assert result.trials[0].cost["failure_category"] == "invalid_metric"
