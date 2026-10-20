"""Prevent shared execution faults from consuming an entire search budget."""

import pytest

from agent.hpo import (
    FailurePolicy, HPOScheduler, HPOService, Objective, RetryPolicy,
    SearchParameter, SearchSpace, TrialBudget,
)
from agent.utils.experiment_tracker import ExperimentTracker


@pytest.fixture
def study_factory(tmp_path, minimal_config, dataset_dir):
    service = HPOService(ExperimentTracker(tmp_path / "experiments"))

    def create(**overrides):
        experiment_id = service.tracker.create_hpo_experiment(
            config_path=str(minimal_config), data_folder=str(dataset_dir),
        )
        options = {
            "strategy": "grid_search",
            "max_training_runs": 4,
            "candidate_batch_size": 3,
            "budgets": [TrialBudget("full", epochs=1)],
        }
        options.update(overrides)
        study = service.create_study(
            experiment_id,
            SearchSpace([SearchParameter("lr", "categorical", choices=[0.1, 0.2, 0.3, 0.4])]),
            [Objective("eer", "min")],
            controller_mode="fixed",
            **options,
        )
        return service, study

    return create


@pytest.mark.parametrize("kind", ["gpu", "wall"])
def test_resource_ceiling_blocks_retry_and_following_trials(study_factory, tmp_path, kind):
    from agent.hpo.resources import CampaignResourceBudget
    service, study = study_factory()
    now = [1000.0]
    limits = {"max_gpu_hours": 3 / 3600} if kind == "gpu" else {"max_wall_seconds": 1}
    budget = CampaignResourceBudget(tmp_path / "resources.json", limits=limits, clock=lambda: now[0])
    executed = []

    def execute(trial, attempt):
        executed.append(trial.trial_id)
        budget.start_stage(experiment_id=study.experiment_id, trial_id=trial.trial_id,
                           attempt=attempt, stage="training", gpu_count=3)
        now[0] += 1
        budget.finish_stage()
        return {"status": "failed", "stage": "training", "error": "TimeoutError: resource stage deadline"}

    try:
        result = HPOScheduler(service, execute, retry_policy=RetryPolicy(5), resource_budget=budget).run(study)
        assert len(executed) == 1
        assert result.study.status == "cancelled"
        assert "resource_budget_exhausted" in result.study.stop_reason
        assert result.study.scheduler_state["resource_budget"]["gpu_seconds"] == 3
        assert any(t.status == "suggested" for t in result.trials)
        assert sum(t.status == "failed" for t in result.trials) == 1
        assert result.study.scheduler_state["failure_guard"]["status"] != "open"
        # Resume cannot reset the budget or start any further training.
        HPOScheduler(service, execute, resource_budget=budget).run(result.study, resume=True)
        assert len(executed) == 1
    finally:
        budget.close()


@pytest.mark.parametrize("error", [
    "AttributeError: Can't get local object 'dataio_prep.<locals>.audio_pipeline'",
    "ModuleNotFoundError: No module named 'speechbrain'",
    "builtins.FileNotFoundError: [Errno 2] No such file: '/data/wav/a.wav'",
    "PermissionError: [Errno 13] Permission denied: '/data/wav/a.wav'",
    "CUDA error: invalid device ordinal",
])
def test_shared_errors_abort_before_retry_review_or_promotion(study_factory, error):
    service, study = study_factory(
        strategy="successive_halving", sampler_strategy="grid_search",
        budgets=[TrialBudget("screen", epochs=1), TrialBudget("full", epochs=2)],
        initial_trial_count=3, promotion_limits=[1],
    )
    calls = []
    reviews = []

    def execute(trial, attempt):
        calls.append((trial.trial_id, attempt))
        return {"status": "failed", "stage": "evaluation", "error": error}

    result = HPOScheduler(
        service, execute, retry_policy=RetryPolicy(5),
        strategy_reviewer=lambda *args: reviews.append(args),
    ).run(study)

    assert len(calls) == 1
    assert reviews == []
    assert result.study.status == "failed"
    assert "system_failure:" in result.study.stop_reason
    assert len([trial for trial in result.trials if trial.status == "suggested"]) == 2
    assert all(trial.rung == 0 for trial in result.trials)
    failed = next(trial for trial in result.trials if trial.status == "failed")
    assert failed.cost["failure_scope"] == "study"
    assert failed.cost["failure_stage"] == "evaluation"
    record = service.tracker.get_experiment(study.experiment_id)
    assert record["status"] == "failed"
    assert record["extensions"]["optimization"]["study"]["failure_guard"]["status"] == "open"


def test_unknown_repeated_failure_matches_different_trial_paths(study_factory):
    service, study = study_factory()
    calls = []

    def execute(trial, attempt):
        calls.append(trial.trial_id)
        return {
            "status": "failed", "stage": "evaluation",
            "error": f"RuntimeError: native scorer crashed at /tmp/{trial.trial_id}/log.txt",
        }

    result = HPOScheduler(service, execute, retry_policy=RetryPolicy(0)).run(study)

    assert len(calls) == 2
    guard = result.study.scheduler_state["failure_guard"]
    assert guard["reason_code"] == "repeated_execution_error"
    assert guard["trial_ids"] == calls
    assert len({trial.parameters["lr"] for trial in result.trials if trial.status == "failed"}) == 2
    assert any(trial.status == "suggested" for trial in result.trials)


@pytest.mark.parametrize("error", [
    "torch.OutOfMemoryError: CUDA out of memory",
    "TimeoutError: training exceeded 10 seconds",
    "InvalidMetricError: invalid metric 'eer': NaN",
    "ValueError: invalid margin for this candidate",
])
def test_candidate_failures_do_not_block_the_search(study_factory, error):
    service, study = study_factory()
    calls = []

    def execute(trial, attempt):
        calls.append(trial.trial_id)
        if len(calls) < 4:
            return {"status": "failed", "error": error}
        return {"status": "success", "metrics": {"eer": 0.1}}

    result = HPOScheduler(service, execute, retry_policy=RetryPolicy(0)).run(study)

    assert len(calls) == 4
    assert result.study.status == "completed"
    assert result.study.scheduler_state["failure_guard"]["status"] == "closed"


def test_success_resets_unknown_failure_streak(study_factory):
    service, study = study_factory()
    calls = []

    def execute(trial, attempt):
        calls.append(trial.trial_id)
        if len(calls) % 2:
            return {"status": "failed", "error": "RuntimeError: native scorer crashed"}
        return {"status": "success", "metrics": {"eer": 0.1}}

    result = HPOScheduler(service, execute).run(study)
    assert len(calls) == 4
    assert result.study.status == "completed"


def test_same_error_in_different_stages_does_not_form_shared_streak(study_factory):
    service, study = study_factory(max_training_runs=3)
    calls = []

    def execute(trial, attempt):
        calls.append(trial.trial_id)
        if len(calls) < 3:
            return {
                "status": "failed", "stage": "training" if len(calls) == 1 else "evaluation",
                "error": "RuntimeError: native code crashed",
            }
        return {"status": "success", "metrics": {"eer": 0.1}}

    result = HPOScheduler(service, execute).run(study)
    assert len(calls) == 3
    assert result.study.status == "completed"


def test_stop_preserves_trained_checkpoint_and_previous_success(study_factory, tmp_path):
    service, study = study_factory()
    checkpoint = tmp_path / "embedding_model.ckpt"
    checkpoint.write_text("retained weights", encoding="utf-8")
    calls = []

    def execute(trial, attempt):
        calls.append(trial.trial_id)
        if len(calls) == 1:
            return {"status": "success", "metrics": {"eer": 0.1}}
        service.record_trial(
            study, trial.trial_id, status="running",
            cost={"training": {"status": "success", "duration_seconds": 100}},
            artifacts=[{"type": "checkpoint", "path": str(checkpoint)}],
        )
        return {"status": "failed", "stage": "evaluation", "error": "AttributeError: bad pipeline"}

    result = HPOScheduler(service, execute).run(study)
    assert len(calls) == 2
    assert result.study.status == "failed"
    assert result.study.best_trial_id == calls[0]
    failed = service.load_trial(study.experiment_id, calls[1])
    assert failed.artifacts[0]["path"] == str(checkpoint)
    assert checkpoint.read_text(encoding="utf-8") == "retained weights"
    assert failed.cost["training"]["status"] == "success"


def test_resume_stays_blocked_until_explicit_reset(study_factory):
    service, study = study_factory()
    calls = []

    def broken(trial, attempt):
        calls.append(trial.trial_id)
        return {"status": "failed", "error": "AttributeError: bad pipeline"}

    first = HPOScheduler(service, broken).run(study)
    second = HPOScheduler(service, broken).run(first.study, resume=True)
    assert len(calls) == 1
    assert second.study.status == "failed"

    def repaired(trial, attempt):
        calls.append(trial.trial_id)
        return {"status": "success", "metrics": {"eer": 0.1}}

    resumed = HPOScheduler(service, repaired).run(
        second.study, resume=True, reset_failure_guard=True,
    )
    assert resumed.study.status == "completed"
    assert len(calls) == 4
    guard = resumed.study.scheduler_state["failure_guard"]
    assert guard["status"] == "closed"
    assert guard["reset_count"] == 1
    assert guard["reset_events"][0]["previous_reason_code"] == "system_error"
    HPOScheduler(service, repaired).run(resumed.study, resume=True)
    assert len(calls) == 4


def test_resume_recovers_system_failure_committed_before_guard(study_factory, monkeypatch):
    service, study = study_factory()
    update = service.update_scheduler_state

    class InjectedCrash(BaseException):
        pass

    def crashing_update(current, **updates):
        if (updates.get("failure_guard") or {}).get("status") == "open":
            raise InjectedCrash()
        return update(current, **updates)

    with monkeypatch.context() as patch:
        patch.setattr(service, "update_scheduler_state", crashing_update)
        with pytest.raises(InjectedCrash):
            HPOScheduler(service, lambda *_: {
                "status": "failed", "stage": "evaluation", "error": "AttributeError: bad pipeline",
            }).run(study)

    calls = []
    result = HPOScheduler(service, lambda *_: calls.append("unexpected")).run(study, resume=True)
    assert calls == []
    assert result.study.status == "failed"
    assert result.study.scheduler_state["failure_guard"]["failure_stage"] == "evaluation"


def test_legacy_completed_study_is_not_invalidated_by_old_failure(study_factory):
    service, study = study_factory(max_training_runs=2)
    trials = service.suggest_trials(study, 2)
    service.record_trial(study, trials[0].trial_id, status="running")
    service.record_trial(study, trials[0].trial_id, status="completed", metrics={"eer": 0.1})
    service.record_trial(study, trials[1].trial_id, status="running")
    service.record_trial(study, trials[1].trial_id, status="failed", stop_reason="AttributeError: old fixed error")
    service.complete_study(study)
    calls = []
    result = HPOScheduler(service, lambda *_: calls.append("unexpected")).run(study, resume=True)
    assert result.study.status == "completed"
    assert calls == []


def test_legacy_failure_before_later_success_does_not_block_resume(study_factory):
    service, study = study_factory()
    trials = service.suggest_trials(study, 3)
    service.record_trial(study, trials[0].trial_id, status="running")
    service.record_trial(study, trials[0].trial_id, status="failed", stop_reason="AttributeError: old fixed error")
    service.record_trial(study, trials[1].trial_id, status="running")
    service.record_trial(study, trials[1].trial_id, status="completed", metrics={"eer": 0.1})
    result = HPOScheduler(service, lambda *_: {
        "status": "success", "metrics": {"eer": 0.2},
    }).run(study, resume=True)
    assert result.study.status == "completed"
    assert result.study.scheduler_state["failure_guard"]["status"] == "closed"


def test_classifier_ignores_incidental_traceback_source_text():
    error = (
        "RuntimeError: CUDA out of memory\nTraceback (most recent call last):\n"
        '  File "/configs/missing/config.py", line 10\n'
        '    raise RuntimeError("CUDA out of memory")\nRuntimeError: CUDA out of memory'
    )
    decision = FailurePolicy().classify(error)
    assert decision.category == "resource"
    assert decision.scope == "trial"
    assert decision.recoverable


@pytest.mark.parametrize("error", [
    "OSError: [Errno 28] No space left on device",
    "RuntimeError: Cannot re-initialize CUDA in forked subprocess",
    "TypeError: compute_embedding() got an unexpected keyword argument 'device'",
    "missing checkpoint artifact from current training attempt",
    "ValueError: numpy.dtype size changed, may indicate binary incompatibility",
])
def test_additional_shared_environment_and_contract_errors_stop_immediately(study_factory, error):
    service, study = study_factory()
    calls = []

    def execute(trial, attempt):
        calls.append(attempt)
        return {"status": "failed", "stage": "training", "error": error}

    result = HPOScheduler(service, execute, retry_policy=RetryPolicy(10)).run(study)
    assert calls == [1]
    assert result.study.scheduler_state["failure_guard"]["status"] == "open"


def test_scheduler_node_failure_is_persisted_and_blocks_resume(study_factory, monkeypatch):
    service, study = study_factory()
    calls = []
    scheduler = HPOScheduler(service, lambda *_: calls.append("unexpected"))

    def broken_suggest(*args):
        raise KeyError("broken sampler state")

    with monkeypatch.context() as patch:
        patch.setattr(service, "suggest_trials", broken_suggest)
        result = scheduler.run(study)
    assert calls == []
    guard = result.study.scheduler_state["failure_guard"]
    assert guard["failure_stage"] == "scheduler"
    assert guard["failure_node"] == "suggest"
    assert "broken_suggest" in guard["traceback"]
    assert guard["experiment_id"] == study.experiment_id
    assert "KeyError" in guard["error_summary"]
    resumed = HPOScheduler(service, lambda *_: calls.append("unexpected")).run(study, resume=True)
    assert resumed.study.status == "failed"
    assert calls == []


def test_persistent_transient_retries_are_bounded_and_recover_guard_after_crash(study_factory, monkeypatch):
    service, study = study_factory(max_training_runs=10, initial_trial_count=1, candidate_batch_size=1)
    update = service.update_scheduler_state
    calls = []

    class InjectedCrash(BaseException):
        pass

    def crashing_update(current, **updates):
        if (updates.get("failure_guard") or {}).get("status") == "open":
            raise InjectedCrash()
        return update(current, **updates)

    def execute(trial, attempt):
        calls.append(attempt)
        return {"status": "failed", "stage": "evaluation", "error": f"RuntimeError: connection refused by worker {attempt}"}

    with monkeypatch.context() as patch:
        patch.setattr(service, "update_scheduler_state", crashing_update)
        with pytest.raises(InjectedCrash):
            HPOScheduler(service, execute, retry_policy=RetryPolicy(9)).run(study)
    assert calls == [1, 2, 3], service.load_study(study.experiment_id).stop_reason
    result = HPOScheduler(service, execute).run(study, resume=True)
    assert calls == [1, 2, 3]
    guard = result.study.scheduler_state["failure_guard"]
    assert guard["reason_code"] == "repeated_transient_error"
    assert guard["observed_failure_count"] == 3


def test_wrapper_and_torchrun_rank_prefix_do_not_hide_underlying_system_error():
    error = (
        "[rank1]: ModuleNotFoundError: No module named 'required_backend'\n"
        "The above exception was the direct cause of the following exception:\n"
        "RuntimeError: worker setup failed\n"
        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: workers failed"
    )
    decision = FailurePolicy().classify(error)
    assert decision.scope == "study"
    assert FailurePolicy.root_error(error).startswith("ModuleNotFoundError:")


def test_non_oom_cuda_failure_cannot_repeat_through_entire_search(study_factory):
    service, study = study_factory()
    calls = []

    def execute(trial, attempt):
        calls.append(trial.trial_id)
        return {"status": "failed", "stage": "training", "error": "RuntimeError: CUDA error: unspecified launch failure"}

    result = HPOScheduler(service, execute, retry_policy=RetryPolicy(0)).run(study)
    assert len(calls) == 2
    assert result.study.scheduler_state["failure_guard"]["reason_code"] == "repeated_transient_error"


def test_worker_wrapper_does_not_turn_oom_into_shared_connection_failure():
    decision = FailurePolicy().classify(
        "torch.OutOfMemoryError: CUDA out of memory\n"
        "The above exception was the direct cause of the following exception:\n"
        "RuntimeError: worker process failed"
    )
    assert decision.category == "resource"
    assert decision.scope == "trial"


def test_campaign_failure_blocks_later_studies_and_older_study_resume(
    tmp_path, minimal_config, dataset_dir, monkeypatch,
):
    from agent.agents.communication import AgentTaskRequest
    from agent.agents.hpo_agent import HPOAgent
    from agent.memory import MemoryService
    from agent.runners import RUNNER_ADAPTERS
    from tests.fakes import FakeRunnerAdapter

    runner = FakeRunnerAdapter()
    monkeypatch.setitem(RUNNER_ADAPTERS, "fake", runner)
    pairs = tmp_path / "validation.txt"
    pairs.write_text("1 val1/a.wav val1/b.wav\n0 val1/a.wav val2/a.wav\n", encoding="utf-8")
    calls = []
    original_train, original_evaluate = runner.run_training, runner.run_evaluation

    def train(*args, **kwargs):
        calls.append("train")
        return original_train(*args, **kwargs)

    def evaluate(*args, **kwargs):
        if len(calls) == 2:
            raise AttributeError("shared evaluation pipeline failed")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, "run_training", train)
    monkeypatch.setattr(runner, "run_evaluation", evaluate)
    agent = HPOAgent(config_path=str(minimal_config), runner="fake", verbose=False,
                     experiments_dir=str(tmp_path / "experiments"), memory_service=MemoryService(tmp_path / "memory"))
    result = agent.execute_task(AgentTaskRequest(
        action="optimize_hyperparameters", objective="campaign guard",
        context={
            "data_folder": str(dataset_dir), "sampler": "grid_search", "pruner": "none", "controller_mode": "fixed",
            "search_space": {"parameters": [{"name": "lr", "parameter_type": "categorical", "choices": [0.001, 0.002, 0.003]}]},
            "budgets": [{"stage": "full", "epochs": 1, "data_fraction": 1.0}],
            "runtime_options": {"verification_config": str(minimal_config), "validation_pairs": str(pairs), "training_exclusion_pairs": str(pairs)},
        }, budget={"max_training_runs": 1, "max_total_training_runs": 3, "max_studies": 3, "campaign_patience": 3},
    ))
    assert result.status == "failed", result.error
    assert calls == ["train", "train"]
    assert result.summary["campaign"]["status"] == "failed"
    assert result.summary["failure_guard"]["status"] == "open"
    ids = result.experiment_ids["campaign"]
    assert len(ids) == 2
    older = agent.execute_task(AgentTaskRequest(
        action="optimize_hyperparameters", objective="resume older Study",
        context={"resume_experiment_id": ids[0]},
    ))
    assert older.status == "failed"
    assert "Campaign blocked" in older.error
    assert calls == ["train", "train"]
