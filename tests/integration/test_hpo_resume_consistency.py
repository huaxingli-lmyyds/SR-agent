"""Crash-boundary regressions; all experiments and training are local fakes."""

import pytest

from agent.hpo import HPOService, Objective, SearchParameter, SearchSpace, StrategyProposal, TrialBudget
from agent.utils.experiment_tracker import ExperimentTracker


class InjectedCrash(BaseException):
    """Model process termination without entering ordinary error recovery."""


@pytest.fixture
def study_factory(tmp_path, minimal_config, dataset_dir):
    tracker = ExperimentTracker(tmp_path / "experiments")
    service = HPOService(tracker)

    def create(**options):
        experiment_id = tracker.create_hpo_experiment(
            config_path=str(minimal_config), data_folder=str(dataset_dir),
        )
        settings = {
            "strategy": "grid_search", "max_training_runs": 2,
            "budgets": [TrialBudget("full", epochs=1)],
            "search_space": SearchSpace([
                SearchParameter("lr", "categorical", choices=[0.1, 0.2]),
            ]),
        }
        settings.update(options)
        study = service.create_study(experiment_id, objectives=[Objective("eer", "min")], **settings)
        return service, study

    return create


def finish_trial(service, study, trial, metric=None):
    service.record_trial(study, trial.trial_id, status="running")
    service.record_trial(
        study, trial.trial_id, status="completed",
        metrics={"eer": metric if metric is not None else trial.parameters["lr"]},
    )


@pytest.mark.parametrize("existing_best", [False, True])
def test_resume_rebuilds_best_after_result_commit_before_study_refresh(
    study_factory, monkeypatch, existing_best,
):
    service, study = study_factory()
    trials = service.suggest_trials(study, 2 if existing_best else 1)
    if existing_best:
        finish_trial(service, study, trials[0], metric=0.5)
    target = trials[-1]
    service.record_trial(study, target.trial_id, status="running")

    with monkeypatch.context() as patch:
        def crash(_study):
            raise InjectedCrash()
        patch.setattr(service, "_refresh_study", crash)
        with pytest.raises(InjectedCrash):
            service.record_trial(study, target.trial_id, status="completed", metrics={"eer": 0.01})

    saved = service.load_study(study.experiment_id)
    assert saved.best_trial_id != target.trial_id
    service.prepare_resume(saved)
    assert saved.best_trial_id == target.trial_id
    assert set(saved.trial_ids) == {trial.trial_id for trial in trials}
    assert service.unreviewed_trial_count(saved) == len(trials)
    assert service.complete_study(saved).status == "completed"
    assert service.prepare_resume(service.load_study(study.experiment_id))["resumed"] is False


@pytest.mark.parametrize("crash_boundary", ["before_child", "before_parent", "before_study"])
def test_partial_promotion_is_recovered_without_duplicate_or_lost_children(
    study_factory, monkeypatch, crash_boundary,
):
    service, study = study_factory(
        strategy="successive_halving", sampler_strategy="grid_search",
        search_space=SearchSpace([SearchParameter("lr", "categorical", choices=list(range(6)))]),
        budgets=[TrialBudget("screen", epochs=1), TrialBudget("confirm", epochs=2)],
        initial_trial_count=6, promotion_limits=[2], min_completed_per_rung=6,
        max_training_runs=8,
    )
    parents = service.suggest_trials(study, 6)
    for trial in parents:
        finish_trial(service, study, trial)
    save_trial = service.save_trial

    with monkeypatch.context() as patch:
        def crashing_save(experiment_id, trial):
            if crash_boundary == "before_child" and trial.rung == 1:
                raise InjectedCrash()
            if crash_boundary == "before_parent" and trial.status == "promoted":
                raise InjectedCrash()
            save_trial(experiment_id, trial)

        def crashing_study(_study):
            raise InjectedCrash()

        patch.setattr(service, "save_trial", crashing_save)
        if crash_boundary == "before_study":
            patch.setattr(service, "_save_study", crashing_study)
        with pytest.raises(InjectedCrash):
            service.promote_trials(study)

    persisted_children = {
        trial.trial_id for trial in service.list_trials(study.experiment_id) if trial.rung == 1
    }
    resumed = service.load_study(study.experiment_id)
    service.prepare_resume(resumed)
    service.promote_trials(resumed)
    children = [trial for trial in service.list_trials(study.experiment_id) if trial.rung == 1]
    assert len(children) == 2
    assert {trial.parameters["lr"] for trial in children} == {0, 1}
    assert persisted_children <= {trial.trial_id for trial in children}
    assert len(set(resumed.trial_ids)) == 8
    assert service.training_runs_used(resumed) == 8
    for child in children:
        finish_trial(service, resumed, child)
    assert service.promote_trials(resumed) == []
    assert service.complete_study(resumed).best_trial_id in {trial.trial_id for trial in children}
    assert service.prepare_resume(service.load_study(study.experiment_id))["resumed"] is False


def test_resume_repairs_legacy_parent_first_promotion_gap(study_factory):
    service, study = study_factory(
        strategy="successive_halving", initial_trial_count=1, promotion_limits=[1],
        budgets=[TrialBudget("screen", epochs=1), TrialBudget("confirm", epochs=2)],
    )
    parent = service.suggest_trials(study, 1)[0]
    finish_trial(service, study, parent)
    parent = service.load_trial(study.experiment_id, parent.trial_id)
    parent.status = "promoted"
    service.save_trial(study.experiment_id, parent)
    # Old versions could even declare this incomplete promotion completed.
    study.status = "completed"
    service._save_study(study)
    resumed = service.load_study(study.experiment_id)
    report = service.prepare_resume(resumed)
    assert report["resumed"] is True
    assert report["promotion_repairs"][0]["status"] == "completed"
    child = service.promote_trials(resumed)[0]
    assert child.parent_trial_id == parent.trial_id
    assert service.promote_trials(resumed) == []


def test_resume_can_itself_be_interrupted_during_legacy_repair(study_factory, monkeypatch):
    service, study = study_factory(
        strategy="successive_halving", initial_trial_count=1, promotion_limits=[1],
        budgets=[TrialBudget("screen", epochs=1), TrialBudget("confirm", epochs=2)],
    )
    parent = service.suggest_trials(study, 1)[0]
    finish_trial(service, study, parent)
    parent = service.load_trial(study.experiment_id, parent.trial_id)
    parent.status = "promoted"
    service.save_trial(study.experiment_id, parent)
    study.status = "completed"
    service._save_study(study)
    save_trial = service.save_trial
    with monkeypatch.context() as patch:
        def crash_after_repair(experiment_id, trial):
            save_trial(experiment_id, trial)
            raise InjectedCrash()
        patch.setattr(service, "save_trial", crash_after_repair)
        with pytest.raises(InjectedCrash):
            service.prepare_resume(service.load_study(study.experiment_id))

    resumed = service.load_study(study.experiment_id)
    assert service.prepare_resume(resumed)["resumed"] is True
    assert len(service.promote_trials(resumed)) == 1


def test_multirung_promotion_without_explicit_limits_does_not_repromote_losers(study_factory):
    pytest.importorskip("langgraph")
    from agent.hpo import HPOScheduler

    service, study = study_factory(
        strategy="successive_halving", sampler_strategy="grid_search",
        search_space=SearchSpace([SearchParameter("lr", "categorical", choices=list(range(6)))]),
        budgets=[TrialBudget("screen", epochs=1), TrialBudget("confirm", epochs=2),
                 TrialBudget("full", epochs=3)],
        initial_trial_count=6, max_training_runs=12,
    )
    result = HPOScheduler(service, lambda trial, attempt: {
        "status": "success", "metrics": {"eer": trial.parameters["lr"]},
    }).run(study)
    assert result.study.status == "completed"
    assert [sum(trial.rung == rung for trial in result.trials) for rung in range(3)] == [6, 2, 1]
    assert service.load_trial(study.experiment_id, result.study.best_trial_id).rung == 2


@pytest.mark.parametrize("saved_counter", [0, 1])
def test_resume_reviews_pending_batch_before_generating_more_candidates(study_factory, saved_counter):
    pytest.importorskip("langgraph")
    from agent.hpo import HPOScheduler

    service, study = study_factory(
        strategy="agent_proposal", candidate_batch_size=1,
        candidate_proposals=[{"parameters": {"lr": 0.1}}],
    )
    first = service.suggest_trials(study, 1)[0]
    finish_trial(service, study, first)
    service.update_scheduler_state(study, completed_since_review=saved_counter)
    events = []

    def review(current_study, feedback):
        events.append("review")
        return StrategyProposal(
            action="keep_strategy", candidate_proposals=[{"parameters": {"lr": 0.2}}],
        )

    def execute(trial, attempt):
        events.append("execute")
        assert trial.parameters == {"lr": 0.2}
        return {"status": "success", "metrics": {"eer": 0.02}}

    result = HPOScheduler(service, execute, strategy_reviewer=review).run(study, resume=True)
    assert events[:2] == ["review", "execute"]
    assert events.count("execute") == 1
    assert result.study.status == "completed"
    assert len(result.trials) == 2
    assert service.unreviewed_trial_count(result.study) == 0


def test_resume_does_not_replay_committed_review_or_overwrite_it_with_stale_study(study_factory):
    pytest.importorskip("langgraph")
    from agent.hpo import HPOScheduler

    service, study = study_factory(
        strategy="agent_proposal", candidate_batch_size=1,
        candidate_proposals=[{"parameters": {"lr": 0.1}}],
    )
    finish_trial(service, study, service.suggest_trials(study, 1)[0])
    stale = service.load_study(study.experiment_id)
    service.review_strategy(study, StrategyProposal(
        action="keep_strategy", candidate_proposals=[{"parameters": {"lr": 0.2}}],
    ))
    events = []

    def review(current_study, feedback):
        events.append("review")
        return StrategyProposal(action="keep_strategy")

    def execute(trial, attempt):
        events.append("execute")
        assert trial.parameters == {"lr": 0.2}
        return {"status": "success", "metrics": {"eer": 0.02}}

    scheduler = HPOScheduler(service, execute, strategy_reviewer=review)
    result = scheduler.run(stale, resume=True)
    assert events == ["execute", "review"]
    assert len(result.study.strategy_reviews) == 2
    assert len(result.trials) == 2
    scheduler.run(stale, resume=True)
    assert events == ["execute", "review"]


def test_final_review_is_deferred_when_current_grid_is_exhausted(study_factory):
    pytest.importorskip("langgraph")
    from agent.hpo import HPOScheduler

    service, study = study_factory(max_training_runs=3)
    executed = []

    def review(current_study, feedback):
        if feedback["review_trigger"] == "final_trials":
            return StrategyProposal(
                action="switch_strategy", requested_sampler="agent_proposal",
                search_space=SearchSpace([
                    SearchParameter("lr", "categorical", choices=[0.1, 0.2, 0.3]),
                ]).to_dict(),
                candidate_proposals=[{"parameters": {"lr": 0.3}}],
            )
        return StrategyProposal(action="keep_strategy")

    result = HPOScheduler(
        service,
        lambda trial, attempt: (
            executed.append(trial.parameters["lr"])
            or {"status": "success", "metrics": {"eer": 0.02}}
        ),
        strategy_reviewer=review, review_interval_trials=3,
    ).run(study)
    assert result.study.status == "completed"
    assert sorted(executed) == [0.1, 0.2]
    assert result.study.pending_candidate_proposals == []
    assert result.study.next_study_proposal is not None
    final_review = result.study.strategy_reviews[-1]
    assert final_review["scope"] == "next_study"
    assert final_review["applied"] is False
    assert final_review["affected_trial_ids"] == []


def test_coordinator_resume_reads_snapshot_before_mutable_source(study_factory, monkeypatch):
    pytest.importorskip("langgraph")
    import agent.agents.orchestrator as orchestrator

    service, study = study_factory()
    snapshot = service.tracker.get_config_snapshot(study.experiment_id)
    coordinator = object.__new__(orchestrator.CoordinatorAgent)
    coordinator.config_path = "missing-original-config.yaml"
    monkeypatch.setattr(orchestrator, "ExperimentTracker", lambda *_: service.tracker)
    monkeypatch.setattr(orchestrator, "get_hpo_experiments_dir", lambda: snapshot.parent)

    def capture_parser(config_path):
        assert config_path == str(snapshot)
        raise InjectedCrash()

    monkeypatch.setattr(orchestrator, "ConfigParser", capture_parser)
    with pytest.raises(InjectedCrash):
        coordinator.run(context={"resume_experiment_id": study.experiment_id})


def test_agent_rejects_missing_snapshot_before_changing_trial_or_study(study_factory):
    pytest.importorskip("langgraph")
    from datetime import datetime

    from agent.agents.communication import AgentTaskRequest
    from agent.agents.hpo_agent import HPOAgent

    service, study = study_factory()
    trial = service.suggest_trials(study, 1)[0]
    service.record_trial(study, trial.trial_id, status="running")
    before_study = service.load_study(study.experiment_id).to_dict()
    before_trial = service.load_trial(study.experiment_id, trial.trial_id).to_dict()
    service.tracker.get_config_snapshot(study.experiment_id).unlink()
    agent = object.__new__(HPOAgent)
    with pytest.raises(FileNotFoundError, match="config snapshot missing"):
        agent._resume_existing_study(
            AgentTaskRequest(action="optimize_hyperparameters", objective="resume"),
            service.tracker, study.experiment_id, "", {}, datetime.now(),
        )
    assert service.load_study(study.experiment_id).to_dict() == before_study
    assert service.load_trial(study.experiment_id, trial.trial_id).to_dict() == before_trial
