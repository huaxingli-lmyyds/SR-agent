"""LangGraph-backed deterministic HPO workflow."""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .contracts import HPOStudy, StrategyProposal, Trial
from .feedback import HPOFeedbackAnalyzer
from .policies import FailurePolicy, RetryPolicy
from .service import HPOService


TrialExecutor = Callable[[Trial, int], Dict[str, Any]]
StrategyAdvisor = Callable[[HPOStudy], Dict[str, Any]]
StrategyReviewer = Callable[[HPOStudy, Dict[str, Any]], Optional[StrategyProposal]]


@dataclass
class SchedulerResult:
    study: HPOStudy
    trials: List[Trial]
    errors: List[str]
    advice: Dict[str, Any]
    strategy_reviews: List[Dict[str, Any]]


class HPOGraphState(TypedDict, total=False):
    experiment_id: str
    current_trial_id: Optional[str]
    attempt: int
    last_result: Dict[str, Any]
    last_error: Optional[str]
    errors: List[str]
    advice: Dict[str, Any]
    route: str
    completed_since_review: int


class DecisionPolicy:
    """Deterministic branch decisions; HPOService still enforces all rules."""

    def initial_count(self, study: HPOStudy) -> int:
        if study.strategy == "adaptive_search":
            return 1
        return study.initial_trial_count or study.max_trials or 1

    def next_trial(self, study: HPOStudy, service: HPOService) -> Optional[Trial]:
        return next(
            (
                trial
                for trial in service.list_trials(study.experiment_id)
                if trial.status == "suggested"
            ),
            None,
        )

    def should_promote(self, study: HPOStudy, service: HPOService) -> bool:
        if study.strategy != "successive_halving":
            return False
        trials = service.list_trials(study.experiment_id)
        return any(
            trial.status == "completed" and trial.rung + 1 < len(study.budgets)
            for trial in trials
        ) and service.remaining_training_runs(study) > 0

    def should_suggest(self, study: HPOStudy, service: HPOService) -> bool:
        return (
            study.strategy != "successive_halving"
            and service.remaining_training_runs(study) > 0
        )


class HPOScheduler:
    """Execute HPO through a compiled LangGraph state machine."""

    def __init__(
        self,
        service: HPOService,
        executor: TrialExecutor,
        *,
        decision_policy: Optional[DecisionPolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
        retry_policy: Optional[RetryPolicy] = None,
        strategy_advisor: Optional[StrategyAdvisor] = None,
        strategy_reviewer: Optional[StrategyReviewer] = None,
        review_interval_trials: int = 3,
    ) -> None:
        self.service = service
        self.executor = executor
        self.decision_policy = decision_policy or DecisionPolicy()
        self.failure_policy = failure_policy or FailurePolicy()
        self.retry_policy = retry_policy or RetryPolicy()
        self.strategy_advisor = strategy_advisor
        self.strategy_reviewer = strategy_reviewer
        self.review_interval_trials = max(int(review_interval_trials), 1)
        self._study: Optional[HPOStudy] = None
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(HPOGraphState)
        graph.add_node("advise", self._advise)
        graph.add_node("suggest", self._suggest)
        graph.add_node("select_trial", self._select_trial)
        graph.add_node("run_trial", self._run_trial)
        graph.add_node("record_result", self._record_result)
        graph.add_node("review_strategy", self._review_strategy)
        graph.add_node("promote", self._promote)
        graph.add_node("complete", self._complete)

        graph.add_edge(START, "advise")
        graph.add_edge("advise", "suggest")
        graph.add_conditional_edges(
            "suggest",
            lambda state: state["route"],
            {"next": "select_trial", "complete": "complete"},
        )
        graph.add_conditional_edges(
            "select_trial",
            lambda state: state["route"],
            {"run": "run_trial", "suggest": "suggest", "promote": "promote", "complete": "complete"},
        )
        graph.add_edge("run_trial", "record_result")
        graph.add_conditional_edges(
            "record_result",
            lambda state: state["route"],
            {"retry": "run_trial", "review": "review_strategy"},
        )
        graph.add_edge("review_strategy", "select_trial")
        graph.add_conditional_edges(
            "promote",
            lambda state: state["route"],
            {"next": "select_trial", "complete": "complete"},
        )
        graph.add_edge("complete", END)
        return graph.compile()

    @property
    def study(self) -> HPOStudy:
        if self._study is None:
            raise RuntimeError("HPO graph has no active study")
        return self._study

    def run(self, study: HPOStudy) -> SchedulerResult:
        self._study = study
        state = self.graph.invoke(
            {
                "experiment_id": study.experiment_id,
                "attempt": 1,
                "errors": [],
                "advice": {},
                "completed_since_review": 0,
            },
            config={"recursion_limit": max(50, int(study.max_training_runs or 1) * 8)},
        )
        current = self.service.load_study(study.experiment_id)
        return SchedulerResult(
            study=current,
            trials=self.service.list_trials(study.experiment_id),
            errors=state.get("errors", []),
            advice=state.get("advice", {}),
            strategy_reviews=current.strategy_reviews,
        )

    def _advise(self, state: HPOGraphState) -> Dict[str, Any]:
        try:
            advice = self.strategy_advisor(self.study) if self.strategy_advisor else {}
        except Exception as exc:
            advice = {"advice_error": f"{type(exc).__name__}: {exc}"}
        return {"advice": dict(advice or {})}

    def _suggest(self, state: HPOGraphState) -> Dict[str, Any]:
        count = self.decision_policy.initial_count(self.study)
        if self.study.strategy != "successive_halving":
            count = min(count, self.review_interval_trials)
        created = self.service.suggest_trials(
            self.study,
            count,
        )
        return {"route": "next" if created else "complete"}

    def _select_trial(self, state: HPOGraphState) -> Dict[str, Any]:
        trial = self.decision_policy.next_trial(self.study, self.service)
        if trial is not None:
            return {"current_trial_id": trial.trial_id, "attempt": 1, "route": "run"}
        if self.decision_policy.should_suggest(self.study, self.service):
            return {"route": "suggest"}
        if self.decision_policy.should_promote(self.study, self.service):
            return {"route": "promote"}
        return {"route": "complete"}

    def _run_trial(self, state: HPOGraphState) -> Dict[str, Any]:
        trial_id = str(state["current_trial_id"])
        trial = self.service.load_trial(self.study.experiment_id, trial_id)
        if trial.status == "suggested":
            self.service.record_trial(self.study, trial_id, status="running")
        try:
            result = dict(self.executor(trial, int(state.get("attempt", 1))) or {})
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        return {"last_result": result, "last_error": result.get("error")}

    def _record_result(self, state: HPOGraphState) -> Dict[str, Any]:
        trial_id = str(state["current_trial_id"])
        attempt = int(state.get("attempt", 1))
        result = state.get("last_result") or {}
        if result.get("status") == "success":
            self.service.record_trial(
                self.study,
                trial_id,
                status="completed",
                metrics=result.get("metrics") or {},
                intermediate_metrics=result.get("intermediate_metrics") or [],
                cost={**(result.get("cost") or {}), "attempts": attempt},
                artifacts=result.get("artifacts") or [],
            )
            completed = int(state.get("completed_since_review", 0)) + 1
            return {"route": "review", "last_error": None, "completed_since_review": completed}

        error = str(result.get("error") or "trial execution failed")
        failure = self.failure_policy.classify(error)
        can_retry = (
            self.retry_policy.should_retry(attempt, failure)
            and self.service.remaining_training_runs(self.study) > 0
        )
        if can_retry:
            current = self.service.load_trial(self.study.experiment_id, trial_id)
            if current.status != "failed":
                self.service.record_trial(
                    self.study,
                    trial_id,
                    status="failed",
                    stop_reason=error,
                )
            self.service.retry_trial(self.study, trial_id, error)
            self.service.record_trial(self.study, trial_id, status="running")
            if self.retry_policy.retry_delay_seconds > 0:
                sleep(self.retry_policy.retry_delay_seconds)
            return {"route": "retry", "attempt": attempt + 1}

        self.service.record_trial(
            self.study,
            trial_id,
            status="failed",
            metrics=result.get("metrics") or {},
            intermediate_metrics=result.get("intermediate_metrics") or [],
            cost={
                **(result.get("cost") or {}),
                "attempts": attempt,
                "failure_category": failure.category,
                "recoverable": failure.recoverable,
            },
            artifacts=result.get("artifacts") or [],
            stop_reason=error,
        )
        errors = list(state.get("errors") or [])
        errors.append(f"{trial_id}: {failure.category}: {error}")
        completed = int(state.get("completed_since_review", 0)) + 1
        return {"route": "review", "errors": errors, "completed_since_review": completed}

    def _review_strategy(self, state: HPOGraphState) -> Dict[str, Any]:
        completed = int(state.get("completed_since_review", 0))
        if self.study.strategy == "successive_halving" or completed < self.review_interval_trials:
            return {}
        proposal = self._make_strategy_proposal("interval_review")
        self.service.review_strategy(self.study, proposal, trigger=f"after_{completed}_trials")
        return {"completed_since_review": 0}

    def _promote(self, state: HPOGraphState) -> Dict[str, Any]:
        source_rung = self._next_promotable_rung()
        if source_rung is not None:
            proposal = self._make_strategy_proposal(f"after_rung_{source_rung}")
            self.service.review_strategy(self.study, proposal, trigger=f"after_rung_{source_rung}")
        promoted = self.service.promote_trials(self.study)
        return {"route": "next" if promoted else "complete", "completed_since_review": 0 if promoted else state.get("completed_since_review", 0)}

    def _make_strategy_proposal(self, trigger: str) -> Optional[StrategyProposal]:
        if not self.strategy_reviewer:
            return None
        feedback = HPOFeedbackAnalyzer().analyze(
            self.study,
            self.service.list_trials(self.study.experiment_id),
        )
        feedback["review_trigger"] = trigger
        try:
            return self.strategy_reviewer(self.study, feedback)
        except Exception as exc:
            return StrategyProposal(
                action="invalid_proposal",
                reason_codes=["runtime_review_error"],
                evidence={"error": f"{type(exc).__name__}: {exc}", "review_trigger": trigger},
            )

    def _next_promotable_rung(self) -> Optional[int]:
        if self.study.strategy != "successive_halving":
            return None
        trials = self.service.list_trials(self.study.experiment_id)
        completed_by_rung: Dict[int, List[Trial]] = {}
        active_by_rung: Dict[int, List[Trial]] = {}
        for trial in trials:
            if trial.status == "completed":
                completed_by_rung.setdefault(trial.rung, []).append(trial)
            elif trial.status in {"suggested", "running"}:
                active_by_rung.setdefault(trial.rung, []).append(trial)
        eligible: List[int] = []
        for rung, items in completed_by_rung.items():
            if rung + 1 >= len(self.study.budgets) or len(items) < self.study.min_completed_per_rung:
                continue
            if active_by_rung.get(rung):
                continue
            limit = self.study.promotion_limits[rung] if rung < len(self.study.promotion_limits) else None
            destination_count = len([trial for trial in trials if trial.rung == rung + 1])
            if limit is None or destination_count < limit:
                eligible.append(rung)
        return min(eligible) if eligible else None

    def _complete(self, state: HPOGraphState) -> Dict[str, Any]:
        errors = list(state.get("errors") or [])
        if int(state.get("completed_since_review", 0)) > 0:
            proposal = self._make_strategy_proposal("final_trials")
            self.service.review_strategy(self.study, proposal, trigger="final_trials")
        try:
            self._study = self.service.complete_study(self.study, "langgraph_completed")
        except ValueError as exc:
            errors.append(str(exc))
            self._study = self.service.finish_study(self.study, "failed", str(exc))
        return {"errors": errors}


__all__ = [
    "DecisionPolicy",
    "HPOGraphState",
    "HPOScheduler",
    "SchedulerResult",
    "StrategyAdvisor",
    "StrategyReviewer",
    "TrialExecutor",
]
