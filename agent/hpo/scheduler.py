"""LangGraph-backed deterministic HPO workflow."""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.core.metrics import InvalidMetricError, is_finite_metric
from .contracts import HPOStudy, StrategyProposal, Trial
from .feedback import HPOFeedbackAnalyzer
from .policies import FailureDecision, FailurePolicy, RetryPolicy
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
        sampler = study.candidate_strategy or study.sampler_strategy or study.strategy
        if sampler == "adaptive_search" and HPOService._active_pruner(study) == "none":
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
        if service._active_pruner(study) != "successive_halving":
            return False
        trials = service.list_trials(study.experiment_id)
        initial_limit = study.initial_trial_count or study.max_trials or 1
        if len([trial for trial in trials if trial.rung == 0]) < initial_limit:
            return False
        return any(
            trial.status == "completed" and trial.rung + 1 < len(study.budgets)
            for trial in trials
        ) and service.remaining_training_runs(study) > 0

    def should_suggest(self, study: HPOStudy, service: HPOService) -> bool:
        if service.remaining_training_runs(study) <= 0:
            return False
        if service._active_pruner(study) == "none":
            return True
        trials = service.list_trials(study.experiment_id)
        initial_limit = study.initial_trial_count or study.max_trials or 1
        return len([trial for trial in trials if trial.rung == 0]) < initial_limit


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
        self._resume_record: Dict[str, Any] = {}
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
        # Resume may have committed results but not the following batch review.
        # Review/select are safe for both a fresh Study and a partial batch.
        graph.add_edge("advise", "review_strategy")
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
        graph.add_conditional_edges(
            "complete",
            lambda state: state["route"],
            {"next": "select_trial", "done": END},
        )
        return graph.compile()

    @property
    def study(self) -> HPOStudy:
        if self._study is None:
            raise RuntimeError("HPO graph has no active study")
        return self._study

    def run(self, study: HPOStudy, *, resume: bool = False) -> SchedulerResult:
        if resume:
            study = self.service.load_study(study.experiment_id)
        self._study = study
        if self.study.controller_mode == "auto":
            self.study.controller_mode = "llm" if self.strategy_reviewer else "rule"
        self._resume_record = self.service.prepare_resume(study) if resume else {}
        if study.status == "completed":
            self.service._refresh_study(study)
            errors = self.service.completion_errors(study)
            if errors:
                self.service.finish_study(study, "failed", "; ".join(errors))
            return SchedulerResult(
                study=study,
                trials=self.service.list_trials(study.experiment_id),
                errors=errors,
                advice={"resume": self._resume_record},
                strategy_reviews=study.strategy_reviews,
            )
        scheduler_configuration = dict(
            study.scheduler_state.get("configuration") or {}
        )
        if resume and scheduler_configuration:
            self.review_interval_trials = max(
                int(
                    scheduler_configuration.get("review_interval_trials")
                    or self.review_interval_trials
                ),
                1,
            )
            if hasattr(self.retry_policy, "max_retries"):
                self.retry_policy.max_retries = max(
                    int(
                        scheduler_configuration.get("max_retries")
                        if scheduler_configuration.get("max_retries") is not None
                        else self.retry_policy.max_retries
                    ),
                    0,
                )
            if hasattr(self.retry_policy, "retry_delay_seconds"):
                self.retry_policy.retry_delay_seconds = max(
                    float(
                        scheduler_configuration.get("retry_delay_seconds")
                        if scheduler_configuration.get("retry_delay_seconds") is not None
                        else self.retry_policy.retry_delay_seconds
                    ),
                    0.0,
                )
        else:
            scheduler_configuration = {
                "review_interval_trials": self.review_interval_trials,
                "max_retries": getattr(self.retry_policy, "max_retries", None),
                "retry_delay_seconds": getattr(
                    self.retry_policy,
                    "retry_delay_seconds",
                    None,
                ),
            }
        completed_since_review = self.service.unreviewed_trial_count(study)
        self.service.update_scheduler_state(
            study,
            current_trial_id=None,
            completed_since_review=completed_since_review,
            terminal=False,
            terminal_status=None,
            configuration=scheduler_configuration,
        )
        state = self.graph.invoke(
            {
                "experiment_id": study.experiment_id,
                "attempt": 1,
                "errors": [],
                "advice": {},
                "completed_since_review": completed_since_review,
            },
            config={"recursion_limit": max(50, int(study.max_training_runs or 1) * 8)},
        )
        current = self.service.load_study(study.experiment_id)
        return SchedulerResult(
            study=current,
            trials=self.service.list_trials(study.experiment_id),
            errors=state.get("errors", []),
            advice={
                **state.get("advice", {}),
                **({"resume": self._resume_record} if self._resume_record else {}),
            },
            strategy_reviews=current.strategy_reviews,
        )

    def _advise(self, state: HPOGraphState) -> Dict[str, Any]:
        try:
            advice = self.strategy_advisor(self.study) if self.strategy_advisor else {}
        except Exception as exc:
            advice = {"advice_error": f"{type(exc).__name__}: {exc}"}
        return {"advice": dict(advice or {})}

    def _suggest(self, state: HPOGraphState) -> Dict[str, Any]:
        existing = self.service.list_trials(self.study.experiment_id)
        if any(trial.status == "suggested" for trial in existing):
            return {"route": "next"}
        count = self.study.candidate_batch_size or self.decision_policy.initial_count(self.study)
        if self.study.candidate_batch_size is not None:
            count = min(count, self.review_interval_trials)
        if self.service._active_pruner(self.study) == "none":
            count = min(count, self.review_interval_trials)
        created = self.service.suggest_trials(
            self.study,
            count,
        )
        if created:
            return {"route": "next"}
        if (
            not self.decision_policy.should_suggest(self.study, self.service)
            and self.decision_policy.should_promote(self.study, self.service)
            and self._next_promotable_rung() is not None
        ):
            return {"route": "next"}
        return {"route": "complete"}

    def _select_trial(self, state: HPOGraphState) -> Dict[str, Any]:
        trial = self.decision_policy.next_trial(self.study, self.service)
        if trial is not None:
            attempt = max(1, int(trial.cost.get("retry_count", 0)) + 1)
            self.service.update_scheduler_state(
                self.study,
                current_trial_id=trial.trial_id,
                attempt=attempt,
            )
            return {"current_trial_id": trial.trial_id, "attempt": attempt, "route": "run"}
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
        invalid_metric = False
        if result.get("status") == "success":
            try:
                self.service.record_trial(
                    self.study,
                    trial_id,
                    status="completed",
                    metrics=result.get("metrics") or {},
                    intermediate_metrics=result.get("intermediate_metrics") or [],
                    cost={**(result.get("cost") or {}), "attempts": attempt},
                    artifacts=result.get("artifacts") or [],
                )
            except InvalidMetricError as exc:
                invalid_metric = True
                result = {**result, "status": "failed", "error": str(exc)}
        if result.get("status") == "success":
            completed = self.service.unreviewed_trial_count(self.study)
            self.service.update_scheduler_state(
                self.study,
                current_trial_id=None,
                completed_since_review=completed,
            )
            return {"route": "review", "last_error": None, "completed_since_review": completed}

        error = str(result.get("error") or "trial execution failed")
        failure = (
            FailureDecision("invalid_metric", False)
            if invalid_metric else self.failure_policy.classify(error)
        )
        retry_requested = self.retry_policy.should_retry(attempt, failure)
        can_retry = retry_requested and self.service.retry_budget_available(self.study)
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
            self.service.update_scheduler_state(
                self.study,
                current_trial_id=trial_id,
                attempt=attempt + 1,
            )
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
                **({
                    "retry_blocked_reason": "reserved_training_budget",
                    "reserved_training_runs": self.service.reserved_training_runs(self.study),
                    "remaining_training_runs": self.service.remaining_training_runs(self.study),
                } if retry_requested and not can_retry else {}),
            },
            artifacts=result.get("artifacts") or [],
            stop_reason=error,
        )
        errors = list(state.get("errors") or [])
        errors.append(f"{trial_id}: {failure.category}: {error}")
        completed = self.service.unreviewed_trial_count(self.study)
        self.service.update_scheduler_state(
            self.study,
            current_trial_id=None,
            completed_since_review=completed,
        )
        return {"route": "review", "errors": errors, "completed_since_review": completed}

    def _review_strategy(self, state: HPOGraphState) -> Dict[str, Any]:
        completed = self.service.unreviewed_trial_count(self.study)
        if completed == 0:
            return {"completed_since_review": 0}
        review_threshold = self.review_interval_trials
        if self.study.candidate_batch_size is not None:
            review_threshold = min(review_threshold, self.study.candidate_batch_size)
        agent_queue_exhausted = (
            (self.study.candidate_strategy or self.study.sampler_strategy)
            == "agent_proposal"
            and not self.study.pending_candidate_proposals
        )
        if completed < review_threshold and not agent_queue_exhausted:
            return {}
        trials = self.service.list_trials(self.study.experiment_id)
        if any(trial.status in {"suggested", "running"} for trial in trials):
            return {}
        if self.service._active_pruner(self.study) == "successive_halving":
            initial_limit = self.study.initial_trial_count or self.study.max_trials or 1
            if len([trial for trial in trials if trial.rung == 0]) >= initial_limit:
                # The promotion node performs the rung-boundary review.
                return {}
        proposal = self._make_strategy_proposal("interval_review")
        self.service.review_strategy(self.study, proposal, trigger=f"after_{completed}_trials")
        self.service.update_scheduler_state(
            self.study,
            current_trial_id=None,
            completed_since_review=0,
        )
        return {"completed_since_review": 0}

    def _promote(self, state: HPOGraphState) -> Dict[str, Any]:
        source_rung = self._next_promotable_rung()
        if source_rung is not None and self.service.unreviewed_trial_count(self.study) > 0:
            proposal = self._make_strategy_proposal(f"after_rung_{source_rung}")
            self.service.review_strategy(self.study, proposal, trigger=f"after_rung_{source_rung}")
        promoted = self.service.promote_trials(self.study)
        if promoted:
            self.service.update_scheduler_state(
                self.study,
                current_trial_id=None,
                completed_since_review=0,
            )
        return {"route": "next" if promoted else "complete", "completed_since_review": 0 if promoted else state.get("completed_since_review", 0)}

    def _make_strategy_proposal(self, trigger: str) -> Optional[StrategyProposal]:
        if self.study.controller_mode != "llm" or not self.strategy_reviewer:
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
        if self.service._active_pruner(self.study) != "successive_halving":
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
        for rung in completed_by_rung:
            cohort_size = sum(
                trial.rung == rung and trial.status in {"completed", "promoted"}
                and is_finite_metric(trial.metrics.get(self.study.objectives[0].metric))
                for trial in trials
            )
            if rung + 1 >= len(self.study.budgets) or cohort_size < self.study.min_completed_per_rung:
                continue
            if active_by_rung.get(rung):
                continue
            limit = self.study.promotion_limits[rung] if rung < len(self.study.promotion_limits) else None
            destination_count = len([trial for trial in trials if trial.rung == rung + 1])
            remaining_limit = max(limit - destination_count, 0) if limit is not None else None
            if self.service.halving_strategy.promote(
                trials, self.study.objectives[0], self.study.reduction_factor,
                rung=rung, limit=remaining_limit,
            ):
                eligible.append(rung)
        return min(eligible) if eligible else None

    def _complete(self, state: HPOGraphState) -> Dict[str, Any]:
        errors = list(state.get("errors") or [])
        reviewed = self.service.unreviewed_trial_count(self.study) > 0
        if reviewed:
            proposal = self._make_strategy_proposal("final_trials")
            self.service.review_strategy(self.study, proposal, trigger="final_trials")
        if (
            reviewed
            and self.study.pending_candidate_proposals
            and self.service.has_future_candidate_capacity(self.study)
        ):
            return {"errors": errors, "route": "next", "completed_since_review": 0}
        try:
            self._study = self.service.complete_study(self.study, "langgraph_completed")
        except ValueError as exc:
            errors.append(str(exc))
            self._study = self.service.finish_study(self.study, "failed", str(exc))
        return {"errors": errors, "route": "done"}


__all__ = [
    "DecisionPolicy",
    "HPOGraphState",
    "HPOScheduler",
    "SchedulerResult",
    "StrategyAdvisor",
    "StrategyReviewer",
    "TrialExecutor",
]
