"""LangGraph HPO agent with a single structured task interface."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.agents.base_agent import LangGraphAgent
from agent.agents.hpo_execution import TrialExecutor
from agent.hpo import reporting, resource_analysis
from agent.agents.advisory_response import decode_json_response, response_diagnostics, trial_ids_from_evidence
from agent.agents.communication import AgentTaskRequest, AgentTaskResult
from agent.hpo.effects import summarize_decision_effect
from agent.hpo.history import budget_key
from agent.hpo.policies import SystemFailureBlocked
from agent.hpo.resources import (
    CampaignResourceBudget, ResourceBudgetExhausted, positive_limit,
)
from agent.hpo.protocol import (
    METRIC_PROTOCOL_ID,
    METRIC_UNITS,
    file_sha256,
    resolve_hpo_validation_protocol,
)
from agent.hpo import (
    HPOFeedbackAnalyzer,
    HPOPlanningPolicy,
    HPOScheduler,
    HPOService,
    Objective,
    OptimizationCampaign,
    OptimizationPlanDecisionPolicy,
    CampaignPolicy,
    RetryPolicy,
    SearchParameter,
    SearchSpace,
    StrategyProposal,
    TrialBudget,
    study_confirmation_signature,
)
from agent.data_processing.handoff import resolve_data_handoff
from agent.memory import EpisodeMemory, MemoryQuery, MemoryScope, MemoryService
from agent.models import get_model_adapter
from agent.prompt import render_prompt
from agent.utils import ConfigParser, ExperimentTracker
from agent.utils.path_tool import (
    get_config_file,
    get_hpo_experiments_dir,
    resolve_config_path,
    resolve_config_value_path,
)


class HPOAgent(LangGraphAgent):
    """Execute validated HPO workflows; the LLM may only submit structured proposals."""

    action = "optimize_hyperparameters"
    proposal_actions = {
        "keep_strategy",
        "refine_search_space",
        "expand_search_space",
        "switch_strategy",
        "adjust_budget",
    }

    def __init__(
        self,
        model_name: str = "GLM-4.7",
        temperature: float = 0.2,
        max_iterations: int = 10,
        verbose: bool = True,
        config_path: str = str(get_config_file("train_ecapa_tdnn.yaml")),
        experiments_dir: Optional[str] = None,
        task_type: str = "speaker_verification",
        model_family: str = "ecapa_tdnn",
        implementation: str = "speechbrain",
        runner: str = "speechbrain",
        enable_llm_advisor: bool = False,
        planning_policy: Optional[HPOPlanningPolicy] = None,
        decision_policy: Optional[OptimizationPlanDecisionPolicy] = None,
        memory_service: Optional[MemoryService] = None,
    ) -> None:
        super().__init__(model_name, temperature, max_iterations, verbose)
        self.config_path = str(resolve_config_path(config_path))
        self.experiments_dir = Path(experiments_dir).resolve() if experiments_dir else get_hpo_experiments_dir()
        self.task_type = task_type
        self.model_family = model_family
        self.implementation = implementation
        self.runner = runner
        self.enable_llm_advisor = enable_llm_advisor
        self.planning_policy = planning_policy or HPOPlanningPolicy()
        self.decision_policy = decision_policy or OptimizationPlanDecisionPolicy(self.planning_policy)
        self.memory_service = memory_service if memory_service is not None else MemoryService()
        self.memory_scope = MemoryScope(
            agent_type="hpo_agent",
            task_type=task_type,
            model_family=model_family,
            tags=["optimization", "langgraph", METRIC_PROTOCOL_ID],
        )

    def run_workflow(self, request: AgentTaskRequest) -> AgentTaskResult:
        self._resource_budget = None
        self._resource_campaign = None
        self._resource_experiment_ids = []
        try:
            return self._run_workflow(request)
        except SystemFailureBlocked as exc:
            campaign = self._resource_campaign
            if campaign:
                campaign.status, campaign.stop_reason = "failed", str(exc)
            tracker = ExperimentTracker(self.experiments_dir)
            for experiment_id in self._resource_experiment_ids:
                tracker.update_hpo_experiment(experiment_id, extensions={"optimization": {
                    "campaign": campaign.to_dict() if campaign else {},
                    "campaign_failure_guard": exc.guard,
                }})
            return AgentTaskResult(
                status="failed", error=str(exc), request_id=request.request_id,
                summary={"campaign": campaign.to_dict() if campaign else {}, "failure_guard": exc.guard},
                experiment_ids={"campaign": list(self._resource_experiment_ids),
                                **({"hpo": self._resource_experiment_ids[-1]} if self._resource_experiment_ids else {})},
            )
        except ResourceBudgetExhausted as exc:
            campaign = self._resource_campaign
            if campaign:
                campaign.status, campaign.stop_reason = "cancelled", str(exc)
                campaign.resource_budget = self._resource_budget.snapshot()
            tracker = ExperimentTracker(self.experiments_dir)
            for experiment_id in self._resource_experiment_ids:
                record = tracker.get_experiment(experiment_id) or {}
                if record.get("status") != "success":
                    tracker.update_hpo_experiment(experiment_id, status="failed", error=str(exc))
                    service = HPOService(tracker)
                    try:
                        study = service.load_study(experiment_id)
                    except FileNotFoundError:
                        continue
                    if study.status not in {"completed", "cancelled"}:
                        service.update_scheduler_state(study, resource_budget=self._resource_budget.snapshot())
                        service.finish_study(study, "cancelled", str(exc))
            return AgentTaskResult(
                status="failed", error=str(exc), request_id=request.request_id,
                experiment_ids={
                    "campaign": list(self._resource_experiment_ids),
                    **({"hpo": self._resource_experiment_ids[-1]} if self._resource_experiment_ids else {}),
                },
                summary={"campaign": campaign.to_dict() if campaign else {},
                         "resource_budget": self._resource_budget.snapshot() if self._resource_budget else {}},
            )
        finally:
            resources = self._resource_budget
            if resources:
                try:
                    snapshot = resources.snapshot()
                    if self._resource_campaign:
                        self._resource_campaign.resource_budget = snapshot
                    tracker = ExperimentTracker(self.experiments_dir)
                    for experiment_id in self._resource_experiment_ids:
                        tracker.update_hpo_experiment(
                            experiment_id, extensions={"optimization": {
                                "resource_budget": snapshot,
                                "campaign": self._resource_campaign.to_dict(),
                            }},
                        )
                finally:
                    resources.close()

    def _bind_resource_budget(self, request, campaign, *, resume=False):
        self._resource_campaign = campaign
        profile = request.context.get("resource_profile") or {}
        limits = {}
        for key, aliases in {
            "max_gpu_hours": ("max_gpu_hours", "gpu_hours", "gpu_hours_remaining"),
            "max_wall_seconds": ("max_wall_seconds", "wall_seconds", "wall_seconds_remaining"),
        }.items():
            values = [positive_limit(request.budget.get(key), key)]
            values += [positive_limit(profile.get(alias), alias) for alias in aliases]
            limits[key] = min((value for value in values if value is not None), default=None)
        persisted = campaign.resource_budget
        if not persisted and not any(value is not None for value in limits.values()):
            return
        if resume and not persisted:
            raise ValueError("cannot add resource limits when resuming a legacy Campaign with unaccounted costs")
        path = Path(self.experiments_dir) / "campaign_resources" / f"{campaign.campaign_id}.json"
        if path.parent.resolve() != (Path(self.experiments_dir) / "campaign_resources").resolve():
            raise ValueError("invalid persisted Campaign resource identifier")
        self._resource_budget = CampaignResourceBudget(path, limits=limits, resume=resume)
        self._resource_campaign = campaign
        campaign.resource_budget = self._resource_budget.snapshot()
        self._resource_budget.check()

    def _check_resource_budget(self):
        resources = getattr(self, "_resource_budget", None)
        if resources:
            return resources.check()
        return None

    def _run_workflow(self, request: AgentTaskRequest) -> AgentTaskResult:
        started_at = datetime.now()
        tracker = ExperimentTracker(self.experiments_dir)
        resume_experiment_id = request.context.get("resume_experiment_id")
        if resume_experiment_id is not None:
            resume_record = tracker.get_experiment(str(resume_experiment_id))
            if resume_record is None:
                raise ValueError(f"resume experiment not found: {resume_experiment_id}")
            resume_task = dict(resume_record.get("task") or {})
            resume_data_folder = str(resume_task.get("dataset") or "")
            resume_handoff = {
                "consumer_uri": resume_data_folder,
                "dataset_id": resume_task.get("dataset_id"),
                "dataset_version": resume_task.get("dataset_version"),
                "data_processing_experiment_id": resume_task.get(
                    "data_processing_experiment_id"
                ),
            }
            return self._resume_existing_study(
                request,
                tracker,
                str(resume_experiment_id),
                resume_data_folder,
                resume_handoff,
                started_at,
            )
        config = ConfigParser(self.config_path).load_config(resolve_references=True)
        data_handoff = resolve_data_handoff(request.context, config.get("data_folder"))
        data_folder = data_handoff["consumer_uri"]
        output = resolve_config_value_path(config.get("output_folder"))
        self.memory_scope.dataset_key = data_folder

        # HPO is validation-only. Resolve these inputs before creating any
        # experiment/Study state so a missing split cannot silently fall back
        # to a recipe's official test list.
        runtime_options = resolve_hpo_validation_protocol(
            request.context.get("runtime_options"),
            require_explicit=True,
        )
        if self.runner == "speechbrain":
            from agent.runners.speechbrain_validation import validate_verification_config

            validate_verification_config(runtime_options["verification_config"])
        if self.runner == "speechbrain" and runtime_options.get("ddp_devices"):
            from agent.runners.speechbrain_distributed import (
                freeze_ddp_runtime_options,
            )

            runtime_options = freeze_ddp_runtime_options(runtime_options)
        request.context["runtime_options"] = runtime_options
        controller_mode = str(
            request.context.get("controller_mode")
            or ("llm" if self.enable_llm_advisor else "fixed")
        )
        if controller_mode not in {"fixed", "rule", "llm"}:
            raise ValueError("controller_mode must be one of: fixed, rule, llm")
        if controller_mode == "llm" and not self.enable_llm_advisor:
            raise ValueError("controller_mode='llm' requires enable_llm_advisor=True")
        request.context["controller_mode"] = controller_mode

        search_space = self._resolve_search_space(request.context.get("search_space"))
        per_study_limit = int(request.budget.get("max_training_runs") or self.max_iterations)
        requested_strategy = str(request.context.get("strategy") or "auto")
        requested_sampler = request.context.get("sampler")
        requested_pruner = request.context.get("pruner")
        budgets = self._build_budgets(
            request.context.get("budgets"),
            requested_pruner or requested_strategy,
        )
        self._validate_search_budget_compatibility(search_space, budgets)
        model_adapter = get_model_adapter(self.model_family)
        parameter_validator = getattr(model_adapter, "validate_parameters", None)
        service = HPOService(
            tracker,
            parameter_validator=parameter_validator if callable(parameter_validator) else None,
        )
        base_sampler, base_pruner = self.planning_policy.select_components(
            requested_strategy,
            str(requested_sampler) if requested_sampler is not None else None,
            str(requested_pruner) if requested_pruner is not None else None,
            search_space,
            budgets,
            per_study_limit,
            service.available_samplers(),
        )
        if base_pruner == "none" and len(budgets) > 1:
            budgets = [budgets[-1]]
        objectives = [Objective(
            str(request.context.get("primary_metric", "eer")),
            str(request.context.get("metric_mode", "min")),
        )]
        max_studies = max(int(request.budget.get("max_studies", 1)), 1)
        target_value = request.context.get("target_value", request.budget.get("target_value"))
        campaign = OptimizationCampaign(
            objective=objectives[0],
            target_value=float(target_value) if target_value is not None else None,
            max_studies=max_studies,
            patience=max(int(request.budget.get("campaign_patience", max_studies)), 1),
            min_improvement=float(request.budget.get("campaign_min_improvement", 0.0)),
            max_total_training_runs=int(request.budget.get("max_total_training_runs") or per_study_limit * max_studies),
        )
        self._bind_resource_budget(request, campaign)
        confirmation_budget = budgets[-1]
        history_comparison_signature = self._confirmation_signature_from_inputs(
            objective=objectives[0],
            confirmation_budget=confirmation_budget,
            data_folder=data_folder,
            data_handoff=data_handoff,
            runtime_options=runtime_options,
        )
        campaign_policy = CampaignPolicy()
        study_results: List[Dict[str, Any]] = []
        best_trial = None
        best_experiment_id = None
        sampler = base_sampler
        pruner = base_pruner
        reduction_factor = int(request.budget.get("reduction_factor", 3))
        requested_sampler_config = request.context.get("sampler_config")
        sampler_config_state = (
            dict(requested_sampler_config)
            if isinstance(requested_sampler_config, dict)
            else {}
        )
        campaign_history: List[Dict[str, Any]] = []
        for study_index in range(max_studies):
            self._check_resource_budget()
            pre_inheritance_sampler = sampler
            remaining = campaign_policy.remaining_runs(campaign)
            run_limit = min(per_study_limit, remaining) if remaining is not None else per_study_limit
            if run_limit <= 0:
                campaign.status, campaign.stop_reason = "completed", "max_total_training_runs_reached"
                break
            base_initial_count = min(
                int(
                    request.budget.get("initial_trial_count")
                    or self._default_initial_trial_count(
                        "successive_halving" if pruner == "successive_halving" else sampler,
                        budgets,
                        run_limit,
                    )
                ),
                run_limit,
            )
            base_promotions = (
                list(request.budget.get("promotion_limits") or self._default_promotion_limits(
                    base_initial_count,
                    budgets,
                    reduction_factor,
                ))
                if pruner == "successive_halving"
                else []
            )
            inherited_proposal = self._next_study_proposal(
                campaign.to_dict(), controller_mode
            )
            inheritance_decision = None
            if inherited_proposal is not None:
                inheritance_decision = self.decision_policy.review(
                    inherited_proposal,
                    base_sampler=sampler,
                    base_pruner=pruner,
                    base_search_space=search_space,
                    base_budgets=budgets,
                    hard_max_training_runs=run_limit,
                    objectives=objectives,
                    available_strategies=service.available_samplers(),
                    validate_plan=service.validate_study_plan,
                    base_initial_trial_count=base_initial_count,
                    base_promotion_limits=base_promotions,
                    base_reduction_factor=reduction_factor,
                    confirmation_budget=confirmation_budget,
                )
                inheritance_decision.scope = "next_study_inheritance"
                inheritance_decision.effective_from_trial_index = 0
                sampler = str(inheritance_decision.adopted_sampler or sampler)
                pruner = str(inheritance_decision.adopted_pruner or pruner)
                search_space = self._resolve_search_space(
                    inheritance_decision.adopted_search_space
                )
                budgets = self._build_budgets(
                    inheritance_decision.adopted_budgets, pruner
                )
                if pruner == "none" and len(budgets) > 1:
                    budgets = [budgets[-1]]
                self._validate_search_budget_compatibility(search_space, budgets)
                run_limit = int(inheritance_decision.adopted_max_training_runs)
                reduction_factor = int(inheritance_decision.adopted_reduction_factor)
                base_initial_count = min(
                    int(inheritance_decision.adopted_initial_trial_count or run_limit),
                    run_limit,
                )
                base_promotions = (
                    list(inheritance_decision.adopted_promotion_limits)
                    if pruner == "successive_halving"
                    else []
                )
                inherited_sampler_config = (
                    inherited_proposal.sampler_config
                    if inherited_proposal.sampler_config
                    else sampler_config_state
                )
                try:
                    sampler_config_state = service.validate_sampler_config(
                        sampler, inherited_sampler_config
                    )
                    if inherited_proposal.sampler_config:
                        inheritance_decision.accepted_fields.append("sampler_config")
                except Exception as exc:
                    sampler_config_state = service.validate_sampler_config(sampler, {})
                    inheritance_decision.rejected_fields.append({
                        "field": "sampler_config",
                        "reason": str(exc),
                    })
                inheritance_decision.adopted_sampler_config = dict(
                    sampler_config_state
                )
                inheritance_decision.accepted_fields = list(dict.fromkeys(
                    inheritance_decision.accepted_fields
                ))
                inheritance_decision.applied = bool(
                    inheritance_decision.accepted_fields
                )
                if inheritance_decision.rejected_fields:
                    inheritance_decision.decision = (
                        "approved_with_changes"
                        if inheritance_decision.applied else "rejected"
                    )

            study_base_sampler = (
                sampler
                if sampler != "agent_proposal"
                else pre_inheritance_sampler
                if pre_inheritance_sampler != "agent_proposal"
                else "random_search"
            )
            submitted_proposal = (
                self._planning_proposal(
                    request,
                    search_space,
                    budgets,
                    sampler,
                    sampler_config_state,
                    pruner,
                    run_limit,
                    base_initial_count,
                    base_promotions,
                    reduction_factor,
                    campaign.to_dict(),
                    history_comparison_signature,
                )
                if controller_mode == "llm" else None
            )
            proposal = (
                submitted_proposal
                if submitted_proposal
                and submitted_proposal.action in self.proposal_actions
                else inherited_proposal
            )
            decision = self.decision_policy.review(
                proposal,
                base_sampler=sampler,
                base_pruner=pruner,
                base_search_space=search_space,
                base_budgets=budgets,
                hard_max_training_runs=run_limit,
                objectives=objectives,
                available_strategies=service.available_samplers(),
                validate_plan=service.validate_study_plan,
                base_initial_trial_count=base_initial_count,
                base_promotion_limits=base_promotions,
                base_reduction_factor=reduction_factor,
                confirmation_budget=confirmation_budget,
            )
            decision.scope = "study_planning"
            decision.effective_from_trial_index = 0
            # A planning decision always becomes the concrete Study plan, even
            # when it simply adopts the fixed/base configuration unchanged.
            decision.applied = True
            sampler = str(decision.adopted_sampler or sampler)
            pruner = str(decision.adopted_pruner or pruner)
            search_space = self._resolve_search_space(decision.adopted_search_space)
            budgets = self._build_budgets(decision.adopted_budgets, pruner)
            if pruner == "none" and len(budgets) > 1:
                budgets = [budgets[-1]]
            self._validate_search_budget_compatibility(search_space, budgets)
            run_limit = decision.adopted_max_training_runs
            reduction_factor = decision.adopted_reduction_factor
            initial_count = min(
                int(decision.adopted_initial_trial_count or run_limit),
                run_limit,
            )
            promotion_limits = (
                list(decision.adopted_promotion_limits)
                if pruner == "successive_halving"
                else []
            )
            if initial_count + sum(promotion_limits) > run_limit:
                initial_count = self._default_initial_trial_count(
                    "successive_halving" if pruner == "successive_halving" else sampler,
                    budgets,
                    run_limit,
                )
                promotion_limits = (
                    self._default_promotion_limits(initial_count, budgets, reduction_factor)
                    if pruner == "successive_halving"
                    else []
                )
            decision.adopted_initial_trial_count = initial_count
            decision.adopted_promotion_limits = promotion_limits
            proposed_sampler_config = (
                proposal.sampler_config
                if proposal and proposal.sampler_config
                else sampler_config_state
            )
            try:
                sampler_config = service.validate_sampler_config(
                    sampler, proposed_sampler_config
                )
                if proposed_sampler_config:
                    decision.accepted_fields.append("sampler_config")
            except Exception as exc:
                sampler_config = service.validate_sampler_config(sampler, {})
                decision.rejected_fields.append({
                    "field": "sampler_config",
                    "reason": str(exc),
                })
            decision.adopted_sampler_config = dict(sampler_config)
            sampler_config_state = dict(sampler_config)
            legacy_strategy = (
                "successive_halving"
                if pruner == "successive_halving"
                else sampler
            )
            resource_profile = self._resource_snapshot(
                request.context.get("runtime_options"),
                request.context.get("resource_profile"),
            )
            search_budget_analysis = self._search_budget_analysis(
                search_space,
                budgets,
                run_limit,
                initial_count,
                promotion_limits,
                reduction_factor,
                campaign.to_dict(),
            )
            adopted_plan = {
                "strategy": legacy_strategy,
                "sampler": sampler,
                "pruner": pruner,
                "search_space": search_space.to_dict(),
                "budgets": [budget.to_dict() for budget in budgets],
                "max_training_runs": run_limit,
                "initial_trial_count": initial_count,
                "promotion_limits": promotion_limits,
                "reduction_factor": reduction_factor,
                "warm_start_trial_count": len(campaign_history),
                "resource_profile": resource_profile,
                "search_budget_analysis": search_budget_analysis,
                "candidate_batch_size": max(
                    int(
                        request.budget.get("candidate_batch_size")
                        or request.budget.get("strategy_review_interval_trials", 3)
                    ),
                    1,
                ),
                "controller_mode": controller_mode,
                "confirmation_budget": confirmation_budget.to_dict(),
                "next_study_inheritance": (
                    inheritance_decision.to_dict()
                    if inheritance_decision is not None else None
                ),
                "planning_proposal_error": (
                    submitted_proposal.to_dict()
                    if submitted_proposal is not None
                    and submitted_proposal.action not in self.proposal_actions
                    else None
                ),
            }
            experiment_id = self._create_study_record(
                tracker, campaign, study_index, objectives[0], data_handoff,
                runtime_options, output, controller_mode,
            )
            self._resource_experiment_ids.append(experiment_id)
            self._check_resource_budget()
            study = service.create_study(
                experiment_id,
                search_space,
                objectives,
                budgets,
                strategy=legacy_strategy,
                sampler_strategy=sampler,
                pruner_strategy=pruner,
                reduction_factor=reduction_factor,
                max_trials=run_limit,
                initial_trial_count=initial_count,
                promotion_limits=promotion_limits,
                max_training_runs=run_limit,
                min_completed_per_rung=int(request.budget.get("min_completed_per_rung", 1)),
                warm_start_trials=campaign_history,
                candidate_batch_size=adopted_plan["candidate_batch_size"],
                sampler_config=sampler_config,
                fallback_sampler_strategy=(
                    study_base_sampler
                    if study_base_sampler != "agent_proposal" else "random_search"
                ),
                hypotheses=(
                    proposal.hypotheses
                    if proposal
                    else []
                ),
                candidate_proposals=(
                    proposal.candidate_proposals
                    if proposal
                    else []
                ),
                proposal_id=(
                    proposal.proposal_id if proposal else None
                ),
                planning_decision_id=decision.decision_id,
                controller_mode=controller_mode,
                random_seed=int(request.context.get("hpo_seed", config.get("seed", 0)) or 0) + study_index,
            )
            confirmation_signature = study_confirmation_signature(study)
            if not campaign.confirmation_signature:
                campaign.confirmation_signature = confirmation_signature
            elif campaign.confirmation_signature != confirmation_signature:
                raise ValueError(
                    "Study confirmation protocol differs from the frozen Campaign protocol"
                )
            actual_sampler = str(study.candidate_strategy or study.sampler_strategy or sampler)
            if actual_sampler != sampler:
                decision.rejected_fields.append({
                    "field": "requested_sampler",
                    "reason": (
                        "agent_proposal had no valid candidates; "
                        f"restored {actual_sampler}"
                    ),
                })
                sampler = actual_sampler
                decision.adopted_sampler = actual_sampler
                decision.adopted_strategy = actual_sampler
                decision.adopted_sampler_config = dict(study.sampler_config)
                adopted_plan["strategy"] = (
                    "successive_halving"
                    if pruner == "successive_halving" else actual_sampler
                )
            planning_candidate_review = next(
                (
                    item for item in reversed(study.candidate_proposal_reviews)
                    if item.get("trigger") == "study_planning"
                ),
                None,
            )
            if planning_candidate_review:
                if planning_candidate_review.get("accepted"):
                    decision.accepted_fields.append("candidate_proposals")
                if planning_candidate_review.get("rejected"):
                    decision.rejected_fields.append({
                        "field": "candidate_proposals",
                        "reason": planning_candidate_review["rejected"],
                    })
            if study.hypotheses:
                decision.accepted_fields.append("hypotheses")
            decision.accepted_fields = list(dict.fromkeys(decision.accepted_fields))
            if decision.rejected_fields:
                decision.decision = (
                    "approved_with_changes" if decision.accepted_fields else "rejected"
                )
                decision.reason_codes = [
                    "proposal_partially_approved"
                    if decision.accepted_fields else "proposal_rejected"
                ]
            elif decision.accepted_fields:
                decision.decision = "approved"
                decision.reason_codes = ["proposal_approved"]
            if inherited_proposal is not None:
                decision.reason_codes = list(dict.fromkeys([
                    *decision.reason_codes,
                    "next_study_proposal_inherited",
                ]))
            decision.applied = True
            decision.scope = "study_planning"
            decision.effective_from_trial_index = 0
            adopted_plan.update({
                "sampler": study.candidate_strategy or study.sampler_strategy,
                "sampler_config": dict(study.sampler_config),
                "accepted_hypothesis_count": len(study.hypotheses),
                "accepted_agent_candidate_count": len(study.pending_candidate_proposals),
            })
            tracker.update_hpo_experiment(
                experiment_id,
                extensions={"optimization": {"study_id": study.study_id}},
            )
            tracker.update_hpo_experiment(experiment_id, status="running", extensions={"optimization": {
                "objective": request.objective, "workflow": "langgraph", "campaign": campaign.to_dict(),
                "strategy_proposal": proposal.to_dict() if proposal else None,
                "strategy_decision": decision.to_dict(), "adopted_plan": adopted_plan, "data_handoff": data_handoff,
            }})
            scheduled = HPOScheduler(
                service, self._trial_executor(experiment_id, data_folder, runtime_options),
                resource_budget=self._resource_budget,
                retry_policy=RetryPolicy(int(request.budget.get("max_retries", 1))),
                strategy_reviewer=self._runtime_strategy_reviewer(request, campaign),
                review_interval_trials=int(request.budget.get("strategy_review_interval_trials", 3)),
            ).run(study)
            if scheduled.study.status != "completed":
                error = "; ".join(scheduled.errors) or "HPO workflow failed"
                tracker.update_hpo_experiment(experiment_id, status="failed", error=error)
                if scheduled.study.scheduler_state.get("resource_budget", {}).get("stop_reason"):
                    raise ResourceBudgetExhausted(error)
                guard = scheduled.study.scheduler_state.get("failure_guard") or {}
                if guard.get("status") == "open":
                    raise SystemFailureBlocked(error, guard)
                raise RuntimeError(error)
            planning_outcome = summarize_decision_effect(
                scheduled.trials,
                decision.decision_id,
                objectives[0],
                decision_created_at=decision.created_at,
            )
            decision.affected_trial_ids = planning_outcome["affected_trial_ids"]
            decision.realized_outcome = planning_outcome["realized_outcome"]
            decision.effect_estimate = planning_outcome["effect_estimate"]
            tracker.update_hpo_experiment(
                experiment_id,
                extensions={"optimization": {
                    "strategy_decision": decision.to_dict(),
                }},
            )
            current_best = service.load_trial(experiment_id, str(scheduled.study.best_trial_id))
            current_value = float(current_best.metrics[objectives[0].metric])
            campaign_policy.record_study(
                campaign, experiment_id=experiment_id, study_id=scheduled.study.study_id,
                best_value=current_value, training_runs=service.training_runs_used(scheduled.study),
                confirmation_signature=study_confirmation_signature(scheduled.study),
            )
            campaign.study_summaries[-1]["best_parameters"] = current_best.parameters
            campaign.study_summaries[-1]["best_metrics"] = self._best_metric_record(current_best, objectives[0])
            campaign.study_summaries[-1]["strategy_reviews"] = scheduled.study.strategy_reviews
            campaign.study_summaries[-1]["sampler"] = scheduled.study.candidate_strategy or scheduled.study.sampler_strategy
            campaign.study_summaries[-1]["pruner"] = scheduled.study.pruner_strategy
            campaign.study_summaries[-1]["warm_start_trial_count"] = len(scheduled.study.warm_start_trials)
            campaign.study_summaries[-1]["learning_summary"] = self._study_learning_summary(scheduled.study, current_best, objectives[0])
            campaign.study_summaries[-1]["next_study_proposal"] = scheduled.study.next_study_proposal
            study_feedback = HPOFeedbackAnalyzer().analyze(
                scheduled.study,
                scheduled.trials,
            )
            campaign.study_summaries[-1]["resource_summary"] = {
                "failure_rate": study_feedback.get("failure_rate"),
                "failure_clusters": study_feedback.get("failure_clusters") or {},
                "cost_summary": study_feedback.get("cost_summary") or {},
                "rung_summaries": study_feedback.get("rung_summaries") or [],
            }
            campaign_history = self._merge_campaign_history(
                campaign_history,
                scheduled.trials,
                objectives[0],
                scheduled.study.pruner_strategy or "none",
            )
            campaign.study_summaries[-1]["campaign_history_count"] = len(campaign_history)
            study_results.append({"experiment_id": experiment_id, "study_id": scheduled.study.study_id, "status": scheduled.study.status, "trial_count": len(scheduled.trials), "best_trial_id": scheduled.study.best_trial_id})
            if best_trial is None or (
                current_value < float(best_trial.metrics[objectives[0].metric])
                if objectives[0].mode == "min" else current_value > float(best_trial.metrics[objectives[0].metric])
            ):
                best_trial, best_experiment_id = current_best, experiment_id
            tracker.update_hpo_experiment(
                experiment_id,
                status="success",
                parameters=current_best.parameters,
                metrics={"best": self._best_metric_record(current_best, objectives[0])},
                extensions={"optimization": {
                    "campaign": campaign.to_dict(),
                    "latest_trial": {
                        "trial_id": current_best.trial_id,
                        "phase": "completed",
                        "status": current_best.status,
                        "updated_at": current_best.updated_at,
                    },
                }},
            )
            if not campaign_policy.should_continue(campaign):
                break
            self._check_resource_budget()
            sampler = (
                scheduled.study.candidate_strategy
                or scheduled.study.sampler_strategy
                or sampler
            )
            pruner = scheduled.study.pruner_strategy or pruner
            search_space = scheduled.study.search_space
            sampler_config_state = dict(scheduled.study.sampler_config or {})

        if best_trial is None or best_experiment_id is None:
            raise RuntimeError("optimization campaign produced no valid completed Study")
        duration = (datetime.now() - started_at).total_seconds()
        for item in study_results:
            tracker.update_hpo_experiment(
                item["experiment_id"],
                duration=duration,
                extensions={"optimization": {"campaign": campaign.to_dict()}},
            )
        self.memory_service.remember_episode(EpisodeMemory(
            agent_type="hpo_agent",
            objective=request.objective,
            action={
                "strategy": "successive_halving" if pruner == "successive_halving" else sampler,
                "sampler": sampler,
                "pruner": pruner,
                "best_config": best_trial.parameters,
            },
            outcome={"best_metrics": best_trial.metrics, "campaign": campaign.to_dict(), "duration": duration},
            summary=f"campaign completed {len(campaign.study_summaries)} studies: {campaign.stop_reason}",
            experiment_ids=[item["experiment_id"] for item in study_results],
            scope=self.memory_scope,
            importance=0.9,
        ))
        campaign_record = campaign.to_dict()
        if self._resource_budget:
            campaign_record["resource_budget"] = self._resource_budget.snapshot()
        pending_next_study = (
            (campaign_record.get("study_summaries") or [{}])[-1].get(
                "next_study_proposal"
            )
            if campaign_record.get("study_summaries")
            else None
        )
        return AgentTaskResult(
            status="success",
            summary={
                "strategy": "successive_halving" if pruner == "successive_halving" else sampler,
                "sampler": sampler,
                "pruner": pruner,
                "best_trial_id": best_trial.trial_id,
                "best_parameters": best_trial.parameters,
                "campaign": campaign_record,
                "studies": study_results,
                "data_handoff": data_handoff,
                "pending_next_study_proposal": pending_next_study,
            },
            metrics=self._best_metric_record(best_trial, objectives[0]),
            recommendations=(
                [{
                    "type": "pending_next_study_proposal",
                    "status": "not_executed_campaign_ended",
                    "proposal": pending_next_study,
                    "required_action": (
                        "run an additional Study under the same frozen "
                        "confirmation protocol"
                    ),
                }]
                if pending_next_study else []
            ),
            artifacts=best_trial.artifacts,
            experiment_ids={"hpo": best_experiment_id, "campaign": [item["experiment_id"] for item in study_results]},
            request_id=request.request_id,
        )

    def _create_study_record(
        self, tracker, campaign, study_index, objective, data_handoff,
        runtime_options, output, controller_mode,
    ) -> str:
        """Create the frozen identity/protocol record before scheduling Trials."""
        data_folder = data_handoff["consumer_uri"]
        return tracker.create_hpo_experiment(
            config_path=self.config_path,
            data_folder=data_folder,
            output_folder=str(output) if output else None,
            description=f"Optimization campaign {campaign.campaign_id} study {study_index + 1}",
            task={
                "type": self.task_type,
                "dataset": data_folder,
                "dataset_id": data_handoff.get("dataset_id"),
                "dataset_version": data_handoff.get("dataset_version"),
                "data_processing_experiment_id": data_handoff.get("data_processing_experiment_id"),
                "primary_metric": objective.metric,
                "metric_mode": objective.mode,
                "metric_protocol": METRIC_PROTOCOL_ID,
                "metric_units": dict(METRIC_UNITS),
                "controller_mode": controller_mode,
            },
            model={"family": self.model_family, "implementation": self.implementation, "config_path": self.config_path},
            execution={
                "runner": self.runner,
                "output_folder": str(output) if output else None,
                "evaluation_config_path": runtime_options["verification_config"],
                "validation_pairs_path": runtime_options["validation_pairs"],
                "training_exclusion_pairs_path": runtime_options[
                    "training_exclusion_pairs"
                ],
                "verification_config_sha256": runtime_options[
                    "verification_config_sha256"
                ],
                "validation_pairs_sha256": runtime_options[
                    "validation_pairs_sha256"
                ],
                "training_exclusion_pairs_sha256": runtime_options[
                    "training_exclusion_pairs_sha256"
                ],
                "training_runtime_options": self._training_runtime_options(
                    runtime_options
                ),
                "metric_protocol": METRIC_PROTOCOL_ID,
            },
            extra_fields={"version": {
                "campaign_id": campaign.campaign_id,
                "study_index": study_index + 1,
                "model_key": f"{self.task_type}/{self.model_family}/{self.implementation}",
                "dataset_id": data_handoff.get("dataset_id"),
                "dataset_version": data_handoff.get("dataset_version"),
                "data_processing_experiment_id": data_handoff.get("data_processing_experiment_id"),
            }},
        )


    def _resume_existing_study(
        self,
        request: AgentTaskRequest,
        tracker: ExperimentTracker,
        experiment_id: str,
        fallback_data_folder: str,
        data_handoff: Dict[str, Any],
        started_at: datetime,
    ) -> AgentTaskResult:
        record = tracker.get_experiment(experiment_id)
        if record is None:
            raise ValueError(f"resume experiment not found: {experiment_id}")
        if record.get("experiment_type") != "hpo":
            raise ValueError(f"resume target is not an HPO experiment: {experiment_id}")
        # Fail before reopening Trials or consuming retry budgets if the
        # original training configuration cannot be recovered.
        tracker.get_config_snapshot(experiment_id)
        task = dict(record.get("task") or {})
        model = dict(record.get("model") or {})
        execution = dict(record.get("execution") or {})
        data_folder = str(task.get("dataset") or fallback_data_folder)
        if not data_folder:
            raise ValueError("resume experiment does not record a dataset path")
        task_type = str(task.get("type") or self.task_type)
        model_family = str(model.get("family") or self.model_family)
        implementation = str(model.get("implementation") or self.implementation)
        runner = str(execution.get("runner") or self.runner)
        self.memory_scope.dataset_key = data_folder
        runtime_options = resolve_hpo_validation_protocol(
            request.context.get("runtime_options"),
            persisted_execution=execution,
            require_explicit=False,
        )
        persisted_training_runtime = dict(
            execution.get("training_runtime_options") or {}
        )
        if runtime_options.get("ddp_devices") is not None:
            requested_training_runtime = runtime_options
            if runner == "speechbrain":
                from agent.runners.speechbrain_distributed import (
                    freeze_ddp_runtime_options,
                )

                requested_training_runtime = freeze_ddp_runtime_options(
                    runtime_options
                )
            if self._training_runtime_signature(
                requested_training_runtime
            ) != self._training_runtime_signature(persisted_training_runtime):
                raise ValueError(
                    "resume DDP options conflict with the persisted Study "
                    "training runtime"
                )
        runtime_options.update(persisted_training_runtime)
        if runner == "speechbrain":
            from agent.runners.speechbrain_validation import validate_verification_config

            validate_verification_config(runtime_options["verification_config"])
        request.context["runtime_options"] = runtime_options

        model_adapter = get_model_adapter(model_family)
        validator = getattr(model_adapter, "validate_parameters", None)
        service = HPOService(
            tracker,
            parameter_validator=validator if callable(validator) else None,
        )
        study = service.load_study(experiment_id)
        if study.experiment_id != experiment_id:
            raise ValueError("persisted Study does not belong to the resume experiment")
        requested_controller = request.context.get("controller_mode")
        if study.controller_mode == "auto":
            study.controller_mode = str(
                requested_controller
                or ("llm" if self.enable_llm_advisor else "rule")
            )
        elif (
            requested_controller is not None
            and str(requested_controller) != study.controller_mode
        ):
            raise ValueError(
                "resume controller_mode conflicts with the persisted Study: "
                f"{requested_controller!r} != {study.controller_mode!r}"
            )
        if study.controller_mode == "llm" and not self.enable_llm_advisor:
            raise ValueError(
                "resuming an LLM-controlled Study requires enable_llm_advisor=True"
            )
        recorded_controller = task.get("controller_mode")
        if (
            recorded_controller is not None
            and str(recorded_controller) != study.controller_mode
        ):
            raise ValueError(
                "persisted experiment controller_mode conflicts with its Study"
            )
        persisted_objective = study.objectives[0]
        if (
            task.get("primary_metric") != persisted_objective.metric
            or task.get("metric_mode") != persisted_objective.mode
        ):
            raise ValueError(
                "persisted experiment task objective conflicts with its Study objective"
            )
        optimization = (record.get("extensions") or {}).get("optimization") or {}
        campaign = self._campaign_from_dict(
            optimization.get("campaign"),
            study.objectives[0],
        )
        self._resource_experiment_ids.append(experiment_id)
        self._bind_resource_budget(request, campaign, resume=True)
        campaign_guard = dict(optimization.get("campaign_failure_guard") or {})
        source_experiment = campaign_guard.get("experiment_id")
        if campaign_guard.get("status") == "open" and source_experiment and source_experiment != experiment_id:
            # Resuming an older successful Study must not bypass the failing
            # Study that stopped the same Campaign. Consult its live guard so a
            # repaired/reset source does not remain blocked by stale snapshots.
            source_study = service.load_study(str(source_experiment))
            source_guard = source_study.scheduler_state.get("failure_guard") or {}
            if source_guard.get("status") == "open":
                raise SystemFailureBlocked(
                    f"system_failure: Campaign blocked by {source_experiment}; repair and resume that Study first",
                    source_guard,
                )
        if request.context.get("reset_failure_guard") is True:
            campaign.status, campaign.stop_reason = "running", None
        resumed_signature = study_confirmation_signature(study)
        if not campaign.confirmation_signature:
            campaign.confirmation_signature = resumed_signature
        elif campaign.confirmation_signature != resumed_signature:
            raise ValueError(
                "persisted Study confirmation protocol conflicts with its Campaign"
            )
        tracker.update_hpo_experiment(
            experiment_id,
            status="running",
            error=None,
            extensions={"optimization": {
                "resume_requested_at": datetime.now().isoformat(),
                "campaign": campaign.to_dict(),
            }},
        )
        scheduled = HPOScheduler(
            service,
            self._trial_executor(
                experiment_id,
                data_folder,
                runtime_options,
                execution_context={
                    "task_type": task_type,
                    "model_family": model_family,
                    "implementation": implementation,
                    "runner": runner,
                },
            ),
            resource_budget=self._resource_budget,
            retry_policy=RetryPolicy(int(request.budget.get("max_retries", 1))),
            strategy_reviewer=self._runtime_strategy_reviewer(request, campaign),
            review_interval_trials=int(
                request.budget.get("strategy_review_interval_trials", 3)
            ),
        ).run(
            study, resume=True,
            reset_failure_guard=request.context.get("reset_failure_guard") is True,
            retry_failed_evaluations=request.context.get("retry_failed_evaluations") is True,
        )
        if scheduled.study.status != "completed":
            error = "; ".join(scheduled.errors) or "resumed HPO workflow failed"
            tracker.update_hpo_experiment(experiment_id, status="failed", error=error)
            if scheduled.study.scheduler_state.get("resource_budget", {}).get("stop_reason"):
                raise ResourceBudgetExhausted(error)
            guard = scheduled.study.scheduler_state.get("failure_guard") or {}
            if guard.get("status") == "open":
                raise SystemFailureBlocked(error, guard)
            raise RuntimeError(error)

        best_trial = service.load_trial(
            experiment_id,
            str(scheduled.study.best_trial_id),
        )
        objective = scheduled.study.objectives[0]
        best_value = float(best_trial.metrics[objective.metric])
        summary = next(
            (
                item for item in campaign.study_summaries
                if item.get("study_id") == scheduled.study.study_id
            ),
            None,
        )
        if summary is None:
            CampaignPolicy().record_study(
                campaign,
                experiment_id=experiment_id,
                study_id=scheduled.study.study_id,
                best_value=best_value,
                training_runs=service.training_runs_used(scheduled.study),
                confirmation_signature=study_confirmation_signature(scheduled.study),
            )
            summary = campaign.study_summaries[-1]
        summary.update({
            "best_parameters": dict(best_trial.parameters),
            "best_metrics": self._best_metric_record(best_trial, objective),
            "strategy_reviews": list(scheduled.study.strategy_reviews),
            "sampler": (
                scheduled.study.candidate_strategy
                or scheduled.study.sampler_strategy
            ),
            "pruner": scheduled.study.pruner_strategy,
            "learning_summary": self._study_learning_summary(
                scheduled.study,
                best_trial,
                objective,
            ),
            "next_study_proposal": scheduled.study.next_study_proposal,
        })
        continuation_pending = CampaignPolicy().should_continue(campaign)
        if continuation_pending:
            campaign.status = "running"
            campaign.stop_reason = None
            campaign.updated_at = datetime.now().isoformat()
        campaign_experiment_ids = list(dict.fromkeys(
            [
                str(item.get("experiment_id"))
                for item in campaign.study_summaries
                if item.get("experiment_id")
            ]
            + [experiment_id]
        ))
        result_best_experiment_id = str(
            campaign.best_experiment_id or experiment_id
        )
        result_best_trial = best_trial
        if result_best_experiment_id != experiment_id:
            best_study = service.load_study(result_best_experiment_id)
            result_best_trial = service.load_trial(
                result_best_experiment_id,
                str(best_study.best_trial_id),
            )
        duration = (datetime.now() - started_at).total_seconds()
        tracker.update_hpo_experiment(
            experiment_id,
            status="success",
            error=None,
            duration=duration,
            parameters=best_trial.parameters,
            metrics={"best": self._best_metric_record(best_trial, objective)},
            extensions={"optimization": {
                "campaign": campaign.to_dict(),
                "resume": scheduled.advice.get("resume") or {},
                "latest_trial": {
                    "trial_id": best_trial.trial_id,
                    "phase": "completed",
                    "status": best_trial.status,
                    "updated_at": best_trial.updated_at,
                },
            }},
        )
        resumed_handoff = {
            **dict(data_handoff),
            "consumer_uri": data_folder,
        }
        self.memory_service.remember_episode(EpisodeMemory(
            agent_type="hpo_agent",
            objective=request.objective,
            action={
                "resume_experiment_id": experiment_id,
                "sampler": summary.get("sampler"),
                "pruner": summary.get("pruner"),
                "best_config": result_best_trial.parameters,
            },
            outcome={
                "best_metrics": result_best_trial.metrics,
                "resume": scheduled.advice.get("resume") or {},
            },
            summary=f"resumed Study {scheduled.study.study_id} to completion",
            experiment_ids=campaign_experiment_ids,
            scope=self.memory_scope,
            importance=0.9,
        ))
        return AgentTaskResult(
            status="success",
            summary={
                "resumed": True,
                "resume_experiment_id": experiment_id,
                "study_id": scheduled.study.study_id,
                "best_experiment_id": result_best_experiment_id,
                "best_trial_id": result_best_trial.trial_id,
                "best_parameters": result_best_trial.parameters,
                "sampler": summary.get("sampler"),
                "pruner": summary.get("pruner"),
                "campaign": campaign.to_dict(),
                "campaign_continuation_pending": continuation_pending,
                "resume_record": scheduled.advice.get("resume") or {},
                "data_handoff": resumed_handoff,
            },
            metrics=self._best_metric_record(result_best_trial, objective),
            artifacts=result_best_trial.artifacts,
            recommendations=(
                [{
                    "action": "continue_campaign",
                    "reason": "resumed Study completed but Campaign has remaining Studies",
                }]
                if continuation_pending else []
            ),
            experiment_ids={
                "hpo": result_best_experiment_id,
                "campaign": campaign_experiment_ids,
            },
            request_id=request.request_id,
        )

    _campaign_from_dict = staticmethod(reporting.campaign_from_dict)

    _merge_campaign_history = staticmethod(reporting.merge_campaign_history)

    _study_learning_summary = staticmethod(reporting.study_learning_summary)

    _cross_study_memory = staticmethod(reporting.cross_study_memory)

    @classmethod
    def _next_study_proposal(
        cls,
        campaign: Dict[str, Any],
        controller_mode: str,
    ) -> Optional[StrategyProposal]:
        """Load the latest deferred proposal for deterministic next-Study review.

        Fixed mode deliberately ignores policy carry-over. Stored audit IDs are
        preserved because this is trusted state produced by the service layer,
        not raw model output.
        """
        if controller_mode not in {"rule", "llm"}:
            return None
        summaries = list(campaign.get("study_summaries") or [])
        if not summaries:
            return None
        latest = summaries[-1]
        raw = latest.get("next_study_proposal")
        if not raw:
            raw = (
                ((latest.get("learning_summary") or {}).get(
                    "next_study_recommendation"
                ) or {}).get("deferred_proposal")
            )
        if not isinstance(raw, dict):
            return None
        try:
            proposal = StrategyProposal.from_dict(
                raw, preserve_audit_fields=True
            )
        except (TypeError, ValueError):
            return None
        return proposal if proposal.action in cls.proposal_actions else None

    _best_metric_record = staticmethod(reporting.best_metric_record)

    _split_trial_metrics = staticmethod(reporting.split_trial_metrics)

    def _trial_executor(
        self,
        experiment_id: str,
        data_folder: str,
        runtime_options: Any = None,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> TrialExecutor:
        context = dict(execution_context or {})
        return TrialExecutor(
            experiment_id=experiment_id,
            data_folder=data_folder,
            experiments_dir=self.experiments_dir,
            runtime_options=dict(runtime_options or {}) if isinstance(runtime_options, dict) else {},
            task_type=str(context.get("task_type") or self.task_type),
            model_family=str(context.get("model_family") or self.model_family),
            implementation=str(context.get("implementation") or self.implementation),
            runner=str(context.get("runner") or self.runner),
            resource_budget=getattr(self, "_resource_budget", None),
        )

    def _planning_proposal(
        self,
        request: AgentTaskRequest,
        search_space: SearchSpace,
        budgets: List[TrialBudget],
        selected_sampler: str,
        selected_sampler_config: Dict[str, Any],
        selected_pruner: str,
        hard_max_training_runs: int,
        initial_trial_count: int,
        promotion_limits: List[int],
        reduction_factor: int,
        campaign: Optional[Dict[str, Any]] = None,
        confirmation_signature: Optional[Dict[str, Any]] = None,
    ) -> StrategyProposal:
        memory_context = self.memory_service.format_context(MemoryQuery(
            agent_type="hpo_agent",
            task_type=self.task_type,
            model_family=self.model_family,
            dataset_key=self.memory_scope.dataset_key,
            tags=[METRIC_PROTOCOL_ID],
            limit=5,
        ))
        prompt = self._strategy_proposal_prompt({
            "phase": "study_planning",
            "objective": request.objective,
            "task_type": self.task_type,
            "model_family": self.model_family,
            "selected_strategy": (
                "successive_halving"
                if selected_pruner == "successive_halving"
                else selected_sampler
            ),
            "selected_sampler": selected_sampler,
            "selected_pruner": selected_pruner,
            "controller_mode": request.context.get("controller_mode"),
            "available_samplers": HPOService.available_samplers(),
            "available_pruners": HPOService.available_pruners(),
            "requested_strategy": request.context.get("strategy", "auto"),
            "requested_sampler": request.context.get("sampler"),
            "selected_sampler_config": dict(selected_sampler_config),
            "requested_pruner": request.context.get("pruner"),
            "primary_metric": request.context.get("primary_metric", "eer"),
            "metric_mode": request.context.get("metric_mode", "min"),
            "metric_protocol": METRIC_PROTOCOL_ID,
            "metric_units": dict(METRIC_UNITS),
            "hard_max_training_runs": hard_max_training_runs,
            "allocation": {
                "initial_trial_count": initial_trial_count,
                "promotion_limits": promotion_limits,
                "reduction_factor": reduction_factor,
            },
            "resource_profile": self._resource_snapshot(
                request.context.get("runtime_options"),
                request.context.get("resource_profile"),
            ),
            "search_budget_analysis": self._search_budget_analysis(
                search_space,
                budgets,
                hard_max_training_runs,
                initial_trial_count,
                promotion_limits,
                reduction_factor,
                campaign,
            ),
            "search_space": search_space.to_dict(),
            "budgets": [item.to_dict() for item in budgets],
            "campaign": self._compact_campaign(campaign or {}),
            "cross_study_memory": self._cross_study_memory(campaign or {}),
            "reference_profile": self._reference_search_profile(self.model_family),
            "historical_memory": memory_context,
        })
        return self._request_strategy_proposal(
            prompt,
            objective_metric=str(request.context.get("primary_metric", "eer")),
            objective_mode=str(request.context.get("metric_mode", "min")),
            confirmation_signature=confirmation_signature,
            error_code="proposal_parse_error",
        )

    def _runtime_strategy_reviewer(self, request: AgentTaskRequest, campaign: OptimizationCampaign):
        if not self.enable_llm_advisor:
            return None

        def review(study: Any, feedback: Dict[str, Any]) -> StrategyProposal:
            memory_context = self.memory_service.format_context(MemoryQuery(
                agent_type="hpo_agent",
                task_type=self.task_type,
                model_family=self.model_family,
                dataset_key=self.memory_scope.dataset_key,
                tags=[METRIC_PROTOCOL_ID],
                limit=5,
            ))
            prompt = self._strategy_proposal_prompt({
                "phase": "runtime_review",
                "objective": request.objective,
                "task_type": self.task_type,
                "model_family": self.model_family,
                "study": self._compact_study(study),
                "primary_metric": study.objectives[0].metric,
                "metric_mode": study.objectives[0].mode,
                "metric_protocol": study.history_context.get(
                    "metric_protocol", METRIC_PROTOCOL_ID
                ),
                "metric_units": study.history_context.get(
                    "metric_units", dict(METRIC_UNITS)
                ),
                "available_samplers": HPOService.available_samplers(),
                "available_pruners": HPOService.available_pruners(),
                "runtime_mutable_fields": [
                    "requested_sampler",
                    "sampler_config",
                    "search_space",
                    "hypotheses",
                    "candidate_proposals",
                ],
                "feedback": feedback,
                "resource_profile": self._resource_snapshot(
                    request.context.get("runtime_options"),
                    request.context.get("resource_profile"),
                ),
                "search_budget_analysis": self._search_budget_analysis(
                    study.search_space,
                    list(study.budgets),
                    int(study.max_training_runs or 1),
                    int(study.initial_trial_count or 1),
                    list(study.promotion_limits or []),
                    int(study.reduction_factor or 3),
                    campaign.to_dict(),
                ),
                "campaign": self._compact_campaign(campaign.to_dict()),
                "cross_study_memory": self._cross_study_memory(campaign.to_dict()),
                "reference_profile": self._reference_search_profile(self.model_family),
                "historical_memory": memory_context,
            })
            return self._request_strategy_proposal(
                prompt,
                objective_metric=study.objectives[0].metric,
                objective_mode=study.objectives[0].mode,
                confirmation_signature=study_confirmation_signature(study),
                error_code="runtime_proposal_parse_error",
            )
        return review

    def _request_strategy_proposal(self, prompt: str, *, error_code: str, **options: Any) -> StrategyProposal:
        with self._advisory_session():
            failures = []
            original_prompt = prompt
            self._advisor_format_events = []
            try:
                context = json.loads(original_prompt).get("context", {})
            except (ValueError, AttributeError):
                context = {}
            self._advisor_evidence_trial_ids.update(trial_ids_from_evidence(context))
            for attempt in range(2):
                response = None
                stage = "request"
                try:
                    response = self._invoke_strategy_model(prompt, **options)
                    stage = "json_decode"
                    value = decode_json_response(response)
                    stage = "schema_validation"
                    proposal = StrategyProposal.from_dict(value, strict_llm=True)
                    for hypothesis in proposal.hypotheses:
                        references = hypothesis.get("evidence_trial_ids", [])
                        if not isinstance(references, list) or any(
                            not isinstance(ref, str) or ref not in self._advisor_evidence_trial_ids
                            for ref in references
                        ):
                            raise ValueError("hypothesis references unobserved Trial IDs")
                    self._advisor_format_events.append({"attempt": attempt + 1, "status": "success"})
                    return proposal
                except ResourceBudgetExhausted:
                    raise
                except Exception as exc:
                    diagnostics = response_diagnostics(response, exc, stage=stage)
                    failures.append(diagnostics)
                    self._advisor_format_events.append({"attempt": attempt + 1, "status": "failed", **diagnostics})
                    if stage == "request" or attempt == 1 or self._advisory_remaining() <= 1:
                        return StrategyProposal(
                            action="invalid_proposal", reason_codes=[error_code],
                            evidence={**diagnostics, "response_attempts": failures},
                        )
                    correction = {
                        "instruction": "Regenerate one complete raw JSON object matching the original schema. "
                            "No prose, fences, comments or extra punctuation. Use only original evidence and exact Trial IDs. "
                            "Do not change budgets or invent missing evidence. This is the final format retry.",
                        "failure_stage": stage,
                        "validation_error": diagnostics["error"],
                    }
                    try:
                        payload = json.loads(original_prompt)
                    except (TypeError, ValueError):
                        payload = {"original_request": original_prompt}
                    if not isinstance(payload, dict):
                        payload = {"original_request": payload}
                    payload["response_correction"] = correction
                    prompt = json.dumps(payload, ensure_ascii=False)

    def _invoke_strategy_model(
        self,
        prompt: str,
        *,
        objective_metric: str = "eer",
        objective_mode: str = "min",
        confirmation_signature: Optional[Dict[str, Any]] = None,
    ) -> Any:
        self._check_resource_budget()
        resources = getattr(self, "_resource_budget", None)
        if resources:
            payload = json.loads(prompt)
            context = payload.setdefault("context", {})
            snapshot = resources.snapshot()
            context["resource_budget"] = snapshot
            if isinstance(context.get("campaign"), dict):
                context["campaign"]["resource_budget"] = snapshot
            prompt = json.dumps(payload, ensure_ascii=False)
        from agent.tools.hpo_analysis_tools import build_hpo_analysis_tools

        adapter = get_model_adapter(self.model_family)
        validator = getattr(adapter, "validate_parameters", None)
        tools = build_hpo_analysis_tools(
            ExperimentTracker(self.experiments_dir),
            parameter_validator=validator if callable(validator) else None,
            objective_metric=objective_metric,
            objective_mode=objective_mode,
            confirmation_signature=confirmation_signature,
        )
        return self._invoke_with_readonly_tools(prompt, tools, max_tool_rounds=2)

    @staticmethod
    def _default_initial_trial_count(strategy: str, budgets: List[TrialBudget], run_limit: int) -> int:
        if strategy != "successive_halving" or len(budgets) <= 1:
            return min(3, max(int(run_limit), 1))
        reduction_factor = 3
        # Start from one complete 3x halving bracket: three rungs -> 9 + 3 + 1.
        baseline = reduction_factor ** max(len(budgets) - 1, 0)
        target = min(max(int(run_limit), 1), baseline)
        for count in range(target, 0, -1):
            planned_runs = count + sum(
                HPOAgent._default_promotion_limits(
                    count,
                    budgets,
                    reduction_factor,
                )
            )
            if planned_runs <= run_limit:
                return count
        return 1

    @staticmethod
    def _default_promotion_limits(
        initial_count: int,
        budgets: List[TrialBudget],
        reduction_factor: int = 3,
    ) -> List[int]:
        limits: List[int] = []
        current = max(int(initial_count), 0)
        for _ in range(max(len(budgets) - 1, 0)):
            current = max(1, (current + reduction_factor - 1) // reduction_factor)
            limits.append(current)
        return limits

    _resource_snapshot = staticmethod(resource_analysis.resource_snapshot)

    _search_budget_analysis = staticmethod(resource_analysis.search_budget_analysis)
    @staticmethod
    def _strategy_proposal_prompt(context: Dict[str, Any]) -> str:
        return render_prompt("hpo_strategy_proposal", context=context)

    _compact_campaign = staticmethod(reporting.compact_campaign)

    _compact_study = staticmethod(reporting.compact_study)

    def _confirmation_signature_from_inputs(
        self,
        *,
        objective: Objective,
        confirmation_budget: TrialBudget,
        data_folder: str,
        data_handoff: Dict[str, Any],
        runtime_options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the exact protocol filter used before the next Study exists."""
        normalized_budget = budget_key(confirmation_budget.to_dict())
        return {
            "objective": objective.to_dict(),
            "final_budget": {
                "epochs": normalized_budget[0],
                "data_fraction": normalized_budget[1],
                "max_duration_seconds": normalized_budget[2],
            },
            "metric_protocol": METRIC_PROTOCOL_ID,
            "metric_units": dict(METRIC_UNITS),
            "dataset": data_folder,
            "dataset_id": data_handoff.get("dataset_id"),
            "dataset_version": data_handoff.get("dataset_version"),
            "task_type": self.task_type,
            "model_family": self.model_family,
            "implementation": self.implementation,
            "runner": self.runner,
            "config_sha256": file_sha256(Path(self.config_path)),
            "verification_config_sha256": runtime_options.get(
                "verification_config_sha256"
            ),
            "validation_pairs_sha256": runtime_options.get(
                "validation_pairs_sha256"
            ),
            "training_exclusion_pairs_sha256": runtime_options.get(
                "training_exclusion_pairs_sha256"
            ),
            "training_runtime": self._training_runtime_signature(runtime_options),
        }

    @staticmethod
    def _training_runtime_options(runtime_options: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: runtime_options.get(key)
            for key in (
                "device",
                "ddp_devices",
                "distributed_world_size",
                "distributed_backend",
                "batch_size_semantics",
                "ddp_plan",
                "precision",
                "eval_precision",
            )
            if runtime_options.get(key) is not None
        }

    @staticmethod
    def _training_runtime_signature(runtime_options: Dict[str, Any]) -> Dict[str, Any]:
        from agent.runners.speechbrain_distributed import training_runtime_signature

        return training_runtime_signature(runtime_options)

    @staticmethod
    def _reference_search_profile(model_family: str) -> Dict[str, Any]:
        policy = {
            "max_changed_parameters_per_review": 2,
            "preferred_strategy_progression": ["successive_halving", "adaptive_search", "tpe"],
        }
        if model_family not in {"ecapa_tdnn", "resnet"}:
            return {
                "baseline_parameters": {},
                "stable_search_space": None,
                "local_adjustment_policy": policy,
            }
        policy.update({
            "lr_boundary_factor": 2.0,
            "weight_decay_boundary_factor": 3.0,
            "margin_step": 0.05,
            "resource_first_on_oom": ["batch_size"] + (
                ["sentence_len"] if model_family == "resnet" else []
            ),
        })
        return {
            "baseline_parameters": {
                "lr": 0.001, "batch_size": 32, "margin": 0.2, "weight_decay": 2e-6,
            },
            # The model adapter owns these bounds; do not maintain a second copy.
            "stable_search_space": get_model_adapter(model_family).default_search_space(),
            "local_adjustment_policy": policy,
            "rationale": (
                f"SpeechBrain {model_family} tuning should stay near the stable recipe "
                "and make local evidence-backed moves before broad exploration."
            ),
        }

    def _resolve_search_space(self, value: Optional[Dict[str, Any]]) -> SearchSpace:
        return self._build_search_space(value or self._default_model_search_space())

    def _default_model_search_space(self) -> Dict[str, Any]:
        adapter = get_model_adapter(self.model_family)
        factory = getattr(adapter, "default_search_space", None)
        if not callable(factory):
            raise ValueError(
                f"model adapter '{self.model_family}' does not declare a default search space; "
                "provide context.search_space explicitly or implement default_search_space()"
            )
        value = factory()
        if not isinstance(value, dict) or not value.get("parameters"):
            raise ValueError(f"model adapter '{self.model_family}' returned an empty default search space")
        return value

    @staticmethod
    def _build_search_space(
        value: Optional[Dict[str, Any]],
        default: Optional[Dict[str, Any]] = None,
    ) -> SearchSpace:
        selected = value or default
        if selected:
            return SearchSpace(
                [SearchParameter(**item) for item in selected.get("parameters") or []],
                selected.get("constraints") or [],
            )
        raise ValueError("search space must contain at least one parameter")

    @staticmethod
    def _validate_search_budget_compatibility(
        search_space: SearchSpace,
        budgets: List[TrialBudget],
    ) -> None:
        """Reject parameters whose values would be silently replaced by a budget."""
        parameter_names = {parameter.name for parameter in search_space.parameters}
        if "number_of_epochs" in parameter_names and any(
            budget.epochs is not None for budget in budgets
        ):
            raise ValueError(
                "search parameter 'number_of_epochs' conflicts with budget.epochs; "
                "remove it from the search space or set every budget epochs value to null"
            )

    @staticmethod
    def _build_budgets(
        value: Optional[List[Dict[str, Any]]],
        requested_strategy: str = "auto",
    ) -> List[TrialBudget]:
        if value:
            return [TrialBudget(**item) for item in value]
        if requested_strategy != "successive_halving":
            return [TrialBudget("full", epochs=20, data_fraction=1.0)]
        return [
            TrialBudget("screening", epochs=3, data_fraction=0.25),
            TrialBudget("promotion", epochs=8, data_fraction=0.5),
            TrialBudget("confirmation", epochs=20, data_fraction=1.0),
        ]

    def get_execution_details(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "workflow": "langgraph",
            "available_samplers": HPOService.available_samplers(),
            "available_pruners": HPOService.available_pruners(),
            "llm_role": "search_method_controller_with_optional_agent_proposal_sampler",
            "decision_authority": "HPOService + OptimizationPlanDecisionPolicy",
            "feedback_loop": "trial/rung reviews + historical memory + optimization campaign",
            "enable_llm_advisor": self.enable_llm_advisor,
        }


def create_hpo_agent(**kwargs: Any) -> HPOAgent:
    return HPOAgent(**kwargs)


__all__ = ["HPOAgent", "create_hpo_agent"]
