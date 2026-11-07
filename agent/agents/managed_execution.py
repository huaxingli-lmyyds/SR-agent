"""Controller-side Trial coroutine. All tool preparation and commits stay here."""
import json
from hashlib import sha256
from pathlib import Path

from agent.agents.hpo_execution import TrialExecutor
from agent.hpo.service import HPOService
from agent.utils import ExperimentTracker
from agent.runners.checkpoint_selection import choose


class ManagedTrialExecutor(TrialExecutor):
    managed = True

    def _tool(self, tool, payload, trial, attempt):
        operation = tool.func.prepare_execution(**payload)
        response = None
        while True:
            try:
                request = operation.send(response)
            except StopIteration as done:
                return json.loads(done.value)
            response = yield (request, trial.trial_id, attempt)

    def steps(self, trial, attempt):
        from agent.tools.training_tools import TrainModel
        from agent.tools.evaluation_tools import RunEvaluation
        from agent.hpo.continuation import prepare_continuation
        service = HPOService(ExperimentTracker(self.experiments_dir))
        options = self.runtime_options
        study = service.load_study(self.experiment_id)
        persisted = service.load_trial(self.experiment_id, trial.trial_id)
        common = dict(experiment_id=self.experiment_id, trial_id=trial.trial_id,
            data_folder=self.data_folder, experiments_dir=str(self.experiments_dir),
            task_type=self.task_type, model_family=self.model_family,
            implementation=self.implementation, runner=self.runner,
            device=options.get("device"), precision=options.get("precision"),
            eval_precision=options.get("eval_precision"))
        # Training is already committed before any evaluation is dispatched.
        train = persisted.cost.get("managed_training_result")
        if not train or train.get("status") != "success":
            continuation = prepare_continuation(service, study, trial, options)
            parameters = dict(trial.parameters)
            if options.get("training_seed") is not None:
                parameters["seed"] = options["training_seed"]
            train = yield from self._tool(TrainModel, {**common,
                "attempt_id": f"attempt_{attempt}",
                "parameters_json": json.dumps(parameters), "budget_json": json.dumps(trial.budget.to_dict()),
                "checkpoint_selection": options.get("checkpoint_selection", "legacy"),
                "training_exclusion_pairs": options.get("training_exclusion_pairs"),
                "ddp_devices": options.get("ddp_devices"),
                "distributed_backend": options.get("distributed_backend"),
                "batch_size_semantics": options.get("batch_size_semantics"),
                "continuation_json": json.dumps(continuation),
                "frozen_data_json": json.dumps(options["frozen_data"]) if options.get("frozen_data") else None,
            }, trial, attempt)
            study = service.load_study(self.experiment_id)
            current = service.load_trial(self.experiment_id, trial.trial_id)
            service.record_trial(study, trial.trial_id, status=current.status,
                cost={"managed_training_result": train, "continuation": continuation})
        if train.get("status") != "success":
            return {**train, "stage": "training"}
        observation = (train.get("extensions") or {}).get("speechbrain") or {}
        current = service.load_trial(self.experiment_id, trial.trial_id)
        topk = options.get("checkpoint_selection") == "validation_topk"
        rows = current.cost.get("checkpoint_evaluations") or []
        if not rows:
            rows = observation.get("checkpoint_candidates") or [] if topk else []
            if not topk:
                checkpoint = next((a["path"] for a in train.get("artifacts", []) if a["type"] == "checkpoint"), None)
                if checkpoint:
                    rows = [{"path": checkpoint, "epoch": observation.get("selected_checkpoint", {}).get("checkpoint_epoch", 0),
                             "observation": observation.get("selected_checkpoint", {})}]
        if not rows:
            return {"status": "failed", "stage": "evaluation", "error": "missing checkpoint candidates"}
        cost = {"checkpoint_evaluations": rows, "evaluated_checkpoint": rows[0]["path"],
            "checkpoint_selection": "validation_topk" if topk else "legacy",
            "evaluation": {"status": "failed", "selection_complete": False},
            "training_observation": {"epoch_data": observation.get("epoch_data", []),
                "final_metrics": observation.get("final_metrics", {}),
                "final_learning_rate_schedule": observation.get("runtime", {}).get("final_learning_rate_schedule", {})}}
        service.record_trial(service.load_study(self.experiment_id), trial.trial_id, status="running", cost=cost)
        for row in rows:
            if row.get("status") == "success":
                continue
            result = yield from self._tool(RunEvaluation, {**common,
                "model_path": row["path"], "defer_commit": True,
                "checkpoint_id": sha256((row["path"] + str(attempt)).encode()).hexdigest()[:24],
                "verification_config": options.get("verification_config"),
                "verification_pairs": options.get("validation_pairs"), "evaluation_split": "validation",
            }, trial, attempt)
            metrics = {k:v for values in (result.get("metrics") or {}).values() for k,v in values.items()}
            row.update(status=result.get("status"), metrics=metrics, error=result.get("error"),
                       artifacts=result.get("artifacts", []))
            service.record_trial(service.load_study(self.experiment_id), trial.trial_id, status="running", cost=cost)
            if result.get("status") != "success":
                return {"status": "failed", "stage": "evaluation", "error": result.get("error"), "cost": cost}
        selected = choose(rows, study.objectives[0].metric, study.objectives[0].mode)
        cost["evaluated_checkpoint"] = selected["path"]
        cost["training_observation"]["selected_checkpoint"] = selected.get("observation", {})
        cost["evaluation"] = {"status": "success", "selection_complete": True}
        return {"status": "success", "stage": "evaluation", "metrics": selected["metrics"], "cost": cost,
                "artifacts": [{"type": "checkpoint", "path": selected["path"], "name": Path(selected["path"]).name},
                              *selected.get("artifacts", [])]}
