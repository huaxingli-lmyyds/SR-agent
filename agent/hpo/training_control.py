"""HPO-owned epoch decisions, serialized so training workers need no agent runtime."""

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from statistics import median

from .records import _write_json_atomic


@dataclass(frozen=True)
class TrainingControlOptions:
    metric: str = "valid_loss"
    min_epochs: int = 3
    patience: int = 3
    min_delta: float = 0.001
    peer_min_trials: int = 3
    peer_patience: int = 2
    peer_factor: float = 2.0

    def __post_init__(self):
        if self.metric not in {"valid_loss", "valid_error_rate"}:
            raise ValueError(
                "training control observes internal validation loss/error only"
            )
        for key in (
            "min_epochs",
            "patience",
            "peer_min_trials",
            "peer_patience",
        ):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 10000
            ):
                raise ValueError(
                    f"training control {key} must be a positive integer <= 10000"
                )
        for key in ("min_delta", "peer_factor"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"training control {key} must be finite")
        if self.min_delta < 0 or self.peer_factor <= 1:
            raise ValueError(
                "min_delta must be nonnegative and peer_factor greater than one"
            )

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or value.keys() - cls.__dataclass_fields__.keys()
        ):
            raise ValueError("invalid training control options")
        return cls(**value)


def build_trial_control(service, study, trial, options):
    """Freeze comparable peer curves and the overall objective before dispatch."""
    policy = TrainingControlOptions.from_dict(options)
    persisted = service.load_trial(study.experiment_id, trial.trial_id)
    saved = persisted.cost.get("training_control_spec")
    if saved:
        return saved
    if trial.budget.epochs is None:
        raise ValueError("training control requires an explicit epoch budget")
    from .history import budget_key

    peers = [
        p
        for p in service.list_trials(study.experiment_id)
        if p.trial_id != trial.trial_id
        and p.status in {"completed", "promoted"}
        and budget_key(p.budget.to_dict()) == budget_key(trial.budget.to_dict())
    ]
    by_epoch, evidence = {}, []
    for peer in peers:
        curve = {}
        for row in peer.intermediate_metrics:
            epoch, value = row.get("epoch"), row.get(policy.metric)
            if (
                isinstance(epoch, (int, float))
                and not isinstance(epoch, bool)
                and isfinite(epoch)
                and int(epoch) == epoch
                and epoch >= 1
                and _finite(value)
            ):
                curve[int(epoch)] = float(value)
        if curve:
            evidence.append(peer.trial_id)
            for epoch, value in curve.items():
                by_epoch.setdefault(epoch, []).append(value)
    thresholds = {
        str(epoch): median(values) * policy.peer_factor + policy.min_delta
        for epoch, values in by_epoch.items()
        if len(values) >= policy.peer_min_trials
    }
    specification = {
        "schema_version": 1,
        "policy": asdict(policy),
        "trial_id": trial.trial_id,
        "objective": study.objectives[0].to_dict(),
        "budget": trial.budget.to_dict(),
        "peer_trial_ids": evidence,
        "peer_thresholds": thresholds,
        "metric_role": "internal_dev_proxy_not_primary_objective",
    }
    service.record_trial(
        study,
        trial.trial_id,
        status=persisted.status,
        cost={"training_control_spec": specification},
    )
    return specification


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


class EpochTrainingController:
    """Pure epoch policy plus a replayable local record of its observations."""

    def __init__(self, specification, path=None):
        import json

        if specification.get("schema_version") != 1:
            raise ValueError("unsupported training control specification")
        self.specification = specification
        self.policy = TrainingControlOptions.from_dict(specification["policy"])
        self.path = Path(path) if path else None
        self.history = []
        self.decision = {"should_stop": False, "reason": "awaiting_validation"}
        if self.path and self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved["specification"] != specification:
                raise ValueError(
                    "training control specification changed on recovery"
                )
            self.history, self.decision = saved["history"], saved["decision"]

    def observe(self, epoch, metrics):
        if self.decision["should_stop"]:
            return self.decision
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError("epoch must be a positive integer")
        if self.history and epoch <= self.history[-1]["epoch"]:
            raise ValueError(
                "training control observations must advance epochs"
            )
        row = {
            "epoch": epoch,
            **{
                key: float(value) if _finite(value) else None
                for key, value in metrics.items()
                if key in {"train_loss", "valid_loss", "valid_error_rate"}
            },
        }
        self.history.append(row)
        decision = self._decide(epoch)
        self.decision = {
            **decision,
            "epoch": epoch,
            "metric": self.policy.metric,
            "value": row.get(self.policy.metric),
            "objective": self.specification.get("objective"),
            "peer_trial_ids": self.specification.get("peer_trial_ids", []),
        }
        if self.path:
            _write_json_atomic(self.path, self.to_dict())
        return self.decision

    def _decide(self, epoch):
        policy = self.policy
        value = self.history[-1].get(policy.metric)
        if value is None:
            return {
                "should_stop": True,
                "reason": "missing_or_nonfinite_validation_metric",
            }
        if epoch >= self.specification["budget"]["epochs"]:
            return {"should_stop": False, "reason": "budget_completed"}
        if epoch < policy.min_epochs:
            return {"should_stop": False, "reason": "warmup"}
        best, last_improvement = float("inf"), 0
        for row in self.history:
            value = row.get(policy.metric)
            if value is not None and value < best - policy.min_delta:
                best, last_improvement = value, row["epoch"]
        if (
            last_improvement
            and epoch - max(last_improvement, policy.min_epochs - 1)
            >= policy.patience
        ):
            return {"should_stop": True, "reason": "validation_plateau"}
        recent = self.history[-policy.peer_patience :]
        thresholds = self.specification.get("peer_thresholds", {})
        if len(recent) == policy.peer_patience and all(
            row["epoch"] >= policy.min_epochs
            and row.get(policy.metric) is not None
            and str(row["epoch"]) in thresholds
            and row[policy.metric] > thresholds[str(row["epoch"])]
            for row in recent
        ):
            return {
                "should_stop": True,
                "reason": "uncompetitive_same_budget_peers",
            }
        return {"should_stop": False, "reason": "continue"}

    def to_dict(self):
        return {
            "specification": self.specification,
            "history": self.history,
            "decision": self.decision,
        }


def initialize_training_control(brain):
    spec = getattr(brain.hparams, "hpo_training_control", None)
    if not spec:
        return
    import torch

    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else 0
    )
    path = Path(brain.hparams.output_folder) / "training_control.json"
    controller, snapshot = None, None
    if rank == 0:
        try:
            controller = EpochTrainingController(spec, path)
            snapshot = controller.to_dict()
        except Exception as exc:
            snapshot = {"error": f"{type(exc).__name__}: {exc}"}
    if torch.distributed.is_initialized():
        objects = [snapshot]
        torch.distributed.broadcast_object_list(
            objects, src=0, device=torch.device(brain.device)
        )
        snapshot = objects[0]
    if "error" in snapshot:
        raise RuntimeError(snapshot["error"])
    if controller is None:
        controller = EpochTrainingController(spec)
        controller.history, controller.decision = (
            snapshot["history"],
            snapshot["decision"],
        )
    brain.hpo_training_controller = controller
    # SpeechBrain restores the latest retained checkpoint, which can predate a
    # control observation. Replay only committed epochs after that checkpoint.
    controller = brain.hpo_training_controller
    current = brain.hparams.epoch_counter.current
    if controller.history and current < controller.history[-1]["epoch"]:
        controller.history = [
            row for row in controller.history if row["epoch"] <= current
        ]
        controller.decision = {
            "should_stop": False,
            "reason": "checkpoint_recovery",
        }
    brain.hpo_training_decision = controller.decision
    if controller.decision["should_stop"]:
        brain.hparams.epoch_counter.limit = current


def observe_training_epoch(brain, epoch, metrics):
    """All ranks join the same boundary; only rank zero publishes decisions."""
    controller = getattr(brain, "hpo_training_controller", None)
    if controller is None:
        return
    import torch

    distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    values = torch.tensor(
        [
            metrics[key]
            for key in ("train_loss", "valid_loss", "valid_error_rate")
        ],
        dtype=torch.float64,
        device=brain.device,
    )
    if distributed:
        torch.distributed.all_reduce(values)
        values /= torch.distributed.get_world_size()
    decision = None
    if rank == 0:
        try:
            decision = controller.observe(
                epoch,
                dict(
                    zip(
                        ("train_loss", "valid_loss", "valid_error_rate"),
                        values.tolist(),
                    )
                ),
            )
        except Exception as exc:
            decision = {"error": f"{type(exc).__name__}: {exc}"}
    if distributed:
        objects = [decision]
        torch.distributed.broadcast_object_list(
            objects, src=0, device=values.device
        )
        decision = objects[0]
    if "error" in decision:
        raise RuntimeError(decision["error"])
    brain.hpo_training_decision = decision
    if decision["should_stop"]:
        brain.hparams.epoch_counter.limit = epoch


def stopped_training_result(result):
    """Map a runner's successful early exit to a pruned Trial, never a failure."""
    extensions = result.get("extensions") or {}
    observation = (
        extensions.get("speechbrain")
        or extensions.get("training_control")
        or {}
    )
    control = (observation.get("runtime") or {}).get("training_control") or {}
    if not control.get("should_stop"):
        return None
    return {
        "status": "stopped",
        "stage": "training",
        "stop_reason": control["reason"],
        "intermediate_metrics": observation.get("epoch_data") or [],
        "artifacts": result.get("artifacts") or [],
        "cost": {"training_control": control},
    }
