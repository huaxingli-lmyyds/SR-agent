"""Epoch-boundary, per-rank SpeechBrain recovery using its registered hooks."""
from hashlib import sha256
import inspect
import json
from pathlib import Path
import random

from agent.hpo.records import _write_json_atomic
from agent.hpo.protocol import file_sha256


def capture_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def semantic_signature(config, overrides, hparams, runtime, fraction):
    # Output locations and cumulative epoch limits do not affect compatibility.
    filtered = {k:v for k,v in overrides.items() if k not in {
        "output_folder", "number_of_epochs", "_run_opts", "_hpo_continuation",
        "_hpo_max_duration_seconds", "_hpo_runtime_deadline_epoch"}}
    source = Path(config).read_text(encoding="utf-8")
    source = "\n".join(line for line in source.splitlines() if not line.startswith(("number_of_epochs:", "output_folder:")))
    return {"config": sha256(source.encode()).hexdigest(), "overrides": filtered,
        "train_manifest": file_sha256(Path(hparams["train_annotation"])),
        "valid_manifest": file_sha256(Path(hparams["valid_annotation"])),
        "data_fraction": fraction or 1.0, "world_size": runtime.get("world_size", 1),
        "precision": runtime.get("precision", "fp32"),
        "batch_size_per_device": hparams.get("batch_size_per_device", hparams.get("batch_size")),
        "seed": hparams.get("seed"),
        "batch_size_semantics": runtime.get("batch_size_semantics", "per_device"),
        "backend": runtime.get("backend"),
        "dataloader_options": {k:v for k,v in hparams.get("dataloader_options", {}).items() if isinstance(v, (str, int, float, bool, type(None)))}}


def bind_continuation(hparams, specification, signature):
    specification = dict(specification or {})
    if hparams.get("dataloader_options", {}).get("persistent_workers"):
        specification.update(enabled=False, capture=False, reason="persistent_worker_rng_state_not_supported")
    if specification.get("enabled"):
        manifest = json.loads(Path(specification["manifest"]).read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
            specification.update(enabled=False, reason="prepared_training_semantics_changed")
    hparams["hpo_continuation"] = specification
    hparams["hpo_training_signature"] = signature


def save_state(brain, epoch):
    import torch
    from speechbrain.utils import checkpoints
    spec = getattr(brain.hparams, "hpo_continuation", {})
    if not spec.get("capture"):
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    root = Path(brain.hparams.output_folder) / "continuation" / f"epoch_{epoch}"
    directory = root / f"rank_{rank}"
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in brain.checkpointer.recoverables.items():
        custom = brain.checkpointer.custom_save_hooks.get(name)
        hook = custom or checkpoints.get_default_hook(value, checkpoints.DEFAULT_SAVE_HOOKS)
        if hook is None:
            raise ValueError(f"missing full-state save hook: {name}")
        inspect.unwrap(hook)(value, directory / f"{name}.ckpt")
    torch.save(capture_rng(), directory / "rng.pt")
    torch.save({"optimizer_step": brain.optimizer_step, "step": brain.step,
                "scaler": brain.scaler.state_dict()}, directory / "execution.pt")
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        files = {str(p.relative_to(root)): file_sha256(p) for p in root.glob("rank_*/*") if p.is_file()}
        _write_json_atomic(root / "manifest.json", {"schema_version": 1, "epoch": epoch,
            "world_size": world, "files": files, "signature": brain.hparams.hpo_training_signature})
        _write_json_atomic(Path(brain.hparams.output_folder) / "continuation_latest.json",
                           {"manifest": str(root / "manifest.json")})
    if world > 1:
        torch.distributed.barrier()


def restore_state(brain):
    import torch
    from speechbrain.utils import checkpoints
    spec = getattr(brain.hparams, "hpo_continuation", {})
    if not spec.get("enabled"):
        return
    root = Path(spec["manifest"]).parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    for name, digest in manifest["files"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("full-state content changed after continuation validation")
    directory = root / f"rank_{rank}"
    for name, value in brain.checkpointer.recoverables.items():
        hook = brain.checkpointer.custom_load_hooks.get(name) or checkpoints.get_default_hook(value, checkpoints.DEFAULT_LOAD_HOOKS)
        if hook is None or not (directory / f"{name}.ckpt").is_file():
            raise ValueError(f"incomplete full-state recovery: {name}")
        inspect.unwrap(hook)(value, directory / f"{name}.ckpt", True)
    execution = torch.load(directory / "execution.pt", map_location="cpu", weights_only=False)
    brain.optimizer_step = execution["optimizer_step"]; brain.step = execution["step"]
    brain.scaler.load_state_dict(execution["scaler"])
    restore_rng(torch.load(directory / "rng.pt", map_location="cpu", weights_only=False))
