"""torchrun worker for one SpeechBrain training Trial."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def _write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request_file).read_text(encoding="utf-8"))
    plan = dict(request["plan"])
    overrides = dict(request["overrides"])
    run_opts = dict(overrides.get("_run_opts") or {})
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    global_rank = int(os.environ["RANK"])
    run_opts.update({
        "device": f"cuda:{local_rank}",
        "distributed_launch": True,
        "distributed_backend": plan["backend"],
        "_sr_world_size": int(plan["world_size"]),
        "_sr_devices": list(plan["devices"]),
        "_sr_batch_size_semantics": plan["batch_size_semantics"],
    })
    for key in ("ddp_devices", "distributed_world_size", "batch_size_semantics"):
        run_opts.pop(key, None)
    overrides["_run_opts"] = run_opts

    from .speechbrain_backend import run_training

    result = run_training(str(request["config_path"]), overrides)
    if global_rank == 0:
        _write_json_atomic(Path(request["result_path"]), result)
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
