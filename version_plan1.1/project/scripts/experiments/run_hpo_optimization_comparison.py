"""Three isolated protocol arms; preview by default, never invent benchmark results."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def plans(request):
    from agent.agents.hpo_preview import preview_plan
    arms = {}
    for name in ("legacy", "p0", "full"):
        arm = deepcopy(request)
        context = arm.setdefault("context", {})
        runtime = context.setdefault("runtime_options", {})
        runtime.update(managed_execution=False, max_concurrent_trials=1, gpu_pool=None,
                       promotion_training="restart", confirmation_enabled=False,
                       decision_opportunities=False, checkpoint_selection="legacy")
        if name != "legacy":
            runtime.update(decision_opportunities=True, checkpoint_selection="validation_topk")
        if name == "full":
            runtime.update(managed_execution=True, promotion_training="resume_if_compatible",
                           confirmation_enabled=True)
            runtime.update(request.get("full_execution_options", {}))
        arms[name] = {"request": arm, "preview": preview_plan(context, arm.get("budget", {}))}
    return arms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path,
                        help="JSON: config_path, context, budget; optional full_execution_options")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    arms = plans(request)
    if not args.execute:
        print(json.dumps(arms, ensure_ascii=False, indent=2))
        return 0
    from agent.hpo.protocol import file_sha256
    from agent.agents.hpo_agent import HPOAgent
    from agent.agents.communication import AgentTaskRequest
    from agent.memory import MemoryService
    if not request.get("budget", {}).get("max_gpu_hours"):
        parser.error("execution requires one explicit max_gpu_hours limit shared by all arms")
    context = request["context"]
    runtime = context["runtime_options"]
    frozen = {}
    for value in (request["config_path"], runtime["verification_config"], runtime["validation_pairs"],
                  runtime.get("training_exclusion_pairs", runtime["validation_pairs"])):
        path = Path(value).resolve()
        if not path.is_file():
            parser.error(f"missing frozen input: {path}")
        frozen[str(path)] = file_sha256(path)
    if not Path(context["data_folder"]).is_dir():
        parser.error("data_folder must exist")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps({"arms": arms, "input_hashes": frozen}, indent=2), encoding="utf-8")
    for name, arm in arms.items():
        if any(file_sha256(Path(path)) != digest for path, digest in frozen.items()):
            raise ValueError("frozen comparison input changed")
        folder = args.output / name
        agent = HPOAgent(config_path=request["config_path"], experiments_dir=str(folder / "experiments"),
                         memory_service=MemoryService(folder / "memory"),
                         enable_llm_advisor=request.get("enable_llm_advisor", False))
        result = agent.run_workflow(AgentTaskRequest(action=agent.action, objective=f"HPO comparison: {name}",
            context=arm["request"]["context"], budget=arm["request"]["budget"]))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
