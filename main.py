"""Project entry point for the orchestration agent."""

from __future__ import annotations

import argparse

from agent.agents.orchestrator import CoordinatorAgent
from agent.utils.path_tool import get_config_file


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SR-agent orchestration entry point")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--data-iterations", type=int, default=6)
    parser.add_argument("--model-name", type=str, default="GLM-4.7")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default="Improve the primary model metric")
    parser.add_argument(
        "--strategy",
        choices=["auto", "random_search", "grid_search", "adaptive_search", "successive_halving"],
        default="auto",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config-path", type=str, default=str(get_config_file("train_ecapa_tdnn.yaml")))
    parser.add_argument("--task-type", type=str, default="speaker_verification")
    parser.add_argument("--model-family", type=str, default="ecapa_tdnn")
    parser.add_argument("--implementation", type=str, default="speechbrain")
    parser.add_argument("--runner", type=str, default="speechbrain")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    pipeline = CoordinatorAgent(
        model_name=args.model_name,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        data_iterations=args.data_iterations,
        verbose=args.verbose,
        config_path=args.config_path,
        task_type=args.task_type,
        model_family=args.model_family,
        implementation=args.implementation,
        runner=args.runner,
    )
    result = pipeline.run(
        objective=args.objective,
        context={"strategy": args.strategy},
        budget={"max_training_runs": args.max_iterations},
    )
    print(result.experiment_id)


if __name__ == "__main__":
    main()
