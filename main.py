"""Project entry point for the orchestration agent."""

from __future__ import annotations

import argparse

from agent.agents.orchestrator import OrchestratedPipeline
from agent.utils.path_tool import get_config_file


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SR-agent orchestration entry point")
    parser.add_argument("--target-eer", type=float, default=0.02)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--data-iterations", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--model-name", type=str, default="GLM-4.7")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config-path", type=str, default=str(get_config_file("train_ecapa_tdnn.yaml")))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    pipeline = OrchestratedPipeline(
        model_name=args.model_name,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        data_iterations=args.data_iterations,
        max_rounds=args.rounds,
        verbose=args.verbose,
        config_path=args.config_path,
    )
    result = pipeline.run(
        target_eer=args.target_eer,
        custom_objective=args.objective,
    )
    print(result.experiment_id)


if __name__ == "__main__":
    main()
