"""Project entry point for the HPO agent."""

from __future__ import annotations

import argparse

from agent.agents.react_agent import create_react_agent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SR-agent HPO entry point")
    parser.add_argument("--target-eer", type=float, default=0.02)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--model-name", type=str, default="GLM-4.7")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    agent = create_react_agent(
        model_name=args.model_name,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
    )

    result = agent.optimize_hyperparameters(
        target_eer=args.target_eer,
        custom_objective=args.objective,
    )

    print(result.final_answer)


if __name__ == "__main__":
    main()
