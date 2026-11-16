"""Backward-compatible lazy calls for the relocated SpeechBrain backend."""

from typing import Any


def run_training(*args: Any, **kwargs: Any):
    from agent.runners.speechbrain_backend import run_training as implementation

    return implementation(*args, **kwargs)


def run_data_prep(*args: Any, **kwargs: Any):
    from agent.runners.speechbrain_backend import run_data_prep as implementation

    return implementation(*args, **kwargs)


def run_evaluation(*args: Any, **kwargs: Any):
    from agent.runners.speechbrain_backend import run_evaluation as implementation

    return implementation(*args, **kwargs)

__all__ = ["run_training", "run_data_prep", "run_evaluation"]
