"""Registry for model configuration adapters."""

from typing import Dict

from .contracts import ModelAdapter


MODEL_ADAPTERS: Dict[str, ModelAdapter] = {}


def register_model_adapter(adapter: ModelAdapter) -> None:
    MODEL_ADAPTERS[adapter.model_family] = adapter


def get_model_adapter(model_family: str) -> ModelAdapter:
    try:
        return MODEL_ADAPTERS[model_family]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_ADAPTERS)) or "none"
        raise ValueError(f"unknown model adapter '{model_family}'; available: {available}") from exc


__all__ = ["MODEL_ADAPTERS", "get_model_adapter", "register_model_adapter"]
