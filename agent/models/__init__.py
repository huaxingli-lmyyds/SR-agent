"""Model configuration adapters."""

from .contracts import ModelAdapter
from .ecapa_tdnn import SpeechBrainEcapaAdapter
from .registry import MODEL_ADAPTERS, get_model_adapter, register_model_adapter


if "ecapa_tdnn" not in MODEL_ADAPTERS:
    register_model_adapter(SpeechBrainEcapaAdapter())


__all__ = [
    "MODEL_ADAPTERS",
    "ModelAdapter",
    "SpeechBrainEcapaAdapter",
    "get_model_adapter",
    "register_model_adapter",
]
