"""Model configuration adapters."""

from .contracts import ModelAdapter
from .ecapa_tdnn import SpeechBrainEcapaAdapter
from .registry import MODEL_ADAPTERS, get_model_adapter, register_model_adapter
from .speechbrain_speaker import SpeechBrainResNetAdapter, SpeechBrainXVectorAdapter


if "ecapa_tdnn" not in MODEL_ADAPTERS:
    register_model_adapter(SpeechBrainEcapaAdapter())
if "resnet" not in MODEL_ADAPTERS:
    register_model_adapter(SpeechBrainResNetAdapter())
if "xvector" not in MODEL_ADAPTERS:
    register_model_adapter(SpeechBrainXVectorAdapter())


__all__ = [
    "MODEL_ADAPTERS",
    "ModelAdapter",
    "SpeechBrainEcapaAdapter",
    "SpeechBrainResNetAdapter",
    "SpeechBrainXVectorAdapter",
    "get_model_adapter",
    "register_model_adapter",
]