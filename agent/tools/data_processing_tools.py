"""Backward-compatible lazy export for SpeechBrain/VoxCeleb data tools."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .speechbrain_data_tools import PrepareVoxCelebData

__all__ = ["PrepareVoxCelebData"]


def __getattr__(name: str):
    if name != "PrepareVoxCelebData":
        raise AttributeError(name)
    from .speechbrain_data_tools import PrepareVoxCelebData

    globals()[name] = PrepareVoxCelebData
    return PrepareVoxCelebData