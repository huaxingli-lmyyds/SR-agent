"""
SpeechBrain ECAPA-TDNN training pipeline wrapper.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List

from agent.utils.runner import run_training


def train_pipeline(
    config_path: str,
    overrides: Optional[List[str]] = None,
    run_opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    """Run training without subprocess and return the validation EER."""
    result = run_training(config_path, overrides or [])
    return {
        "eer": result.get("eer"),
        "status": result.get("status"),
        "error": result.get("error"),
    }
