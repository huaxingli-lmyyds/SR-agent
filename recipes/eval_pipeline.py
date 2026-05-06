"""
SpeechBrain ECAPA-TDNN evaluation pipeline wrapper.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List

from agent.utils import runner


def eval_pipeline(
    config_path: str,
    model_path: Optional[str] = None,
    data_folder: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    run_opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run evaluation without subprocess and return metrics and outputs."""
    return runner.run_evaluation(
        config_path=config_path,
        model_path=model_path,
        data_folder=data_folder,
        overrides=overrides,
        run_opts=run_opts,
    )
