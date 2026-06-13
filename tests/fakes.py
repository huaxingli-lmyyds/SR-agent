from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from agent.agents.communication import AgentTaskRequest, AgentTaskResult
from agent.core.contracts import Artifact, OperationResult


@dataclass
class FakeRunnerAdapter:
    runner: str = "fake"

    def run_training(self, config_path: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "success",
            "metrics": {"valid_error_rate": 0.08},
            "model_paths": [str(Path(overrides["output_folder"]) / "fake.ckpt")],
            "output_folder": overrides["output_folder"],
        }

    def run_evaluation(
        self,
        config_path: str,
        model_path: str | None,
        data_path: str | None,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "status": "success",
            "metrics": {"eer": 0.03, "min_dcf": 0.12},
            "output_folder": overrides["output_folder"],
        }

    def normalize_training_result(self, raw: Dict[str, Any]) -> OperationResult:
        return OperationResult(
            status=raw["status"],
            stage="training",
            metrics={"validation": raw.get("metrics") or {}},
            artifacts=[
                Artifact("checkpoint", Path(path).name, path)
                for path in raw.get("model_paths") or []
            ],
        )

    def normalize_evaluation_result(self, raw: Dict[str, Any]) -> OperationResult:
        return OperationResult(
            status=raw["status"],
            stage="evaluation",
            metrics={"test": raw.get("metrics") or {}},
            error=raw.get("error"),
        )


class FakeAgent:
    def execute_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        return AgentTaskResult(
            status="success",
            summary={"action": request.action},
            request_id=request.request_id,
        )


class FailingAgent:
    def execute_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        raise RuntimeError("fake failure")
