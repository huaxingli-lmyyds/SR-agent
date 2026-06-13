import json
from pathlib import Path

from agent.agents.communication import AgentTaskRequest, AgentTaskResult
from agent.core.contracts import Artifact, OperationRequest, OperationResult
from agent.data_processing.contracts import DatasetSpec, DataOperationResult
from agent.hpo.contracts import Objective, SearchParameter, SearchSpace, TrialBudget


def test_core_contracts_are_json_serializable(tmp_path: Path) -> None:
    request = OperationRequest(
        stage="training",
        task_type="speaker_verification",
        model_family="ecapa_tdnn",
        runner="fake",
        config_path=str(tmp_path / "config.yaml"),
        parameters={"lr": 0.001},
    )
    result = OperationResult(
        status="success",
        stage="training",
        metrics={"validation": {"loss": 0.2}},
        artifacts=[Artifact("checkpoint", "best", str(tmp_path / "best.ckpt"))],
    )

    assert json.loads(request.to_json())["stage"] == "training"
    payload = json.loads(result.to_json())
    assert payload["metrics"]["validation"]["loss"] == 0.2
    assert payload["artifacts"][0]["type"] == "checkpoint"


def test_agent_and_domain_contracts_keep_required_fields(tmp_path: Path) -> None:
    request = AgentTaskRequest(action="optimize", objective="improve")
    result = AgentTaskResult(
        status="success",
        summary={"path": tmp_path},
        request_id=request.request_id,
    )
    dataset = DatasetSpec("demo", "text", str(tmp_path))
    operation = DataOperationResult(status="success", operation="validate_dataset")
    search_space = SearchSpace([SearchParameter("lr", "float", low=1e-4, high=1e-2)])

    assert json.loads(result.to_json())["request_id"] == request.request_id
    assert dataset.to_dict()["dataset_type"] == "text"
    assert operation.to_dict()["status"] == "success"
    assert search_space.to_dict()["parameters"][0]["name"] == "lr"
    assert Objective("eer").to_dict()["mode"] == "min"
    assert TrialBudget("small", epochs=1).to_dict()["epochs"] == 1
