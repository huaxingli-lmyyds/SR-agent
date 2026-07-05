import json
from pathlib import Path

from agent.agents.communication import AgentTaskRequest, AgentTaskResult
from agent.core.contracts import Artifact, OperationRequest, OperationResult
from agent.data_processing.contracts import DatasetSpec, DataOperationResult
from agent.hpo.contracts import (
    Objective,
    SearchParameter,
    SearchSpace,
    StrategyDecisionRecord,
    StrategyProposal,
    TrialBudget,
)


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


def test_hpo_strategy_audit_contracts_are_json_serializable() -> None:
    proposal = StrategyProposal(
        action="switch_strategy",
        requested_strategy="adaptive_search",
        reason_codes=["use_history"],
    )
    decision = StrategyDecisionRecord(
        decision="approved",
        proposal_id=proposal.proposal_id,
        proposal=proposal.to_dict(),
        adopted_strategy="adaptive_search",
        adopted_search_space={"parameters": [], "constraints": []},
        adopted_budgets=[{"stage": "full", "epochs": 10}],
        adopted_max_training_runs=4,
        accepted_fields=["requested_strategy"],
        reason_codes=["proposal_approved"],
    )

    assert proposal.to_dict()["proposal_id"].startswith("proposal_")
    assert decision.to_dict()["proposal_id"] == proposal.proposal_id


def test_llm_cannot_supply_strategy_proposal_audit_identity() -> None:
    proposal = StrategyProposal.from_dict({
        "action": "keep_strategy",
        "proposal_id": "spoofed",
        "created_at": "spoofed",
    })

    assert proposal.proposal_id != "spoofed"
    assert proposal.created_at != "spoofed"


def test_trial_result_does_not_finish_parent_experiment(tmp_path: Path, minimal_config, dataset_dir) -> None:
    from agent.core.experiment_service import ExperimentService
    from agent.utils.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(tmp_path / "experiments")
    experiment_id = tracker.create_hpo_experiment(
        config_path=str(minimal_config),
        data_folder=str(dataset_dir),
    )
    result = OperationResult(
        status="failed",
        stage="training",
        metrics={"validation": {"loss": 0.1}},
        error="trial failed",
    )

    assert ExperimentService(tracker).record_result(
        experiment_id,
        result,
        update_status=False,
    )
    record = tracker.get_experiment(experiment_id)
    assert record["status"] == "created"
    assert record.get("error") is None
    assert record["metrics"]["validation"]["loss"] == 0.1


def test_other_model_and_runner_resolve_through_public_adapter_boundary() -> None:
    from agent.core.adapters import resolve_adapter_bundle
    from agent.models import MODEL_ADAPTERS, register_model_adapter
    from agent.runners import RUNNER_ADAPTERS, register_runner_adapter
    from tests.fakes import FakeRunnerAdapter

    class DemoModelAdapter:
        model_family = "demo_encoder"
        implementation = "demo_runtime"
        default_evaluation_config = None

        def validate_config(self, config):
            if "network" not in config:
                raise ValueError("missing network")

        def default_search_space(self):
            return {
                "parameters": [{
                    "name": "width",
                    "parameter_type": "categorical",
                    "choices": [64, 128],
                }],
                "constraints": [],
            }

        def validate_parameters(self, parameters):
            if int(parameters.get("width", 64)) not in {64, 128}:
                raise ValueError("unsupported width")

    class DemoRunner(FakeRunnerAdapter):
        runner = "demo_runtime"
        supported_implementations = {"demo_runtime"}
        supported_model_families = {"demo_encoder"}

    previous_model = MODEL_ADAPTERS.get("demo_encoder")
    previous_runner = RUNNER_ADAPTERS.get("demo_runtime")
    register_model_adapter(DemoModelAdapter())
    register_runner_adapter(DemoRunner(runner="demo_runtime"))
    try:
        bundle = resolve_adapter_bundle(
            "speaker_verification",
            "demo_encoder",
            "demo_runtime",
            "demo_runtime",
        )
        assert bundle.model.default_search_space()["parameters"][0]["name"] == "width"
        bundle.model.validate_parameters({"width": 128})
    finally:
        if previous_model is None:
            MODEL_ADAPTERS.pop("demo_encoder", None)
        else:
            MODEL_ADAPTERS["demo_encoder"] = previous_model
        if previous_runner is None:
            RUNNER_ADAPTERS.pop("demo_runtime", None)
        else:
            RUNNER_ADAPTERS["demo_runtime"] = previous_runner

def test_speaker_model_adapters_are_registered() -> None:
    from agent.models import get_model_adapter

    resnet = get_model_adapter("resnet")
    xvector = get_model_adapter("xvector")

    assert resnet.default_evaluation_config.endswith("verification_resnet.yaml")
    assert xvector.default_evaluation_config.endswith("verification_plda_xvector.yaml")
    assert any(item["name"] == "lr" for item in resnet.default_search_space()["parameters"])
    assert any(item["name"] == "lr_final" for item in xvector.default_search_space()["parameters"])