"""Preparation lives in the controller; workers only receive immutable runner inputs."""
from dataclasses import dataclass
from functools import wraps
import json


@dataclass(frozen=True)
class StageExecution:
    stage: str
    runner: str
    config_path: str
    payload_json: str
    adapter_type: str = ""

    @classmethod
    def prepare_execution(cls, stage, runner, config_path, **payload):
        from agent.runners.registry import get_runner_adapter
        adapter = type(get_runner_adapter(runner))
        return cls(stage, runner, str(config_path), json.dumps(payload, allow_nan=False),
                   f"{adapter.__module__}:{adapter.__qualname__}")


def execute_stage(request):
    from agent.runners.registry import get_runner_adapter
    from agent.runners.contracts import validate_runner_result
    try:
        adapter = get_runner_adapter(request.runner)
    except ValueError:
        import importlib
        module, name = request.adapter_type.split(":", 1)
        factory = importlib.import_module(module)
        for part in name.split("."):
            factory = getattr(factory, part)
        adapter = factory()
        if adapter.runner != request.runner:
            raise ValueError("worker adapter identity differs from prepared request")
    payload = json.loads(request.payload_json)
    if request.stage == "training":
        result = adapter.run_training(request.config_path, payload["overrides"])
    else:
        result = adapter.run_evaluation(request.config_path, **payload)
    return validate_runner_result(result, request.stage)


def stage_operation(function):
    """Keep a synchronous tool API while exposing its controller coroutine."""
    @wraps(function)
    def synchronous(*args, **kwargs):
        operation = function(*args, **kwargs)
        response = None
        while True:
            try:
                request = operation.send(response)
            except StopIteration as done:
                return done.value
            try:
                response = execute_stage(request)
            except Exception as exc:
                response = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    synchronous.prepare_execution = function
    return synchronous
