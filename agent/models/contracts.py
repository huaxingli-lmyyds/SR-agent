"""Model configuration adapter contracts."""

from typing import Any, Dict, Protocol


class ModelAdapter(Protocol):
    model_family: str
    implementation: str
    default_evaluation_config: str | None

    def validate_config(self, config: Dict[str, Any]) -> None: ...

    def default_search_space(self) -> Dict[str, Any]: ...

    def validate_parameters(self, parameters: Dict[str, Any]) -> None: ...


__all__ = ["ModelAdapter"]
