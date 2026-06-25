"""Registry for data-type-independent processing operations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Protocol

from .contracts import DataOperationResult, DatasetSpec


class DataProcessor(Protocol):
    operation_name: str
    supported_data_types: Iterable[str]
    parameter_schema: Dict[str, Any]

    def validate(self, dataset: DatasetSpec, parameters: Dict[str, Any]) -> None: ...

    def execute(
        self,
        dataset: DatasetSpec,
        parameters: Dict[str, Any],
    ) -> DataOperationResult: ...


class DataProcessorRegistry:
    def __init__(self) -> None:
        self._processors: Dict[str, DataProcessor] = {}

    def register(self, processor: DataProcessor) -> None:
        self._processors[processor.operation_name] = processor

    def get(self, operation_name: str, dataset_type: str) -> DataProcessor:
        processor = self._processors.get(operation_name)
        if processor is None:
            raise KeyError(f"unknown data operation: {operation_name}")
        supported = set(processor.supported_data_types)
        if "*" not in supported and dataset_type not in supported:
            raise ValueError(
                f"operation {operation_name} does not support dataset type {dataset_type}"
            )
        return processor

    def describe(self) -> Dict[str, Any]:
        return {
            name: {
                "supported_data_types": list(processor.supported_data_types),
                "parameter_schema": dict(getattr(processor, "parameter_schema", {})),
            }
            for name, processor in sorted(self._processors.items())
        }


PROCESSORS = DataProcessorRegistry()


def register_processor(processor: DataProcessor) -> None:
    PROCESSORS.register(processor)
