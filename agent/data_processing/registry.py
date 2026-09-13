"""Registry for data-type-independent processing operations."""

from __future__ import annotations

from typing import Any, Dict, Protocol

from .contracts import (
    DataOperationResult,
    DataProfile,
    DatasetSpec,
    OperationImpact,
    QualityDecision,
    QualityPolicy,
)


class DataProfiler(Protocol):
    data_type: str

    def profile(
        self,
        dataset: DatasetSpec,
        policy: QualityPolicy,
    ) -> DataProfile: ...


class DataProcessor(Protocol):
    operation_name: str
    supported_data_types: set[str]
    parameter_schema: Dict[str, Any]

    def validate(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> None: ...

    def preview(
        self,
        dataset: DatasetSpec,
        parameters: Dict[str, Any],
    ) -> OperationImpact: ...

    def execute(
        self,
        dataset: DatasetSpec,
        parameters: Dict[str, Any],
    ) -> DataOperationResult: ...


class QualityGate(Protocol):
    def evaluate(self, profile: DataProfile) -> QualityDecision: ...


class DataProfilerRegistry:
    def __init__(self) -> None:
        self._profilers: Dict[str, DataProfiler] = {}

    def register(self, profiler: DataProfiler) -> None:
        self._profilers[profiler.data_type] = profiler

    def get(self, data_type: str) -> DataProfiler | None:
        return self._profilers.get(data_type)


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
                "supported_data_types": sorted(processor.supported_data_types),
                "parameter_schema": dict(
                    getattr(processor, "parameter_schema", {})
                ),
            }
            for name, processor in sorted(self._processors.items())
        }


PROCESSORS = DataProcessorRegistry()


PROFILERS = DataProfilerRegistry()


def register_processor(processor: DataProcessor) -> None:
    PROCESSORS.register(processor)


def register_profiler(profiler: DataProfiler) -> None:
    PROFILERS.register(profiler)
