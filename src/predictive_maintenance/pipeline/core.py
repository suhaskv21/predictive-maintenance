"""The ``Stage[T]`` contract and the thin ``Pipeline[T]`` orchestrator (ADR 0002)."""

from collections.abc import Sequence
from typing import Protocol


class Stage[T](Protocol):
    """A processing step that takes data of type ``T`` and returns data of type ``T``."""

    def run(self, data: T) -> T:
        """Process ``data`` and return the result."""
        ...


class Pipeline[T]:
    """Run stages in their configured order, passing each output to the next stage."""

    def __init__(self, stages: Sequence[Stage[T]]) -> None:
        """Create a pipeline from an ordered sequence of stages."""
        self._stages = tuple(stages)

    def run(self, data: T) -> T:
        """Run every stage in order and return the final output."""
        for stage in self._stages:
            data = stage.run(data)
        return data
