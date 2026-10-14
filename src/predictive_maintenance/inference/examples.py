"""The Example readings the app offers and the recorded outcome shown for an unedited one.

Imports no UI framework. The Readings come from the unit's manifest, where training stored them
(train partition only, ADR 0009), because the browser has no dataset. ``recorded_outcome`` is the
only place that decides whether an outcome may be shown: only for a Reading still equal to the
Example reading that was chosen.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from predictive_maintenance.data import schema
from predictive_maintenance.inference.artifact import DisabledUnit, LoadedUnit


@dataclass(frozen=True)
class ExampleReading:
    """One Example reading: a label that hides the outcome, the six inputs and the outcome."""

    label: str
    reading: dict[str, Any]
    machine_failure: bool


def _parse(row: object, number: int) -> ExampleReading | None:
    if not isinstance(row, dict) or not {*schema.FEATURE_COLUMNS, schema.TARGET} <= set(row):
        return None
    reading = {name: row[name] for name in schema.FEATURE_COLUMNS}
    return ExampleReading(f"Example reading {number}", reading, bool(row[schema.TARGET]))


def load_examples(unit: LoadedUnit | DisabledUnit) -> tuple[ExampleReading, ...]:
    """The Example readings stored in ``unit``; none if the unit is disabled or stores none.

    A malformed entry is left out. Labels are numbered in stored order and never name the outcome.
    """
    if isinstance(unit, DisabledUnit):
        return ()
    rows = unit.manifest.get("examples")
    if not isinstance(rows, list):
        return ()
    parsed = (_parse(row, number) for number, row in enumerate(rows, start=1))
    return tuple(example for example in parsed if example is not None)


def _same(chosen: object, entered: object) -> bool:
    if isinstance(chosen, str) or isinstance(entered, str) or entered is None:
        return chosen == entered
    try:
        return math.isclose(float(chosen), float(entered), rel_tol=0, abs_tol=1e-9)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def recorded_outcome(example: ExampleReading | None, reading: Mapping[str, Any]) -> bool | None:
    """The recorded ``Machine failure`` of ``example``, or ``None`` once any input differs.

    ``None`` also when no Example reading is chosen. Editing any of the six inputs, or emptying
    one, makes the entered Reading a different one, so its outcome is no longer shown.
    """
    if example is None:
        return None
    if all(_same(example.reading[name], reading.get(name)) for name in schema.FEATURE_COLUMNS):
        return example.machine_failure
    return None


def find_example(examples: Sequence[ExampleReading], label: str | None) -> ExampleReading | None:
    """The Example reading named ``label``; ``None`` if nothing or an unknown label is chosen."""
    return next((example for example in examples if example.label == label), None)
