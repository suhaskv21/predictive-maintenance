"""Assess one Reading against a checked unit; imports no UI framework and writes no log.

A Reading is a mapping from the six input names (``schema.FEATURE_COLUMNS``) to values; other keys
are ignored. Each input is classified as present or unusable (absent, not a number, not finite, or
out of range per ``[plausibility]``); an unusable value counts as absent. Any Incomplete reading is
refused (this build's ``reject`` policy), with no Failure risk, Flag, threshold or fallback score.
Otherwise the Pipeline scores the Reading and the Flag is
decided on the unrounded score against the model's own Operating threshold. Rounding and wording
belong to the presentation, not here.
"""

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, SupportsFloat, cast

import polars as pl

from predictive_maintenance.config import InputRange, PlausibilityConfig
from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.compare import score_readings
from predictive_maintenance.inference.artifact import LoadedUnit

ABSENT = "absent"
NOT_A_NUMBER = "not a number"
NOT_FINITE = "not finite"
OUT_OF_RANGE = "out of range"


@dataclass(frozen=True)
class MissingInput:
    """One input that is absent or unusable, and why."""

    name: str
    reason: str


@dataclass(frozen=True)
class Assessment:
    """The result of assessing one Reading.

    ``status`` is ``complete`` (``failure_risk`` is the unrounded score and ``flag`` the call at the
    Operating threshold) or ``refused`` (``missing``, ``refusal_reason`` and ``recoverable`` are
    set, and ``failure_risk`` and ``flag`` are None).
    """

    status: Literal["complete", "refused"]
    missing: tuple[MissingInput, ...]
    refusal_reason: str | None
    recoverable: bool
    failure_risk: float | None
    flag: bool | None


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _classify_type(value: object, types: tuple[str, ...]) -> str | None:
    if _is_blank(value):
        return ABSENT
    return None if isinstance(value, str) and value in types else OUT_OF_RANGE


def _classify_number(value: object, valid: InputRange) -> str | None:
    if _is_blank(value):
        return ABSENT
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return NOT_A_NUMBER
    try:
        number = float(value)
    except OverflowError:
        return OUT_OF_RANGE
    if math.isnan(number):
        return NOT_A_NUMBER
    if math.isinf(number):
        return NOT_FINITE
    return None if valid.contains(number) else OUT_OF_RANGE


def _missing_inputs(
    reading: Mapping[str, object], plausibility: PlausibilityConfig
) -> tuple[MissingInput, ...]:
    reasons = {
        schema.CATEGORICAL_FEATURE: _classify_type(
            reading.get(schema.CATEGORICAL_FEATURE), plausibility.types
        ),
        **{
            name: _classify_number(reading.get(name), plausibility.ranges[name])
            for name in schema.NUMERIC_FEATURES
        },
    }
    return tuple(
        MissingInput(name, reason)
        for name in schema.FEATURE_COLUMNS
        if (reason := reasons[name]) is not None
    )


def assess(
    unit: LoadedUnit, reading: Mapping[str, object], plausibility: PlausibilityConfig
) -> Assessment:
    """Assess one Reading: refuse it if any input is absent or unusable, else score and Flag it.

    This build scores no Incomplete reading whatever the unit's policy lists, and a refusal is
    never recoverable because the policy's reference is empty (the loader accepts only ``reject``).
    """
    missing = _missing_inputs(reading, plausibility)
    if missing:
        return Assessment(
            status="refused",
            missing=missing,
            refusal_reason="inputs absent or unusable: " + ", ".join(m.name for m in missing),
            recoverable=False,
            failure_risk=None,
            flag=None,
        )
    frame = pl.DataFrame(
        {
            schema.CATEGORICAL_FEATURE: [reading[schema.CATEGORICAL_FEATURE]],
            **{
                name: [float(cast(SupportsFloat, reading[name]))]
                for name in schema.NUMERIC_FEATURES
            },
        }
    )
    score = float(score_readings(unit.pipeline, frame)[0])
    return Assessment(
        status="complete",
        missing=(),
        refusal_reason=None,
        recoverable=False,
        failure_risk=score,
        flag=score >= unit.operating_point.threshold,
    )
