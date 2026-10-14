"""Golden check: re-score the manifest's fixed train Readings and compare with the stored scores.

Imports no UI framework. A difference means the loaded Pipeline scores differently here (for
example in the browser) than where the unit was built.
"""

from dataclasses import dataclass

import polars as pl

from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.compare import score_readings
from predictive_maintenance.inference.artifact import LoadedUnit

TOLERANCE = 1e-9


@dataclass(frozen=True)
class GoldenResult:
    """``differences`` holds one line per stored score that is missed or cannot be checked."""

    differences: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """True when nothing differs."""
        return not self.differences


def _unusable(why: str) -> GoldenResult:
    return GoldenResult(differences=(f"the stored scores cannot be checked: {why}",))


def check_golden(unit: LoadedUnit, tolerance: float = TOLERANCE) -> GoldenResult:
    """Score the manifest's golden Readings with the unit's Pipeline and compare.

    A golden block that is absent, empty or malformed is a failure, never a pass.
    """
    try:
        golden = unit.manifest["golden"]
        stored = [float(score) for score in golden["scores"]]
        rows = golden["readings"]
        if any(not set(schema.FEATURE_COLUMNS) <= set(row) for row in rows):
            return _unusable("a Reading lacks one of the six inputs")
        frame = pl.DataFrame(rows).select(schema.FEATURE_COLUMNS)
        if not stored or frame.height != len(stored):
            return _unusable("it has no Readings or the Readings and scores differ in number")
        fresh = score_readings(unit.pipeline, frame).tolist()
    except Exception as error:
        return _unusable(" ".join(repr(error).split()))
    differences = tuple(
        f"Reading {number}: stored score {stored_score!r}, scored {fresh_score!r}"
        for number, (stored_score, fresh_score) in enumerate(
            zip(stored, fresh, strict=True), start=1
        )
        if not abs(stored_score - fresh_score) <= tolerance
    )
    return GoldenResult(differences=differences)
