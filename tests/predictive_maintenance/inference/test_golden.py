import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from predictive_maintenance.evaluation.threshold import OperatingPoint
from predictive_maintenance.inference.artifact import LoadedUnit, load_unit
from predictive_maintenance.inference.golden import check_golden
from predictive_maintenance.training.unit import (
    ArtifactUnit,
    build_operating_metrics,
    reject_policy,
    write_unit,
)

MODEL = "toy"


def readings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Type": ["L", "M", "H", "L"],
            "Air temperature [K]": [298.0, 300.0, 302.0, 299.0],
            "Process temperature [K]": [308.0, 310.0, 312.0, 309.0],
            "Rotational speed [rpm]": [1400, 1500, 1600, 1450],
            "Torque [Nm]": [30.0, 40.0, 50.0, 35.0],
            "Tool wear [min]": [0, 100, 200, 50],
        }
    )


def write(models_dir: Path, *, stored_scores: list[float]) -> ArtifactUnit:
    """A unit whose Pipeline scores every Reading 0.5; ``stored_scores`` are the manifest's."""
    frame = readings()
    pipeline = Pipeline([("model", DummyClassifier(strategy="prior"))]).fit(frame, [0, 1, 0, 1])
    thresholds = {5.0: 0.3, 10.0: 0.2}
    return write_unit(
        models_dir,
        pipeline=pipeline,
        point=OperatingPoint(
            model=MODEL,
            default_cost_ratio=10.0,
            thresholds=thresholds,
            bands=None,
            risk_floor=0.01,
        ),
        policy=reject_policy(MODEL, 10.0),
        metrics=build_operating_metrics(
            MODEL, [0, 0, 1, 1], np.array([[0.1, 0.2, 0.6, 0.9]]), thresholds, frame
        ),
        golden=frame.head(len(stored_scores)),
        golden_scores=stored_scores,
    )


def loaded(models_dir: Path) -> LoadedUnit:
    unit = load_unit(models_dir, MODEL)
    assert isinstance(unit, LoadedUnit)
    return unit


def test_a_fresh_unit_passes_its_golden_check(tmp_path: Path) -> None:
    write(tmp_path, stored_scores=[0.5, 0.5])
    result = check_golden(loaded(tmp_path))
    assert result.passed
    assert result.differences == ()


def test_an_altered_stored_score_fails_and_names_the_reading(tmp_path: Path) -> None:
    write(tmp_path, stored_scores=[0.5, 0.25])
    result = check_golden(loaded(tmp_path))
    assert not result.passed
    assert len(result.differences) == 1
    assert "Reading 2" in result.differences[0]
    assert "0.25" in result.differences[0]
    assert "0.5" in result.differences[0]


def test_a_difference_within_the_tolerance_passes(tmp_path: Path) -> None:
    write(tmp_path, stored_scores=[0.5 + 5e-10, 0.5])
    assert check_golden(loaded(tmp_path)).passed


def test_a_difference_beyond_the_tolerance_fails(tmp_path: Path) -> None:
    write(tmp_path, stored_scores=[0.5 + 2e-9, 0.5])
    assert not check_golden(loaded(tmp_path)).passed


@pytest.mark.parametrize(
    "damage",
    [
        lambda m: m.pop("golden"),
        lambda m: m["golden"].pop("scores"),
        lambda m: m["golden"]["scores"].pop(),
        lambda m: m["golden"]["readings"][0].pop("Torque [Nm]"),
        lambda m: m["golden"].update(readings=[]),
        lambda m: m["golden"]["scores"].__setitem__(0, "high"),
    ],
)
def test_an_unusable_golden_block_fails_in_one_line_instead_of_raising(
    tmp_path: Path, damage: Callable[[dict[str, Any]], None]
) -> None:
    unit = write(tmp_path, stored_scores=[0.5, 0.5])
    manifest = json.loads(unit.manifest.read_text())
    damage(manifest)
    unit.manifest.write_text(json.dumps(manifest))
    result = check_golden(loaded(tmp_path))
    assert not result.passed
    assert len(result.differences) == 1
    assert "\n" not in result.differences[0]
