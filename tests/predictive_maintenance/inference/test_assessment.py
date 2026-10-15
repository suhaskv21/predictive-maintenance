import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from predictive_maintenance.config import load_config
from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.compare import score_readings
from predictive_maintenance.evaluation.missingness import MissingDataPolicy, PatternDegradation
from predictive_maintenance.evaluation.threshold import OperatingPoint
from predictive_maintenance.inference.artifact import LoadedUnit, load_unit
from predictive_maintenance.inference.assessment import (
    ABSENT,
    NOT_A_NUMBER,
    NOT_FINITE,
    OUT_OF_RANGE,
    Assessment,
    MissingInput,
    assess,
)
from predictive_maintenance.training.candidates import build_candidates
from predictive_maintenance.training.unit import (
    build_operating_metrics,
    reject_policy,
    write_unit,
)

MODEL = "toy"
PLAUSIBILITY = load_config().plausibility
AIR, PROCESS, SPEED, TORQUE, WEAR = schema.NUMERIC_FEATURES
TYPE = schema.CATEGORICAL_FEATURE
READING: dict[str, Any] = {
    TYPE: "M",
    AIR: 300.0,
    PROCESS: 310.0,
    SPEED: 1500,
    TORQUE: 40.0,
    WEAR: 100,
}


def readings() -> pl.DataFrame:
    rng = np.random.default_rng(1)
    n = 300
    torque = rng.normal(40, 10, n)
    return pl.DataFrame(
        {
            TYPE: rng.choice(list(schema.TYPE_LEVELS), n),
            AIR: rng.normal(300, 2, n),
            PROCESS: rng.normal(310, 1, n),
            SPEED: rng.integers(1200, 2000, n),
            TORQUE: torque,
            WEAR: rng.integers(0, 250, n),
        }
    ), (torque > 48).astype(int)


def write(
    models_dir: Path, pipeline: Pipeline, frame: pl.DataFrame, y: Any, threshold: float
) -> LoadedUnit:
    thresholds = {5.0: threshold * 1.5, 10.0: threshold}
    write_unit(
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
        metrics=build_operating_metrics(MODEL, y, np.array([y * 0.5]), thresholds, frame),
        golden=frame.head(1),
        golden_scores=[0.5],
    )
    unit = load_unit(models_dir, MODEL)
    assert isinstance(unit, LoadedUnit)
    return unit


def constant_unit(tmp_path: Path, *, positives: int, threshold: float) -> LoadedUnit:
    """A unit that scores every Reading ``positives / 1000``, so the score is known exactly."""
    frame = pl.concat([readings()[0].head(250)] * 4)
    y = np.array([1] * positives + [0] * (1000 - positives))
    pipeline = Pipeline([("model", DummyClassifier(strategy="prior"))]).fit(frame, y)
    return write(tmp_path, pipeline, frame, y, threshold)


class Unscorable:
    """Stands in for a Pipeline that must never be asked for a score."""

    def predict_proba(self, x: object) -> object:
        raise AssertionError("a refused Reading must not be scored")


def with_(**changes: object) -> dict[str, Any]:
    return {**READING, **changes}


def without(name: str) -> dict[str, Any]:
    return {k: v for k, v in READING.items() if k != name}


@pytest.fixture
def unit(tmp_path: Path) -> LoadedUnit:
    return constant_unit(tmp_path, positives=300, threshold=0.2)


def test_the_flag_follows_the_chosen_cost_ratio(tmp_path: Path) -> None:
    # Thresholds: 5:1 -> 0.375, 10:1 -> 0.25; the score is 0.3.
    unit = constant_unit(tmp_path, positives=300, threshold=0.25)
    assert assess(unit, READING, PLAUSIBILITY).flag is True
    assert assess(unit, READING, PLAUSIBILITY, 10.0).flag is True
    assert assess(unit, READING, PLAUSIBILITY, 5.0).flag is False


def test_a_cost_ratio_without_a_threshold_is_refused(unit: LoadedUnit) -> None:
    with pytest.raises(ValueError, match="no threshold for cost ratio 20"):
        assess(unit, READING, PLAUSIBILITY, 20.0)


def test_a_complete_reading_gives_its_unrounded_failure_risk_and_the_flag(
    unit: LoadedUnit,
) -> None:
    result = assess(unit, READING, PLAUSIBILITY)
    assert result == Assessment(
        status="complete",
        missing=(),
        refusal_reason=None,
        recoverable=False,
        failure_risk=0.3,
        flag=True,
    )


def test_a_failure_risk_below_the_operating_threshold_is_not_flagged(tmp_path: Path) -> None:
    result = assess(constant_unit(tmp_path, positives=100, threshold=0.2), READING, PLAUSIBILITY)
    assert (result.status, result.failure_risk, result.flag) == ("complete", 0.1, False)


def test_a_failure_risk_exactly_at_the_operating_threshold_is_flagged(tmp_path: Path) -> None:
    result = assess(constant_unit(tmp_path, positives=200, threshold=0.2), READING, PLAUSIBILITY)
    assert result.failure_risk == 0.2 and result.flag is True


@pytest.mark.parametrize(("positives", "flagged"), [(218, False), (219, True)])
def test_the_flag_follows_the_unrounded_score_when_the_rounded_percent_equals_the_thresholds(
    tmp_path: Path, positives: int, flagged: bool
) -> None:
    # Threshold 21.85 % and scores 21.8 % / 21.9 % all display as 22 %.
    unit = constant_unit(tmp_path, positives=positives, threshold=0.2185)
    result = assess(unit, READING, PLAUSIBILITY)
    assert result.failure_risk == positives / 1000
    assert result.flag is flagged


def test_the_flag_uses_the_default_cost_ratios_threshold_not_the_other_ratios(
    tmp_path: Path,
) -> None:
    # The other ratio's threshold is 1.5 x 0.2 = 0.3, which a score of 0.25 would not reach.
    unit = constant_unit(tmp_path, positives=250, threshold=0.2)
    assert assess(unit, READING, PLAUSIBILITY).flag is True


def test_numpy_numbers_and_integers_are_accepted_as_numbers(unit: LoadedUnit) -> None:
    reading = with_(**{AIR: np.float64(300.0), SPEED: np.int64(1500), WEAR: 100.0})
    assert assess(unit, reading, PLAUSIBILITY).status == "complete"


def test_columns_that_are_not_inputs_are_ignored(unit: LoadedUnit) -> None:
    extra = {**READING, "TWF": 1, "Machine failure": 1, "UDI": 5}
    assert assess(unit, extra, PLAUSIBILITY) == assess(unit, READING, PLAUSIBILITY)


def test_a_complete_reading_scores_bit_identically_to_the_pipeline_scored_directly(
    tmp_path: Path,
) -> None:
    frame, y = readings()
    candidate = build_candidates(3)["random_forest_unweighted"]
    pipeline = candidate.make_pipeline().fit(frame, y)
    unit = write(tmp_path, pipeline, frame, y, threshold=0.3)
    direct = score_readings(pipeline, frame)
    for row, expected in zip(frame.to_dicts(), direct, strict=True):
        result = assess(unit, row, PLAUSIBILITY)
        assert result.failure_risk == expected
        assert result.flag is bool(expected >= 0.3)


UNUSABLE = [
    # (input, value, reason)
    *[(name, None, ABSENT) for name in schema.FEATURE_COLUMNS],
    *[(name, "", ABSENT) for name in schema.FEATURE_COLUMNS],
    *[(name, "abc", NOT_A_NUMBER) for name in schema.NUMERIC_FEATURES],
    *[(name, "300", NOT_A_NUMBER) for name in schema.NUMERIC_FEATURES],
    *[(name, True, NOT_A_NUMBER) for name in schema.NUMERIC_FEATURES],
    *[(name, [1.0], NOT_A_NUMBER) for name in schema.NUMERIC_FEATURES],
    *[(name, math.nan, NOT_A_NUMBER) for name in schema.NUMERIC_FEATURES],
    *[(name, np.float64("nan"), NOT_A_NUMBER) for name in schema.NUMERIC_FEATURES],
    *[(name, math.inf, NOT_FINITE) for name in schema.NUMERIC_FEATURES],
    *[(name, -math.inf, NOT_FINITE) for name in schema.NUMERIC_FEATURES],
    *[(TYPE, bad, OUT_OF_RANGE) for bad in ("X", "l", "M ", 5, 1.5, True, ["M"], math.nan)],
    (AIR, 199.99, OUT_OF_RANGE),
    (AIR, 400.01, OUT_OF_RANGE),
    (PROCESS, 199.99, OUT_OF_RANGE),
    (PROCESS, 450.01, OUT_OF_RANGE),
    (SPEED, 0, OUT_OF_RANGE),
    (SPEED, -1, OUT_OF_RANGE),
    (SPEED, 10_000.01, OUT_OF_RANGE),
    (TORQUE, -0.01, OUT_OF_RANGE),
    (TORQUE, 500.01, OUT_OF_RANGE),
    (WEAR, -0.01, OUT_OF_RANGE),
    (WEAR, 1_000.01, OUT_OF_RANGE),
]


@pytest.mark.parametrize(("name", "value", "reason"), UNUSABLE)
def test_an_absent_or_unusable_input_refuses_the_reading_and_names_it_with_its_reason(
    unit: LoadedUnit, name: str, value: object, reason: str
) -> None:
    refused = dataclasses.replace(unit, pipeline=Unscorable())  # type: ignore[arg-type]
    result = assess(refused, with_(**{name: value}), PLAUSIBILITY)
    assert result.status == "refused"
    assert result.missing == (MissingInput(name, reason),)
    assert result.refusal_reason and name in result.refusal_reason
    assert result.recoverable is False
    assert result.failure_risk is None
    assert result.flag is None


def test_a_refusal_is_never_recoverable_in_this_build_even_with_a_measured_reference(
    unit: LoadedUnit,
) -> None:
    degradation = PatternDegradation((TORQUE,), 0.95, 0.01, 0.02, 1.0)
    with_reference = MissingDataPolicy(
        MODEL, "reject", 10.0, (), measured_strategy="median", measured=(degradation,)
    )
    result = assess(dataclasses.replace(unit, policy=with_reference), without(TORQUE), PLAUSIBILITY)
    assert result.status == "refused" and result.recoverable is False


def test_an_incomplete_reading_is_refused_whatever_the_policy_says_is_scorable(
    unit: LoadedUnit,
) -> None:
    degradation = PatternDegradation((TORQUE,), 0.95, 0.01, 0.02, 1.0)
    scoring = MissingDataPolicy(MODEL, "median", 10.0, (degradation,))
    refused = dataclasses.replace(unit, policy=scoring, pipeline=Unscorable())  # type: ignore[arg-type]
    result = assess(refused, without(TORQUE), PLAUSIBILITY)
    assert result.status == "refused" and result.failure_risk is None


@pytest.mark.parametrize("name", schema.FEATURE_COLUMNS)
def test_an_input_missing_from_the_reading_is_absent(unit: LoadedUnit, name: str) -> None:
    result = assess(unit, without(name), PLAUSIBILITY)
    assert result.status == "refused"
    assert result.missing == (MissingInput(name, ABSENT),)


def test_every_unusable_input_is_listed_in_input_order(unit: LoadedUnit) -> None:
    reading = {TORQUE: -5, AIR: math.inf, SPEED: "fast", WEAR: 50.0}
    result = assess(unit, reading, PLAUSIBILITY)
    assert result.missing == (
        MissingInput(TYPE, ABSENT),
        MissingInput(AIR, NOT_FINITE),
        MissingInput(PROCESS, ABSENT),
        MissingInput(SPEED, NOT_A_NUMBER),
        MissingInput(TORQUE, OUT_OF_RANGE),
    )


VALID_LIMITS = [
    (AIR, 200),
    (AIR, 400),
    (PROCESS, 200),
    (PROCESS, 450),
    (SPEED, 0.001),
    (SPEED, 10_000),
    (TORQUE, 0),
    (TORQUE, 500),
    (WEAR, 0),
    (WEAR, 1_000),
]


@pytest.mark.parametrize(("name", "value"), VALID_LIMITS)
def test_a_value_on_a_range_limit_that_is_valid_is_scored(
    unit: LoadedUnit, name: str, value: float
) -> None:
    result = assess(unit, with_(**{name: value}), PLAUSIBILITY)
    assert result.status == "complete" and result.failure_risk == 0.3


@pytest.mark.parametrize("value", ["L", "M", "H"])
def test_each_product_type_is_scored(unit: LoadedUnit, value: str) -> None:
    assert assess(unit, with_(Type=value), PLAUSIBILITY).status == "complete"


def test_the_ranges_come_from_the_configuration_not_from_code(unit: LoadedUnit) -> None:
    narrow = dataclasses.replace(
        PLAUSIBILITY,
        ranges={
            **PLAUSIBILITY.ranges,
            TORQUE: dataclasses.replace(PLAUSIBILITY.ranges[TORQUE], high=30),
        },
    )
    assert assess(unit, READING, narrow).missing == (MissingInput(TORQUE, OUT_OF_RANGE),)


# --- outside the training range -------------------------------------------------------------------


def training_range(unit: LoadedUnit, name: str) -> dict[str, float]:
    return dict(unit.operating_metrics["training_ranges"][name])


def test_a_reading_inside_the_training_ranges_has_nothing_outside(unit: LoadedUnit) -> None:
    assert assess(unit, READING, PLAUSIBILITY).outside_training == ()


@pytest.mark.parametrize("name", schema.NUMERIC_FEATURES)
@pytest.mark.parametrize("edge", ["min", "max"])
def test_a_value_on_the_training_limit_is_inside(unit: LoadedUnit, name: str, edge: str) -> None:
    limit = training_range(unit, name)[edge]
    assert assess(unit, with_(**{name: limit}), PLAUSIBILITY).outside_training == ()


BEYOND = [  # tool wear starts at the plausible minimum of 0, so only its upper side is testable
    (name, edge, step)
    for name in schema.NUMERIC_FEATURES
    for edge, step in (("min", -0.5), ("max", 0.5))
    if (name, edge) != (WEAR, "min")
]


@pytest.mark.parametrize(("name", "edge", "step"), BEYOND)
def test_a_plausible_value_beyond_the_training_range_is_scored_and_named(
    unit: LoadedUnit, name: str, edge: str, step: float
) -> None:
    value = training_range(unit, name)[edge] + step
    result = assess(unit, with_(**{name: value}), PLAUSIBILITY)
    assert result.status == "complete" and result.failure_risk == 0.3 and result.flag
    assert result.outside_training == (name,)


def test_every_input_outside_the_training_range_is_listed_in_input_order(
    unit: LoadedUnit,
) -> None:
    far = {name: training_range(unit, name)["max"] + 1 for name in (WEAR, AIR)}
    assert assess(unit, with_(**far), PLAUSIBILITY).outside_training == (AIR, WEAR)


def test_an_implausible_value_is_refused_not_noted_as_outside_the_training_range(
    unit: LoadedUnit,
) -> None:
    result = assess(unit, with_(**{TORQUE: 900}), PLAUSIBILITY)
    assert result.status == "refused" and result.outside_training == ()


def test_operating_metrics_without_training_ranges_are_refused(unit: LoadedUnit) -> None:
    metrics = {k: v for k, v in unit.operating_metrics.items() if k != "training_ranges"}
    broken = dataclasses.replace(unit, operating_metrics=metrics)
    with pytest.raises(ValueError, match="training_ranges"):
        assess(broken, READING, PLAUSIBILITY)
