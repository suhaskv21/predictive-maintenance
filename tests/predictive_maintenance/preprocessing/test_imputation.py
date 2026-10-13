import numpy as np
import polars as pl
import pytest

from predictive_maintenance.data import schema
from predictive_maintenance.preprocessing.imputation import (
    IMPUTING_STRATEGIES,
    STRATEGIES,
    InputImputer,
)

TORQUE = "Torque [Nm]"
AIR = "Air temperature [K]"
PROCESS = "Process temperature [K]"
SPEED = "Rotational speed [rpm]"


def complete(n: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    air = rng.normal(300, 2, n)
    return pl.DataFrame(
        {
            "Type": rng.choice(["L", "L", "M", "H"], n),
            AIR: air,
            PROCESS: air + 10 + rng.normal(0, 0.1, n),  # tightly tied to air temperature
            SPEED: rng.integers(1200, 2000, n),
            TORQUE: rng.normal(40, 10, n),
            "Tool wear [min]": rng.integers(0, 250, n),
        }
    )


def test_strategies_run_simplest_first_and_reject_has_no_transformer() -> None:
    assert STRATEGIES == ("reject", "median", "regression", "native")
    assert IMPUTING_STRATEGIES == ("median", "regression", "native")


def test_an_unknown_strategy_is_rejected_at_fit() -> None:
    with pytest.raises(ValueError, match="strategy"):
        InputImputer("reject").fit(complete())


def test_fit_refuses_training_rows_with_absent_inputs() -> None:
    frame = complete().with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col(TORQUE)).alias(TORQUE)
    )
    with pytest.raises(ValueError, match="complete"):
        InputImputer("median").fit(frame)


@pytest.mark.parametrize("strategy", IMPUTING_STRATEGIES)
def test_a_complete_reading_passes_through_unchanged(strategy: str) -> None:
    frame = complete()
    out = InputImputer(strategy).fit(frame).transform(frame)
    expected = frame.select(schema.FEATURE_COLUMNS).with_columns(
        pl.col(c).cast(pl.Float64) for c in schema.NUMERIC_FEATURES
    )
    assert out.equals(expected)


def test_median_fills_sensors_with_the_training_median_and_type_with_the_mode() -> None:
    train = complete()
    test = train[:3].with_columns(
        pl.lit(None, dtype=pl.Float64).alias(TORQUE),
        pl.lit(None, dtype=pl.String).alias("Type"),
    )
    out = InputImputer("median").fit(train).transform(test)
    assert out[TORQUE].to_list() == [float(train[TORQUE].median())] * 3  # type: ignore[arg-type]
    assert out["Type"].to_list() == ["L"] * 3
    assert out[AIR].to_list() == train[AIR][:3].to_list()


def test_statistics_come_from_the_training_fold_only() -> None:
    train, other = complete(), complete().with_columns(pl.col(TORQUE) + 1000)
    absent = other[:1].with_columns(pl.lit(None, dtype=pl.Float64).alias(TORQUE))
    out = InputImputer("median").fit(train).transform(absent)
    assert out[TORQUE][0] == pytest.approx(float(train[TORQUE].median()))  # type: ignore[arg-type]


def test_regression_uses_the_sensors_that_are_present() -> None:
    train = complete()
    test = train[:50].with_columns(pl.lit(None, dtype=pl.Float64).alias(PROCESS))
    regression = InputImputer("regression").fit(train).transform(test)
    median = InputImputer("median").fit(train).transform(test)
    truth = train[PROCESS][:50].to_numpy()
    assert np.abs(regression[PROCESS].to_numpy() - truth).max() < 0.5
    assert np.abs(median[PROCESS].to_numpy() - truth).mean() > 1.0


def test_regression_handles_several_absent_sensors_and_a_full_outage() -> None:
    train = complete()
    several = train[:5].with_columns(
        pl.lit(None, dtype=pl.Float64).alias(PROCESS), pl.lit(None, dtype=pl.Float64).alias(TORQUE)
    )
    outage = train[:5].with_columns(
        pl.lit(None, dtype=pl.Float64).alias(c) for c in schema.NUMERIC_FEATURES
    )
    imputer = InputImputer("regression").fit(train)
    assert imputer.transform(several).null_count().sum_horizontal().item() == 0
    filled = imputer.transform(outage)
    assert filled.null_count().sum_horizontal().item() == 0
    assert filled[TORQUE].to_list() == [float(train[TORQUE].median())] * 5  # type: ignore[arg-type]


def test_nan_counts_as_absent() -> None:
    train = complete()
    test = train[:2].with_columns(pl.Series(TORQUE, [float("nan"), 40.0]))
    out = InputImputer("median").fit(train).transform(test)
    assert out[TORQUE][0] == pytest.approx(float(train[TORQUE].median()))  # type: ignore[arg-type]
    assert out[TORQUE][1] == 40.0


def test_native_leaves_absent_sensors_and_type_for_the_model() -> None:
    train = complete()
    test = train[:2].with_columns(
        pl.lit(None, dtype=pl.Float64).alias(TORQUE), pl.lit(None, dtype=pl.String).alias("Type")
    )
    out = InputImputer("native").fit(train).transform(test)
    assert out[TORQUE].null_count() == 2
    assert out["Type"].null_count() == 2
