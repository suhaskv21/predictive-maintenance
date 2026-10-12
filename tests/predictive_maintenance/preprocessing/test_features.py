from pathlib import Path

import joblib
import polars as pl
import pytest

from predictive_maintenance.data import schema
from predictive_maintenance.preprocessing.features import (
    ENGINEERED_FEATURES,
    TEMPERATURE_DELTA,
    add_engineered_features,
    build_feature_transformer,
)

RAW_SENSOR_NAMES = list(schema.NUMERIC_FEATURES)


def readings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Type": ["L", "M", "H", "L"],
            "Air temperature [K]": [298.0, 299.0, 300.0, 301.0],
            "Process temperature [K]": [308.0, 310.0, 309.0, 311.5],
            "Rotational speed [rpm]": [1500, 1400, 2000, 1000],
            "Torque [Nm]": [40.0, 50.0, 30.0, 60.0],
            "Tool wear [min]": [10, 100, 200, 0],
        }
    )


def test_engineered_features_follow_the_physics_definitions() -> None:
    out = add_engineered_features(readings())
    assert out.columns == [*readings().columns, *ENGINEERED_FEATURES]
    first = out.row(0, named=True)
    # 40 Nm at 1500 rpm = 40 * 1500 * 2*pi/60 W
    assert first["Power [W]"] == pytest.approx(6283.185, rel=1e-6)
    assert first["Temperature delta [K]"] == pytest.approx(10.0)
    assert first["Wear x torque [min Nm]"] == pytest.approx(400.0)


def test_baseline_transformer_outputs_one_hot_type_and_scaled_sensors() -> None:
    out = build_feature_transformer().fit_transform(readings())
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["Type_L", "Type_M", "Type_H", *RAW_SENSOR_NAMES]
    assert out["Type_L"].to_list() == [1.0, 0.0, 0.0, 1.0]
    for sensor in RAW_SENSOR_NAMES:
        assert out[sensor].mean() == pytest.approx(0.0, abs=1e-9)
        assert out[sensor].std(ddof=0) == pytest.approx(1.0)


def test_engineered_transformer_appends_the_physics_features() -> None:
    out = build_feature_transformer(ENGINEERED_FEATURES).fit_transform(readings())
    assert out.columns == [
        "Type_L",
        "Type_M",
        "Type_H",
        *RAW_SENSOR_NAMES,
        *ENGINEERED_FEATURES,
    ]
    for name in ENGINEERED_FEATURES:
        assert out[name].std(ddof=0) == pytest.approx(1.0)


def test_a_subset_of_engineered_features_can_be_requested() -> None:
    out = build_feature_transformer([TEMPERATURE_DELTA]).fit_transform(readings())
    assert out.columns == ["Type_L", "Type_M", "Type_H", *RAW_SENSOR_NAMES, TEMPERATURE_DELTA]


def test_unknown_engineered_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="Power"):
        build_feature_transformer(["Power [kW]"])


@pytest.mark.parametrize("engineered", [(), ENGINEERED_FEATURES])
def test_outcome_and_identifier_columns_are_never_features(engineered: tuple[str, ...]) -> None:
    extra = readings().with_columns(
        pl.Series("Machine failure", [0, 1, 0, 0]),
        pl.Series("TWF", [0, 0, 0, 1]),
        pl.Series("UDI", [1, 2, 3, 4]),
        pl.Series("Product ID", ["a", "b", "c", "d"]),
    )
    with_extra = build_feature_transformer(engineered).fit_transform(extra)
    without = build_feature_transformer(engineered).fit_transform(readings())
    assert with_extra.columns == without.columns
    assert with_extra.equals(without)


def test_scaling_is_fitted_on_training_readings_only() -> None:
    train = readings()
    new = train.head(1).with_columns(pl.lit(1100.0).alias("Torque [Nm]"))
    transformer = build_feature_transformer().fit(train)
    torque = train["Torque [Nm]"]
    expected = (1100.0 - torque.mean()) / torque.std(ddof=0)  # type: ignore[operator]
    assert transformer.transform(new)["Torque [Nm]"].item() == pytest.approx(expected)


def test_transform_needs_only_the_feature_columns() -> None:
    transformer = build_feature_transformer(ENGINEERED_FEATURES).fit(readings())
    one_reading = readings().select(schema.FEATURE_COLUMNS).head(1)
    assert transformer.transform(one_reading).shape == (1, 3 + 5 + 3)


def test_unknown_type_is_rejected_at_transform_time() -> None:
    transformer = build_feature_transformer().fit(readings())
    unknown = readings().head(1).with_columns(pl.lit("X").alias("Type"))
    with pytest.raises(ValueError):
        transformer.transform(unknown)


def test_fitted_transformer_survives_a_joblib_round_trip(tmp_path: Path) -> None:
    transformer = build_feature_transformer(ENGINEERED_FEATURES).fit(readings())
    path = tmp_path / "transformer.joblib"
    joblib.dump(transformer, path)
    restored = joblib.load(path)
    assert restored.transform(readings()).equals(transformer.transform(readings()))
