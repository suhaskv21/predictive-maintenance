import polars as pl
import pytest

from predictive_maintenance.data.clean import CleanStage, DataValidationError


def valid_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "UDI": [1, 2],
            "Product ID": ["M14860", "L47181"],
            "Type": ["M", "L"],
            "Air temperature [K]": [298.1, 298.2],
            "Process temperature [K]": [308.6, 308.7],
            "Rotational speed [rpm]": [1551, 1408],
            "Torque [Nm]": [42.8, 46.3],
            "Tool wear [min]": [0, 3],
            "Machine failure": [0, 1],
            "TWF": [0, 0],
            "HDF": [0, 1],
            "PWF": [0, 0],
            "OSF": [0, 0],
            "RNF": [0, 0],
        }
    )


def test_valid_frame_passes_through_with_every_column_kept() -> None:
    frame = valid_frame()
    assert CleanStage().run(frame).equals(frame)


def test_bom_is_stripped_from_the_first_column_name() -> None:
    frame = valid_frame().rename({"UDI": "\ufeffUDI"})
    cleaned = CleanStage().run(frame)
    assert cleaned.columns[0] == "UDI"
    assert cleaned.equals(valid_frame())


def test_missing_required_column_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="Torque"):
        CleanStage().run(valid_frame().drop("Torque [Nm]"))


def test_non_numeric_sensor_is_rejected() -> None:
    frame = valid_frame().with_columns(pl.col("Torque [Nm]").cast(pl.String))
    with pytest.raises(DataValidationError, match="Torque"):
        CleanStage().run(frame)


def test_nulls_are_rejected() -> None:
    frame = valid_frame().with_columns(pl.Series("Torque [Nm]", [42.8, None]))
    with pytest.raises(DataValidationError, match="null"):
        CleanStage().run(frame)


def test_duplicate_udi_is_rejected() -> None:
    frame = valid_frame().with_columns(pl.Series("UDI", [1, 1]))
    with pytest.raises(DataValidationError, match="UDI"):
        CleanStage().run(frame)


def test_unknown_type_level_is_rejected() -> None:
    frame = valid_frame().with_columns(pl.Series("Type", ["M", "X"]))
    with pytest.raises(DataValidationError, match="Type"):
        CleanStage().run(frame)


@pytest.mark.parametrize("column", ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"])
def test_non_binary_target_or_failure_mode_is_rejected(column: str) -> None:
    frame = valid_frame().with_columns(pl.Series(column, [0, 2]))
    with pytest.raises(DataValidationError, match=column):
        CleanStage().run(frame)
