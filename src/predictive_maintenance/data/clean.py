"""Validation stage for the raw dataset (ADR 0002 cleaning rules). Drops no columns."""

import polars as pl

from predictive_maintenance.data import schema

_BOM = "\ufeff"
_BINARY_COLUMNS = (schema.TARGET, *schema.FAILURE_MODES)


class DataValidationError(ValueError):
    """Raised when the raw dataset violates the expected schema or its integrity rules."""


class CleanStage:
    """Strip the BOM from the first column name and validate the dataset.

    Validates required columns and dtypes, absence of nulls, unique ``UDI``, known ``Type``
    levels and binary target/failure-mode flags. Never drops or alters data.
    """

    def run(self, data: pl.DataFrame) -> pl.DataFrame:
        """Return ``data`` with a clean first column name, or raise ``DataValidationError``."""
        frame = _strip_bom(data)
        _check_columns(frame)
        _check_no_nulls(frame)
        _check_unique_udi(frame)
        _check_type_levels(frame)
        _check_binary_flags(frame)
        return frame


def _strip_bom(frame: pl.DataFrame) -> pl.DataFrame:
    first = frame.columns[0] if frame.columns else ""
    if first.startswith(_BOM):
        return frame.rename({first: first.removeprefix(_BOM)})
    return frame


def _check_columns(frame: pl.DataFrame) -> None:
    missing = [name for name in schema.REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise DataValidationError(f"missing required columns: {missing}")
    string_columns = (schema.CATEGORICAL_FEATURE, "Product ID")
    integer_columns = ("UDI", *_BINARY_COLUMNS)
    for name in schema.REQUIRED_COLUMNS:
        dtype = frame.schema[name]
        if name in string_columns:
            ok = dtype == pl.String
        elif name in integer_columns:
            ok = dtype.is_integer()
        else:
            ok = dtype.is_numeric()
        if not ok:
            raise DataValidationError(f"column {name!r} has unexpected dtype {dtype}")


def _check_no_nulls(frame: pl.DataFrame) -> None:
    counts = frame.null_count().row(0, named=True)
    with_nulls = {name: n for name, n in counts.items() if n}
    if with_nulls:
        raise DataValidationError(f"null values found: {with_nulls}")


def _check_unique_udi(frame: pl.DataFrame) -> None:
    if frame["UDI"].n_unique() != frame.height:
        raise DataValidationError("duplicate UDI values found")


def _check_type_levels(frame: pl.DataFrame) -> None:
    unknown = set(frame[schema.CATEGORICAL_FEATURE].unique()) - set(schema.TYPE_LEVELS)
    if unknown:
        raise DataValidationError(
            f"column 'Type' has unknown levels {sorted(unknown)}; expected {schema.TYPE_LEVELS}"
        )


def _check_binary_flags(frame: pl.DataFrame) -> None:
    for name in _BINARY_COLUMNS:
        if not set(frame[name].unique()) <= {0, 1}:
            raise DataValidationError(f"column {name!r} must contain only 0 and 1")
