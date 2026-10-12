"""Load the raw AI4I 2020 CSV and validate it through the data-prep pipeline (ADR 0002)."""

from pathlib import Path

import polars as pl

from predictive_maintenance.config import PROJECT_ROOT
from predictive_maintenance.data.clean import CleanStage
from predictive_maintenance.pipeline.core import Pipeline

DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


def load_dataset(path: Path = DEFAULT_RAW_PATH) -> pl.DataFrame:
    """Read the raw CSV and run it through the validation pipeline.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        DataValidationError: if the data violates the schema or integrity rules.
    """
    raw = pl.read_csv(path)
    return Pipeline([CleanStage()]).run(raw)
