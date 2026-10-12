from pathlib import Path

import polars as pl
import pytest

from predictive_maintenance.data.clean import DataValidationError
from predictive_maintenance.data.ingest import DEFAULT_RAW_PATH, load_dataset

HEADER = (
    "UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],"
    "Torque [Nm],Tool wear [min],Machine failure,TWF,HDF,PWF,OSF,RNF"
)


def write_csv(path: Path, rows: list[str], bom: bool = True) -> Path:
    prefix = "\ufeff" if bom else ""
    path.write_text(prefix + "\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_load_dataset_reads_a_bom_prefixed_csv(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "data.csv",
        [
            "1,M14860,M,298.1,308.6,1551,42.8,0,0,0,0,0,0,0",
            "2,L47181,L,298.2,308.7,1408,46.3,3,1,0,1,0,0,0",
        ],
    )
    frame = load_dataset(path)
    assert frame.columns[0] == "UDI"
    assert frame.height == 2
    assert frame["Machine failure"].to_list() == [0, 1]


def test_load_dataset_rejects_invalid_data(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "data.csv",
        [
            "1,M14860,M,298.1,308.6,1551,42.8,0,0,0,0,0,0,0",
            "1,L47181,L,298.2,308.7,1408,46.3,3,1,0,1,0,0,0",
        ],
    )
    with pytest.raises(DataValidationError, match="UDI"):
        load_dataset(path)


def test_load_dataset_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "absent.csv")


def test_supplied_dataset_matches_the_established_facts() -> None:
    frame = load_dataset(DEFAULT_RAW_PATH)
    assert frame.shape == (10_000, 14)
    assert frame["Machine failure"].sum() == 339
    assert frame.select(pl.col("Product ID").n_unique()).item() == 10_000
    assert frame.select(pl.col("TWF", "HDF", "PWF", "OSF", "RNF").sum()).row(0) == (
        46,
        115,
        95,
        98,
        19,
    )
