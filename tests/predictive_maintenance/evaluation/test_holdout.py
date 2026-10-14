import dataclasses
import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import pytest

from predictive_maintenance.config import Config, EvaluationConfig, load_config
from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.compare import derived_seeds, score_readings
from predictive_maintenance.evaluation.holdout import (
    HoldoutAlreadyScoredError,
    HoldoutMismatchError,
    apply_report,
    check_report,
    score_holdout,
)
from predictive_maintenance.evaluation.split import stratified_holdout
from predictive_maintenance.evaluation.threshold import (
    BOOTSTRAP_STREAM,
    OperatingPoint,
    bootstrap_metric_cis,
)
from predictive_maintenance.inference.artifact import LoadedUnit, load_unit
from predictive_maintenance.inference.assessment import assess
from predictive_maintenance.inference.presentation import present
from predictive_maintenance.training.train import DEFAULT_MODEL, train_unit
from predictive_maintenance.training.unit import file_sha256, unit_paths

SEED = 3
RESAMPLES = 30
METRICS = {
    "recall",
    "precision",
    "expected_cost_per_1000",
    "flags_per_1000",
    "false_alarms_per_caught",
    "pr_auc",
    "roc_auc",
    "brier",
}


def fixture_readings(n: int = 600, failures: int = 60) -> pl.DataFrame:
    """A small AI4I-shaped dataset whose failures have high torque, with all raw columns."""
    rng = np.random.default_rng(0)
    torque = rng.normal(40, 10, n)
    target = np.zeros(n, dtype=int)
    target[np.argsort(-(torque + rng.normal(0, 4, n)))[:failures]] = 1
    modes = {mode: rng.integers(0, 2, n) for mode in schema.FAILURE_MODES}
    return pl.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"M{i}" for i in range(n)],
            "Type": rng.choice(list(schema.TYPE_LEVELS), n),
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 1, n),
            "Rotational speed [rpm]": rng.normal(1500, 100, n),
            "Torque [Nm]": torque,
            "Tool wear [min]": rng.uniform(0, 250, n),
            schema.TARGET: target,
            **modes,
        }
    )


def config() -> Config:
    """The real configuration with a small, fast CV and a seed of its own."""
    real = load_config()
    return dataclasses.replace(
        real,
        seed=SEED,
        split=dataclasses.replace(real.split, cv_folds=3, cv_repeats=2),
        evaluation=EvaluationConfig(bootstrap_resamples=RESAMPLES),
        deploy=dataclasses.replace(
            real.deploy,
            accepted_thresholds={DEFAULT_MODEL: 0.5},
            threshold_tolerance=1.0,
        ),
    )


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models")
    train_unit(fixture_readings(), config(), path)
    return path


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports"


def read_report(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def test_the_report_names_the_pipeline_hash_the_date_and_every_metric_with_its_interval(
    models_dir: Path, reports_dir: Path
) -> None:
    path = score_holdout(fixture_readings(), config(), models_dir, reports_dir)

    assert path == reports_dir / "holdout-metrics.json"
    report = read_report(path)
    assert report["model"] == DEFAULT_MODEL
    assert report["pipeline_sha256"] == file_sha256(unit_paths(models_dir, DEFAULT_MODEL).pipeline)
    assert date.fromisoformat(report["date"]) == date.today()
    assert report["cost_ratio"] == 10.0
    assert report["resamples"] == RESAMPLES
    assert report["interval_level"] == 0.95
    assert set(report["metrics"]) == METRICS
    for figure in report["metrics"].values():
        assert figure["low"] <= figure["estimate"] <= figure["high"]


def test_the_figures_are_the_frozen_threshold_scored_on_the_holdout_with_the_adr_seed_stream(
    models_dir: Path, reports_dir: Path
) -> None:
    report = read_report(score_holdout(fixture_readings(), config(), models_dir, reports_dir))

    unit = unit_paths(models_dir, DEFAULT_MODEL)
    point = OperatingPoint.from_json(unit.operating_point.read_text())
    _, holdout = stratified_holdout(
        fixture_readings(), test_size=config().split.test_size, seed=SEED
    )
    scores = score_readings(joblib.load(unit.pipeline), holdout.select(schema.FEATURE_COLUMNS))
    expected = bootstrap_metric_cis(
        holdout[schema.TARGET].to_numpy(),
        scores,
        threshold=point.threshold,
        cost_ratio=point.default_cost_ratio,
        n_resamples=RESAMPLES,
        seed=derived_seeds(SEED, BOOTSTRAP_STREAM, 1)[0],
    )
    assert report["threshold"] == point.threshold
    for row in expected.iter_rows(named=True):
        assert report["metrics"][row["metric"]] == {
            "estimate": row["estimate"],
            "low": row["low"],
            "high": row["high"],
        }


def test_a_second_run_refuses_and_says_a_rescore_needs_a_new_adr(
    models_dir: Path, reports_dir: Path
) -> None:
    path = score_holdout(fixture_readings(), config(), models_dir, reports_dir)
    first = path.read_text()

    with pytest.raises(HoldoutAlreadyScoredError, match="new ADR"):
        score_holdout(fixture_readings(), config(), models_dir, reports_dir)
    assert path.read_text() == first


def test_the_explicit_flag_allows_a_rescore(models_dir: Path, reports_dir: Path) -> None:
    path = score_holdout(fixture_readings(), config(), models_dir, reports_dir)
    path.write_text("{}")

    score_holdout(fixture_readings(), config(), models_dir, reports_dir, rescore=True)

    assert "pipeline_sha256" in read_report(path)


def test_the_check_passes_for_the_pipeline_that_was_scored(
    models_dir: Path, reports_dir: Path
) -> None:
    path = score_holdout(fixture_readings(), config(), models_dir, reports_dir)

    check_report(path, unit_paths(models_dir, DEFAULT_MODEL).pipeline)


def test_the_check_fails_for_a_different_pipeline(
    models_dir: Path, reports_dir: Path, tmp_path: Path
) -> None:
    path = score_holdout(fixture_readings(), config(), models_dir, reports_dir)
    other = tmp_path / "other"
    train_unit(fixture_readings(), dataclasses.replace(config(), seed=SEED + 1), other)

    with pytest.raises(HoldoutMismatchError, match="holdout figures belong to a different model"):
        check_report(path, unit_paths(other, DEFAULT_MODEL).pipeline)


def test_the_check_fails_when_there_is_no_report(models_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check_report(tmp_path / "missing.json", unit_paths(models_dir, DEFAULT_MODEL).pipeline)


def read_json(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


@pytest.fixture
def scored(tmp_path: Path) -> tuple[Path, Path]:
    """A fresh copy of the unit and the Holdout report written for it."""
    copy = tmp_path / "unit"
    train_unit(fixture_readings(), config(), copy)
    return copy, score_holdout(fixture_readings(), config(), copy, tmp_path / "reports")


def test_applying_the_report_labels_the_forest_figures_holdout_with_intervals(
    scored: tuple[Path, Path],
) -> None:
    unit_dir, report_path = scored
    unit = unit_paths(unit_dir, DEFAULT_MODEL)
    before = read_json(unit.operating_metrics)

    apply_report(report_path, unit_dir)

    after = read_json(unit.operating_metrics)
    report = read_json(report_path)
    assert after["source"] == "holdout"
    assert list(after["ratios"]) == ["10.0"]
    figures = after["ratios"]["10.0"]
    for name in ("recall", "precision", "flags_per_1000", "false_alarms_per_caught"):
        assert figures[name] == report["metrics"][name]["estimate"]
        assert figures["intervals"][name] == {
            "low": report["metrics"][name]["low"],
            "high": report["metrics"][name]["high"],
        }
    assert after["holdout"]["date"] == report["date"]
    assert after["training_ranges"] == before["training_ranges"]
    assert after["training_types"] == before["training_types"]


def test_the_cv_figures_stay_only_in_the_sensitivity_data(scored: tuple[Path, Path]) -> None:
    unit_dir, report_path = scored
    unit = unit_paths(unit_dir, DEFAULT_MODEL)
    before = read_json(unit.operating_metrics)

    apply_report(report_path, unit_dir)

    after = read_json(unit.operating_metrics)
    assert after["sensitivity"] == {"source": "cv_train", "ratios": before["ratios"]}
    assert "cv_train" not in json.dumps({k: v for k, v in after.items() if k != "sensitivity"})


def test_an_applied_unit_still_loads_and_the_flag_meaning_quotes_the_holdout(
    scored: tuple[Path, Path],
) -> None:
    unit_dir, report_path = scored
    pipeline = unit_paths(unit_dir, DEFAULT_MODEL).pipeline
    hash_before = file_sha256(pipeline)

    apply_report(report_path, unit_dir)

    loaded = load_unit(unit_dir, DEFAULT_MODEL)
    assert isinstance(loaded, LoadedUnit)
    assert file_sha256(pipeline) == hash_before
    check_report(report_path, pipeline)
    reading = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 10,
    }
    shown = present(
        assess(loaded, reading, load_config().plausibility),
        loaded.operating_point,
        loaded.operating_metrics,
        cost_ratio=10.0,
    )
    assert shown.meaning[-1] == "Figures from held-out test data."


def test_applying_is_idempotent(scored: tuple[Path, Path]) -> None:
    unit_dir, report_path = scored
    unit = unit_paths(unit_dir, DEFAULT_MODEL)
    apply_report(report_path, unit_dir)
    first = unit.operating_metrics.read_text()

    apply_report(report_path, unit_dir)

    assert unit.operating_metrics.read_text() == first
    assert isinstance(load_unit(unit_dir, DEFAULT_MODEL), LoadedUnit)


def test_applying_a_report_of_a_different_model_fails_and_changes_nothing(
    scored: tuple[Path, Path], tmp_path: Path
) -> None:
    unit_dir, _ = scored
    other = tmp_path / "other"
    train_unit(fixture_readings(), dataclasses.replace(config(), seed=SEED + 1), other)
    other_report = score_holdout(fixture_readings(), config(), other, tmp_path / "other-reports")
    unit = unit_paths(unit_dir, DEFAULT_MODEL)
    before = (unit.operating_metrics.read_text(), unit.manifest.read_text())

    with pytest.raises(HoldoutMismatchError, match="holdout figures belong to a different model"):
        apply_report(other_report, unit_dir)

    assert (unit.operating_metrics.read_text(), unit.manifest.read_text()) == before
