import dataclasses
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import pytest

from predictive_maintenance.config import (
    AppConfig,
    Config,
    DeployConfig,
    EvaluationConfig,
    FailureModesConfig,
    SplitConfig,
    ThresholdConfig,
    load_config,
)
from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.missingness import MissingDataPolicy
from predictive_maintenance.evaluation.split import stratified_holdout
from predictive_maintenance.evaluation.threshold import OperatingPoint
from predictive_maintenance.training import train as train_module
from predictive_maintenance.training.train import (
    DEFAULT_MODEL,
    ThresholdDriftError,
    golden_rows,
    train_unit,
)
from predictive_maintenance.training.unit import LIBRARIES, file_sha256, unit_paths

SEED = 3


def fixture_readings(n: int = 400, failures: int = 40) -> pl.DataFrame:
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


def config(*, accepted: float = 0.5, tolerance: float = 1.0) -> Config:
    """A small, fast run configuration; the real one is covered in test_config."""
    real = load_config()
    return Config(
        seed=SEED,
        split=SplitConfig(test_size=0.3, cv_folds=3, cv_repeats=2),
        threshold=ThresholdConfig(default_cost_ratio=10.0, cost_ratios=(5.0, 10.0, 20.0)),
        evaluation=EvaluationConfig(bootstrap_resamples=10),
        missing_data=real.missing_data,
        pipeline_health=real.pipeline_health,
        failure_modes=FailureModesConfig(min_pr_auc=0.3, min_lift=5.0),
        app=AppConfig(example_count=10, example_failures=3),
        deploy=DeployConfig(
            accepted_thresholds={DEFAULT_MODEL: accepted},
            threshold_tolerance=tolerance,
            site_budget_mb=100.0,
        ),
        plausibility=real.plausibility,
    )


@pytest.fixture(scope="module")
def unit_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models")
    train_unit(fixture_readings(), config(), path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def operating_point(directory: Path) -> OperatingPoint:
    text = unit_paths(directory, DEFAULT_MODEL).operating_point.read_text()
    return OperatingPoint.from_json(text)


def test_training_writes_exactly_the_five_files_of_the_unit(unit_dir: Path) -> None:
    unit = unit_paths(unit_dir, DEFAULT_MODEL)
    assert sorted(p.name for p in unit_dir.iterdir()) == sorted(
        p.name for p in (*unit.sidecars, unit.manifest)
    )
    assert len(list(unit_dir.iterdir())) == 5


def test_the_manifest_names_the_model_the_versions_and_the_hash_of_the_other_four(
    unit_dir: Path,
) -> None:
    unit = unit_paths(unit_dir, DEFAULT_MODEL)
    manifest = read_json(unit.manifest)
    assert manifest["model"] == DEFAULT_MODEL
    assert manifest["libraries"] == {name: version(name) for name in LIBRARIES}
    assert manifest["files"] == {p.name: file_sha256(p) for p in unit.sidecars}
    assert len(manifest["files"]) == 4


def test_the_pipeline_bundles_preprocessing_and_model_and_takes_only_the_six_inputs(
    unit_dir: Path,
) -> None:
    pipeline = joblib.load(unit_paths(unit_dir, DEFAULT_MODEL).pipeline)
    assert [name for name, _ in pipeline.steps][-1] == "model"
    assert list(pipeline.feature_names_in_) == list(schema.FEATURE_COLUMNS)
    banned = {*schema.FAILURE_MODES, *schema.IDENTIFIER_COLUMNS, schema.TARGET}
    assert banned.isdisjoint(pipeline.feature_names_in_)
    scores = pipeline.predict_proba(fixture_readings().select(schema.FEATURE_COLUMNS))[:, 1]
    assert ((scores >= 0) & (scores <= 1)).all()


def test_the_operating_point_has_the_default_ratio_a_threshold_per_ratio_and_no_bands(
    unit_dir: Path,
) -> None:
    point = operating_point(unit_dir)
    assert point.model == DEFAULT_MODEL
    assert point.default_cost_ratio == 10.0
    assert set(point.thresholds) == {5.0, 10.0, 20.0}
    assert all(0 < t < 1 for t in point.thresholds.values())
    assert point.bands is None
    assert 0.01 <= point.risk_floor < 1


def test_the_missing_data_policy_rejects_with_an_empty_reference_not_measured_in_this_build(
    unit_dir: Path,
) -> None:
    path = unit_paths(unit_dir, DEFAULT_MODEL).missing
    policy = MissingDataPolicy.from_json(path.read_text())
    assert (policy.model, policy.strategy) == (DEFAULT_MODEL, "reject")
    assert policy.scorable == () and policy.measured == () and policy.measured_strategy is None
    assert "not measured in this build" in read_json(path)["reference_note"]


def test_the_operating_metrics_are_cv_figures_per_ratio_with_the_training_ranges(
    unit_dir: Path,
) -> None:
    metrics = read_json(unit_paths(unit_dir, DEFAULT_MODEL).operating_metrics)
    assert metrics["model"] == DEFAULT_MODEL
    assert metrics["source"] == "cv_train"
    assert set(metrics["ratios"]) == {"5.0", "10.0", "20.0"}
    for figures in metrics["ratios"].values():
        assert {"recall", "precision", "flags_per_1000", "false_alarms_per_caught"} <= set(figures)
    assert set(metrics["training_ranges"]) == set(schema.NUMERIC_FEATURES)
    train, _ = stratified_holdout(fixture_readings(), test_size=0.3, seed=SEED)
    air = train["Air temperature [K]"]
    assert metrics["training_ranges"]["Air temperature [K]"] == {"min": air.min(), "max": air.max()}


def test_the_golden_readings_are_train_rows_whose_stored_scores_span_the_threshold(
    unit_dir: Path,
) -> None:
    unit = unit_paths(unit_dir, DEFAULT_MODEL)
    block = read_json(unit.manifest)["golden"]
    golden = pl.DataFrame(block["readings"]).select(schema.FEATURE_COLUMNS)
    scores = block["scores"]
    assert joblib.load(unit.pipeline).predict_proba(golden)[:, 1].tolist() == scores
    train, holdout = stratified_holdout(fixture_readings(), test_size=0.3, seed=SEED)
    columns = list(schema.FEATURE_COLUMNS)
    assert golden.join(train.select(columns), on=columns).height == golden.height
    assert golden.join(holdout.select(columns), on=columns).height == 0
    threshold = operating_point(unit_dir).threshold
    assert len(set(scores)) > 1
    assert min(scores) < threshold <= max(scores)


def test_golden_rows_pick_the_extremes_the_median_and_both_sides_of_the_threshold() -> None:
    scores = np.array([0.5, 0.0, 0.9, 0.2, 0.3, 1.0, 0.1])
    assert golden_rows(scores, threshold=0.25) == [1, 3, 4, 5]


def test_golden_rows_tolerate_a_threshold_outside_the_score_range() -> None:
    scores = np.array([0.4, 0.1, 0.4])
    assert golden_rows(scores, threshold=0.9) == [0, 1]
    assert golden_rows(scores, threshold=0.0) == [0, 1]


def assert_same_figures(first: Path, second: Path) -> None:
    for part in ("operating_point", "operating_metrics", "missing"):
        one, other = (getattr(unit_paths(d, DEFAULT_MODEL), part) for d in (first, second))
        assert one.read_text() == other.read_text()


def test_the_examples_are_train_rows_never_holdout_rows_with_the_configured_failures(
    unit_dir: Path,
) -> None:
    rows = read_json(unit_paths(unit_dir, DEFAULT_MODEL).manifest)["examples"]
    examples = pl.DataFrame(rows)
    assert examples.columns == [*schema.FEATURE_COLUMNS, schema.TARGET]
    assert examples.height == 10
    assert examples[schema.TARGET].sum() == 3
    train, holdout = stratified_holdout(fixture_readings(), test_size=0.3, seed=SEED)
    columns = [*schema.FEATURE_COLUMNS, schema.TARGET]
    assert examples.join(train.select(columns), on=columns).height == examples.height
    assert examples.join(holdout.select(columns), on=columns).height == 0


def test_training_fails_loudly_when_the_train_partition_has_too_few_failures(
    tmp_path: Path,
) -> None:
    few = fixture_readings(failures=2)
    with pytest.raises(ValueError, match="failures"):
        train_unit(few, config(), tmp_path)
    assert not list(tmp_path.iterdir())


def test_training_is_reproducible_for_the_seed(unit_dir: Path, tmp_path: Path) -> None:
    train_unit(fixture_readings(), config(), tmp_path)
    assert_same_figures(tmp_path, unit_dir)
    first, second = (
        read_json(unit_paths(d, DEFAULT_MODEL).manifest)["golden"] for d in (tmp_path, unit_dir)
    )
    assert first == second
    first_examples, second_examples = (
        read_json(unit_paths(d, DEFAULT_MODEL).manifest)["examples"] for d in (tmp_path, unit_dir)
    )
    assert first_examples == second_examples


class PoisonedHoldout:
    """Stands in for the Holdout partition: any use of it fails the test."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the Holdout was read: {name}")

    def __len__(self) -> int:
        raise AssertionError("the Holdout was read: len")

    def __iter__(self) -> Any:
        raise AssertionError("the Holdout was read: iter")

    def __getitem__(self, key: object) -> Any:
        raise AssertionError("the Holdout was read: getitem")


def test_training_never_reads_the_holdout_partition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def split(readings: pl.DataFrame, *, test_size: float, seed: int) -> tuple[Any, Any]:
        train, _ = stratified_holdout(readings, test_size=test_size, seed=seed)
        return train, PoisonedHoldout()

    monkeypatch.setattr(train_module, "stratified_holdout", split)
    train_unit(fixture_readings(), config(), tmp_path)
    assert unit_paths(tmp_path, DEFAULT_MODEL).manifest.exists()


def test_the_unit_does_not_depend_on_the_holdout_rows(unit_dir: Path, tmp_path: Path) -> None:
    readings = fixture_readings()
    _, holdout = stratified_holdout(readings, test_size=0.3, seed=SEED)
    # Wreck a feature of every Holdout row; labels stay, so the split is the same.
    wrecked = readings.with_columns(
        pl.when(pl.col("UDI").is_in(holdout["UDI"].implode()))
        .then(pl.col("Torque [Nm]") * -50)
        .otherwise(pl.col("Torque [Nm]"))
        .alias("Torque [Nm]")
    )
    train_unit(wrecked, config(), tmp_path)
    assert_same_figures(tmp_path, unit_dir)


def test_training_passes_when_the_threshold_is_within_tolerance_of_the_accepted_figure(
    unit_dir: Path, tmp_path: Path
) -> None:
    accepted = round(operating_point(unit_dir).threshold, 3)
    train_unit(fixture_readings(), config(accepted=accepted, tolerance=0.0005), tmp_path)
    assert unit_paths(tmp_path, DEFAULT_MODEL).manifest.exists()


def test_training_fails_and_writes_nothing_when_the_threshold_drifts_beyond_tolerance(
    unit_dir: Path, tmp_path: Path
) -> None:
    accepted = round(operating_point(unit_dir).threshold, 3) + 0.01
    target = tmp_path / "models"
    with pytest.raises(ThresholdDriftError, match="0.0005"):
        train_unit(fixture_readings(), config(accepted=accepted, tolerance=0.0005), target)
    assert not target.exists() or not list(target.iterdir())


def test_training_fails_for_a_model_without_an_accepted_threshold(tmp_path: Path) -> None:
    base = config()
    unlisted = dataclasses.replace(
        base, deploy=dataclasses.replace(base.deploy, accepted_thresholds={"other": 0.1})
    )
    with pytest.raises(ThresholdDriftError, match=DEFAULT_MODEL):
        train_unit(fixture_readings(), unlisted, tmp_path)
