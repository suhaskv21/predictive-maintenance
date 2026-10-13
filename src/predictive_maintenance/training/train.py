"""Train the default model and write its verified five-file artifact unit (ADR 0009, ADR 0012).

``just train`` runs ``main``: it fits the default candidate on the train partition only, with the
seed and settings from ``configs/default.toml``, re-derives the Operating threshold by the ADR 0007
method (one threshold per CV repeat, median across repeats) and fails if it differs from the
accepted figure by more than the configured tolerance. The Holdout is discarded at the split and
never read here. The unit's layout and formats are in ``training.unit``.
"""

from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from predictive_maintenance.config import PROJECT_ROOT, Config, load_config
from predictive_maintenance.data import schema
from predictive_maintenance.data.ingest import load_dataset
from predictive_maintenance.evaluation.compare import cv_folds, score_readings
from predictive_maintenance.evaluation.split import repeated_stratified_cv, stratified_holdout
from predictive_maintenance.evaluation.threshold import (
    OperatingPoint,
    median_threshold,
    repeat_scores,
    risk_floor,
)
from predictive_maintenance.training.candidates import build_candidates
from predictive_maintenance.training.unit import (
    ArtifactUnit,
    build_operating_metrics,
    reject_policy,
    write_unit,
)

DEFAULT_MODEL = "random_forest_unweighted"
MODELS_DIR = PROJECT_ROOT / "models"
THRESHOLD_PLACES = 3  # the precision ADR 0007 quotes its thresholds at


class ThresholdDriftError(RuntimeError):
    """The re-derived Operating threshold is not within tolerance of the accepted figure."""


def check_threshold(config: Config, model: str, derived: float) -> None:
    """Fail unless ``derived`` is within tolerance of the accepted threshold of ``model``.

    The comparison uses ``derived`` rounded to 3 places, the precision the ADR quotes.

    Raises:
        ThresholdDriftError: if ``[deploy]`` has no accepted threshold for ``model`` or the
            difference exceeds ``threshold_tolerance``.
    """
    accepted = config.deploy.accepted_thresholds.get(model)
    if accepted is None:
        raise ThresholdDriftError(f"no accepted threshold for {model!r} in [deploy]")
    tolerance = config.deploy.threshold_tolerance
    if abs(round(derived, THRESHOLD_PLACES) - accepted) > tolerance:
        raise ThresholdDriftError(
            f"{model}: re-derived Operating threshold {derived:.4f} differs from the accepted "
            f"{accepted} by more than {tolerance}; a model or library version has moved"
        )


def golden_rows(scores: NDArray[np.float64], threshold: float) -> list[int]:
    """Row positions of fixed Readings that span the score range, in row order.

    The lowest, the median-ranked and the highest score, plus the nearest score on each side of
    ``threshold``, so a changed model or preprocessing step moves at least one stored score. The
    choice depends only on ``scores`` (ties go to the first row), so it is the same on every run.
    """
    below = np.flatnonzero(scores < threshold)
    above = np.flatnonzero(scores >= threshold)
    median_rank = int(np.argsort(scores, kind="stable")[len(scores) // 2])
    picks = {int(np.argmin(scores)), int(np.argmax(scores)), median_rank}
    if below.size:
        picks.add(int(below[np.argmax(scores[below])]))
    if above.size:
        picks.add(int(above[np.argmin(scores[above])]))
    return sorted(picks)


def train_unit(
    readings: pl.DataFrame, config: Config, models_dir: Path, *, model: str = DEFAULT_MODEL
) -> ArtifactUnit:
    """Fit ``model`` on the train partition of ``readings`` and write its five-file unit.

    Nothing is written if the re-derived Operating threshold is outside tolerance.

    Raises:
        ThresholdDriftError: if the threshold at the default Cost ratio differs from the accepted
            figure in ``[deploy]`` by more than ``threshold_tolerance``, or none is accepted.
        KeyError: if ``model`` is not a known candidate.
    """
    train, _ = stratified_holdout(readings, test_size=config.split.test_size, seed=config.seed)
    candidate = build_candidates(config.seed)[model]
    y = train[schema.TARGET].to_numpy()
    folds = cv_folds(
        train,
        repeated_stratified_cv(
            n_splits=config.split.cv_folds, n_repeats=config.split.cv_repeats, seed=config.seed
        ),
    )
    out_of_fold = repeat_scores(candidate, train, folds, n_splits=config.split.cv_folds)
    thresholds = {
        ratio: median_threshold(y, out_of_fold, ratio) for ratio in config.threshold.cost_ratios
    }
    default_ratio = config.threshold.default_cost_ratio
    check_threshold(config, model, thresholds[default_ratio])

    x = train.select(schema.FEATURE_COLUMNS)
    pipeline = candidate.make_pipeline().fit(x, y)
    golden = x[golden_rows(score_readings(pipeline, x), thresholds[default_ratio])]
    return write_unit(
        models_dir,
        pipeline=pipeline,
        point=OperatingPoint(
            model=model,
            default_cost_ratio=default_ratio,
            thresholds=thresholds,
            bands=None,
            risk_floor=risk_floor(y, out_of_fold),
        ),
        policy=reject_policy(model, default_ratio),
        metrics=build_operating_metrics(model, y, out_of_fold, thresholds, train),
        golden=golden,
        golden_scores=score_readings(pipeline, golden).tolist(),
    )


def main() -> None:
    """Train the default model from ``configs/default.toml`` and write its unit to ``models/``."""
    unit = train_unit(load_dataset(), load_config(), MODELS_DIR)
    point = OperatingPoint.from_json(unit.operating_point.read_text())
    print(
        f"{unit.model}: Operating threshold {point.threshold:.4f} at {point.default_cost_ratio}:1"
    )
    for path in (*unit.sidecars, unit.manifest):
        print(f"  wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
