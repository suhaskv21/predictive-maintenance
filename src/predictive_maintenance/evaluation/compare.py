"""Model comparison on the train partition only (ADR 0006): scores, selection rule, side checks.

Nothing here sees the holdout: every function takes the train partition (and folds indexing it).
Folds are materialised once so every candidate is scored on identical folds and deltas are paired.
"""

import io
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

import joblib
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline

from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.split import stratified_holdout
from predictive_maintenance.training.candidates import Candidate, variant_names

Fold = tuple[NDArray[np.intp], NDArray[np.intp]]

RAW = "raw"
MARGIN = 0.02  # smallest PR-AUC gain that counts as a real difference (judgement call, ADR 0006)
CALIBRATION_MIN_RELATIVE_GAIN = 0.10
MAX_LATENCY_MS = 100.0
MAX_ARTIFACT_MB = 50.0
LEARNING_CURVE_TRAIN_FRACTION = 0.70 / 0.80  # the 80/20 -> 70/30 step of ADR 0005
LEARNING_CURVE_MAX_PENALTY = 0.01


def derived_seeds(seed: int, stream: int, n: int) -> list[int]:
    """``n`` reproducible seeds derived from the single config seed, one ``stream`` per purpose."""
    return [int(s) for s in np.random.SeedSequence([seed, stream]).generate_state(n)]


def cv_folds(train: pl.DataFrame, cv: Any) -> list[Fold]:
    """Materialise the splits of ``cv`` over the train partition so all candidates share them."""
    x = train.select(schema.FEATURE_COLUMNS)
    return [(a, b) for a, b in cv.split(x, train[schema.TARGET].to_numpy())]


def score_readings(pipeline: Pipeline, x: pl.DataFrame) -> NDArray[np.float64]:
    """Failure probability per Reading if the model has one, else its anomaly score."""
    if hasattr(pipeline, "predict_proba"):
        scores: NDArray[np.float64] = pipeline.predict_proba(x)[:, 1]
    else:
        scores = pipeline.decision_function(x)
    return scores


def _check_folds(train: pl.DataFrame, folds: Sequence[Fold]) -> None:
    if not folds:
        raise ValueError("folds must not be empty")
    if max(int(max(a.max(), b.max())) for a, b in folds) >= train.height:
        raise ValueError("folds index rows outside the train partition passed in")


def _evaluate_fold(
    candidate: Candidate,
    variant: str | None,
    x: pl.DataFrame,
    y: NDArray[np.int_],
    fold: int,
    train_idx: NDArray[np.intp],
    test_idx: NDArray[np.intp],
) -> tuple[str, str, int, float, float | None, float]:
    pipeline = candidate.make_pipeline(variant)
    started = time.perf_counter()
    pipeline.fit(x[train_idx], y[train_idx])
    fit_seconds = time.perf_counter() - started
    scores = score_readings(pipeline, x[test_idx])
    brier = (
        float(brier_score_loss(y[test_idx], scores)) if hasattr(pipeline, "predict_proba") else None
    )
    pr_auc = float(average_precision_score(y[test_idx], scores))
    return candidate.name, variant or RAW, fold, pr_auc, brier, fit_seconds


def compare_candidates(
    candidates: Iterable[Candidate],
    train: pl.DataFrame,
    folds: Sequence[Fold],
    *,
    variants: Sequence[str | None] = (None,),
    n_jobs: int = -1,
) -> pl.DataFrame:
    """Score every candidate (and calibration variant) on every fold of the train partition.

    Preprocessing is fitted inside each fold. Columns: ``candidate``, ``variant`` ("raw" or a
    calibration method), ``fold``, ``pr_auc``, ``brier`` (null for score-only models) and
    ``fit_seconds``.

    Raises:
        ValueError: if ``folds`` is empty or indexes rows outside ``train``.
    """
    _check_folds(train, folds)
    x = train.select(schema.FEATURE_COLUMNS)
    y = train[schema.TARGET].to_numpy()
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_fold)(candidate, variant, x, y, i, a, b)
        for candidate in candidates
        for variant in variants
        for i, (a, b) in enumerate(folds)
    )
    return pl.DataFrame(
        rows,
        schema={
            "candidate": pl.String,
            "variant": pl.String,
            "fold": pl.Int64,
            "pr_auc": pl.Float64,
            "brier": pl.Float64,
            "fit_seconds": pl.Float64,
        },
        orient="row",
    )


def paired_deltas(
    scores: pl.DataFrame,
    challenger: str,
    incumbent: str,
    *,
    metric: str = "pr_auc",
    variant: str = RAW,
) -> NDArray[np.float64]:
    """Per-fold ``metric`` of ``challenger`` minus ``incumbent``, on identical folds."""

    def column(name: str) -> NDArray[np.float64]:
        rows = scores.filter((pl.col("candidate") == name) & (pl.col("variant") == variant))
        return rows.sort("fold")[metric].to_numpy()

    return column(challenger) - column(incumbent)


def corrected_se(deltas: NDArray[np.float64], *, n_cv_folds: int) -> float:
    """Nadeau-Bengio corrected standard error of the mean of paired deltas from repeated k-fold CV.

    Overlapping training sets make the naive SE too small; the correction adds the test/train size
    ratio ``1 / (k - 1)`` to ``1 / n``.
    """
    n = len(deltas)
    return float(np.sqrt((1 / n + 1 / (n_cv_folds - 1)) * np.var(deltas, ddof=1)))


@dataclass(frozen=True)
class SelectionStep:
    """One comparison of a more complex challenger against the current incumbent."""

    challenger: str
    incumbent: str
    mean_delta: float
    corrected_se: float
    threshold: float
    replaced: bool


@dataclass(frozen=True)
class Selection:
    """The chosen candidate and the trail of comparisons that led to it."""

    winner: str
    steps: tuple[SelectionStep, ...]


def select_model(
    scores: pl.DataFrame,
    order: Sequence[str],
    *,
    n_cv_folds: int,
    margin: float = MARGIN,
) -> Selection:
    """Pick the simplest model unless a more complex one clearly beats the incumbent.

    Walk ``order`` (simplest first). A challenger replaces the incumbent only if its mean paired
    PR-AUC delta exceeds ``max(margin, corrected_se)``.

    Raises:
        ValueError: if ``order`` is empty.
    """
    if not order:
        raise ValueError("order must name at least one candidate")
    incumbent, steps = order[0], []
    for challenger in order[1:]:
        deltas = paired_deltas(scores, challenger, incumbent)
        se = corrected_se(deltas, n_cv_folds=n_cv_folds)
        threshold = max(margin, se)
        replaced = float(deltas.mean()) > threshold
        steps.append(
            SelectionStep(challenger, incumbent, float(deltas.mean()), se, threshold, replaced)
        )
        if replaced:
            incumbent = challenger
    return Selection(incumbent, tuple(steps))


def select_variants(
    scores: pl.DataFrame,
    families: Sequence[str],
    *,
    n_cv_folds: int,
    margin: float = MARGIN,
) -> dict[str, Selection]:
    """Per family, keep the unweighted model unless class weighting clearly beats it.

    The weighted variant replaces the unweighted one only if its mean paired PR-AUC delta exceeds
    ``max(margin, corrected_se)``, the same rule as ``select_model``.
    """
    return {
        family: select_model(scores, variant_names(family), n_cv_folds=n_cv_folds, margin=margin)
        for family in families
    }


@dataclass(frozen=True)
class CalibrationDecision:
    """Whether to calibrate the winner, and with which method (None = leave uncalibrated)."""

    method: str | None
    relative_brier_gain: dict[str, float]


def calibration_decision(
    scores: pl.DataFrame,
    candidate: str,
    *,
    min_gain: float = CALIBRATION_MIN_RELATIVE_GAIN,
) -> CalibrationDecision:
    """Calibrate if a method cuts mean CV Brier by >= ``min_gain`` relative; sigmoid over isotonic.

    ``scores`` must hold the ``raw``, ``sigmoid`` and ``isotonic`` variants of ``candidate``.
    Isotonic is chosen over sigmoid only if it beats sigmoid by ``min_gain`` as well.
    """
    mean_brier = {
        v: float(
            scores.filter((pl.col("candidate") == candidate) & (pl.col("variant") == v))[
                "brier"
            ].mean()  # type: ignore[arg-type]
        )
        for v in (RAW, "sigmoid", "isotonic")
    }
    gains = {v: 1 - mean_brier[v] / mean_brier[RAW] for v in ("sigmoid", "isotonic")}
    if gains["sigmoid"] < min_gain and gains["isotonic"] < min_gain:
        return CalibrationDecision(None, gains)
    isotonic_over_sigmoid = 1 - mean_brier["isotonic"] / mean_brier["sigmoid"]
    use_isotonic = gains["isotonic"] >= min_gain and isotonic_over_sigmoid >= min_gain
    return CalibrationDecision("isotonic" if use_isotonic else "sigmoid", gains)


def out_of_fold_scores(
    candidate: Candidate,
    train: pl.DataFrame,
    folds: Sequence[Fold],
    *,
    variant: str | None = None,
) -> NDArray[np.float64]:
    """Mean out-of-fold failure probability per train Reading, averaged over the CV repeats."""
    _check_folds(train, folds)
    x = train.select(schema.FEATURE_COLUMNS)
    y = train[schema.TARGET].to_numpy()
    total = np.zeros(train.height)
    count = np.zeros(train.height)
    for a, b in folds:
        pipeline = candidate.make_pipeline(variant).fit(x[a], y[a])
        total[b] += score_readings(pipeline, x[b])
        count[b] += 1
    return total / np.maximum(count, 1)


def reliability_table(
    y_true: NDArray[np.int_], probabilities: NDArray[np.float64], *, n_bins: int = 10
) -> pl.DataFrame:
    """Observed failure rate against mean predicted probability in equal-count probability bins."""
    order = np.argsort(probabilities, kind="stable")
    rows = [
        (i, len(idx), float(probabilities[idx].mean()), float(y_true[idx].mean()))
        for i, idx in enumerate(np.array_split(order, n_bins))
    ]
    return pl.DataFrame(rows, schema=["bin", "n", "mean_predicted", "observed_rate"], orient="row")


@dataclass(frozen=True)
class OperationalCost:
    """Fit time, single-Reading latency and serialized size of a candidate fitted on train data."""

    fit_seconds: float
    latency_ms: float
    artifact_mb: float

    def passes_vetoes(
        self, *, max_latency_ms: float = MAX_LATENCY_MS, max_artifact_mb: float = MAX_ARTIFACT_MB
    ) -> bool:
        """True if latency and artifact size are within their ceilings."""
        return self.latency_ms <= max_latency_ms and self.artifact_mb <= max_artifact_mb


def measure_operational_cost(
    candidate: Candidate, train: pl.DataFrame, *, n_calls: int = 200
) -> OperationalCost:
    """Fit on the whole train partition; time single-Reading scoring and measure the artifact."""
    x = train.select(schema.FEATURE_COLUMNS)
    started = time.perf_counter()
    pipeline = candidate.make_pipeline().fit(x, train[schema.TARGET].to_numpy())
    fit_seconds = time.perf_counter() - started
    latencies = []
    for i in range(n_calls):
        reading = x[i % x.height : i % x.height + 1]
        tick = time.perf_counter()
        score_readings(pipeline, reading)
        latencies.append((time.perf_counter() - tick) * 1000)
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    return OperationalCost(fit_seconds, float(median(latencies)), buffer.getbuffer().nbytes / 1e6)


def _learning_curve_seed(
    candidate: Candidate, train: pl.DataFrame, seed: int, validation_size: float, fraction: float
) -> tuple[int, float, float]:
    fit_part, validation = stratified_holdout(train, test_size=validation_size, seed=seed)
    reduced, _ = stratified_holdout(fit_part, test_size=1 - fraction, seed=seed)
    x_val = validation.select(schema.FEATURE_COLUMNS)
    y_val = validation[schema.TARGET].to_numpy()
    scores = []
    for part in (fit_part, reduced):
        pipeline = candidate.make_pipeline().fit(
            part.select(schema.FEATURE_COLUMNS), part[schema.TARGET].to_numpy()
        )
        scores.append(float(average_precision_score(y_val, score_readings(pipeline, x_val))))
    return seed, scores[0], scores[1]


def train_size_penalty(
    candidate: Candidate,
    train: pl.DataFrame,
    seeds: Sequence[int],
    *,
    validation_size: float = 0.3,
    fraction: float = LEARNING_CURVE_TRAIN_FRACTION,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """PR-AUC with all vs a ``fraction`` of the training rows, on throwaway validation splits.

    Works inside the train partition, so the holdout is never used. Columns: ``seed``,
    ``pr_auc_full``, ``pr_auc_reduced``, ``penalty`` (full - reduced).
    """
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_learning_curve_seed)(candidate, train, s, validation_size, fraction) for s in seeds
    )
    return pl.DataFrame(
        rows, schema=["seed", "pr_auc_full", "pr_auc_reduced"], orient="row"
    ).with_columns(penalty=pl.col("pr_auc_full") - pl.col("pr_auc_reduced"))
