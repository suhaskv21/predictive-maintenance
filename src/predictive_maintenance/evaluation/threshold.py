"""Operating point analysis (ADR 0007): cost, threshold, risk bands, risk floor, intervals.

Everything here works on labels and scores. Threshold selection takes the out-of-fold scores of the
train partition's repeated CV (``repeat_scores``: one row of pooled out-of-fold scores per CV
repeat); the holdout is only ever passed to ``bootstrap_metric_cis``, once, for the final model.

Vocabulary follows ``GLOSSARY.md``: a *missed failure* is a Machine failure that was not flagged, a
*false alarm* a flag on a Reading that did not fail, and the *cost ratio* says how many false alarms
one missed failure is worth. A Reading is flagged when ``score >= threshold``.

The margins below are judgement calls fixed in ADR 0007 before the numbers they judge.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.compare import Fold, corrected_se, score_readings
from predictive_maintenance.training.candidates import Candidate

MIN_COST_SAVING = 0.25  # a model must beat "flag nothing" by this relative saving at every ratio
MIN_BAND_FAILURES = 10  # failures each flagged band needs (mean per repeat) to estimate its rate
COST_DIFFERENCE_MARGIN = 0.10  # relative expected-cost difference that makes one model "cheaper"
FLAT_OPTIMUM_TOLERANCE = 0.10  # relative cost excess at the threshold interval ends still "flat"
FLOOR_MINIMUM = 0.01
FLOOR_LOW_FRACTION = 0.5  # the lowest-scoring share of Readings that sets the risk floor
CI_LEVEL = 0.95
THRESHOLD_INTERVAL_LEVEL = 0.90
THRESHOLD_STREAM = 10  # seed streams (see ``compare.derived_seeds``), one per purpose
BOOTSTRAP_STREAM = 11
WILSON_Z = 1.959964  # 95 %
BAND_NAMES = ("Low", "Elevated", "High")

FloatArray = NDArray[np.float64]


def _binary(y: Sequence[int] | NDArray[Any]) -> NDArray[np.bool_]:
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.size == 0 or not np.isin(labels, (0, 1)).all():
        raise ValueError("y must be a non-empty 1-D array of 0/1 labels")
    return labels.astype(bool)


def _scores(y: NDArray[np.bool_], scores: Sequence[float] | NDArray[Any]) -> FloatArray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != y.shape:
        raise ValueError(f"scores shape {values.shape} does not match y shape {y.shape}")
    return values


def _score_fold(
    candidate: Candidate,
    x: pl.DataFrame,
    y: NDArray[np.int_],
    a: NDArray[np.intp],
    b: NDArray[np.intp],
) -> FloatArray:
    pipeline: Pipeline = candidate.make_pipeline().fit(x[a], y[a])
    return score_readings(pipeline, x[b])


def repeat_scores(
    candidate: Candidate,
    train: pl.DataFrame,
    folds: Sequence[Fold],
    *,
    n_splits: int,
    n_jobs: int = -1,
) -> FloatArray:
    """Out-of-fold failure probability per Reading for each CV repeat: shape ``(repeats, rows)``.

    Each row pools the ``n_splits`` test folds of one repeat, so it is what one model would score
    on Readings it was not trained on. Unlike ``compare.out_of_fold_scores`` the repeats are not
    averaged, so the threshold can be chosen per repeat and its spread reported. ``folds`` must be
    the repeat-major folds of ``repeated_stratified_cv`` on the train partition.

    Raises:
        ValueError: if ``folds`` is not a whole number of repeats or a repeat does not score every
            Reading exactly once.
    """
    if not folds or len(folds) % n_splits:
        raise ValueError(f"{len(folds)} folds are not a whole number of repeats of {n_splits}")
    x = train.select(schema.FEATURE_COLUMNS)
    y = train[schema.TARGET].to_numpy()
    results = Parallel(n_jobs=n_jobs)(delayed(_score_fold)(candidate, x, y, a, b) for a, b in folds)
    scores = np.full((len(folds) // n_splits, train.height), np.nan)
    seen = np.zeros(scores.shape, dtype=int)
    for i, ((_, b), fold_scores) in enumerate(zip(folds, results, strict=True)):
        scores[i // n_splits, b] = fold_scores
        seen[i // n_splits, b] += 1
    if not (seen == 1).all():
        raise ValueError("every repeat's test folds must cover each Reading exactly once")
    return scores


def expected_cost(
    y: Sequence[int] | NDArray[Any], flagged: NDArray[np.bool_], cost_ratio: float
) -> float:
    """Expected cost per Reading in false-alarm units: ``(ratio * missed + false alarms) / n``."""
    labels = _binary(y)
    missed = np.sum(labels & ~flagged)
    false_alarms = np.sum(~labels & flagged)
    return float((cost_ratio * missed + false_alarms) / labels.size)


def flag_nothing_cost(y: Sequence[int] | NDArray[Any], cost_ratio: float) -> float:
    """Expected cost per Reading of never flagging: every failure is missed."""
    labels = _binary(y)
    return float(cost_ratio * labels.sum() / labels.size)


def flag_everything_cost(y: Sequence[int] | NDArray[Any]) -> float:
    """Expected cost per Reading of flagging every Reading: every non-failure is a false alarm."""
    labels = _binary(y)
    return float((labels.size - labels.sum()) / labels.size)


def _threshold_costs(
    labels: NDArray[np.bool_], scores: FloatArray, cost_ratio: float
) -> tuple[FloatArray, FloatArray]:
    """Distinct observed scores (ascending) and the expected cost of flagging ``score >= value``."""
    values, inverse = np.unique(scores, return_inverse=True)
    positives = np.bincount(inverse, weights=labels, minlength=len(values))
    negatives = np.bincount(inverse, minlength=len(values)) - positives
    caught = np.cumsum(positives[::-1])[::-1]
    false_alarms = np.cumsum(negatives[::-1])[::-1]
    costs = (cost_ratio * (positives.sum() - caught) + false_alarms) / labels.size
    return values, costs


def pick_threshold(
    y: Sequence[int] | NDArray[Any], scores: Sequence[float] | NDArray[Any], cost_ratio: float
) -> float:
    """The threshold that minimises expected cost at ``cost_ratio``.

    Tied scores are never split: candidates are the distinct observed scores. The thresholds that
    give the exact minimum cost form intervals (a threshold anywhere in ``(previous score, score]``
    flags the same Readings). The widest such interval wins (the lowest if equally wide) and its
    midpoint is returned, so the threshold does not sit on a single observed score.
    """
    labels = _binary(y)
    values_in = _scores(labels, scores)
    values, costs = _threshold_costs(labels, values_in, cost_ratio)
    lower = np.concatenate(([min(0.0, float(values[0]))], values[:-1]))
    best = np.flatnonzero(costs <= costs.min() + 1e-12)
    runs = np.split(best, np.flatnonzero(np.diff(best) > 1) + 1)
    low, high = max(
        ((float(lower[run[0]]), float(values[run[-1]])) for run in runs),
        key=lambda interval: (interval[1] - interval[0], -interval[0]),
    )
    return (low + high) / 2


def per_repeat_thresholds(
    y: Sequence[int] | NDArray[Any], repeat_scores: FloatArray, cost_ratio: float
) -> tuple[float, ...]:
    """The cost-minimising threshold at ``cost_ratio`` picked on each CV repeat's scores."""
    labels = _binary(y)
    return tuple(pick_threshold(labels, row, cost_ratio) for row in repeat_scores)


def median_threshold(
    y: Sequence[int] | NDArray[Any], repeat_scores: FloatArray, cost_ratio: float
) -> float:
    """The Operating threshold at ``cost_ratio``: the median of the per-repeat picks (ADR 0007)."""
    return float(np.median(per_repeat_thresholds(y, repeat_scores, cost_ratio)))


def mean_cost(
    y: Sequence[int] | NDArray[Any], repeat_scores: FloatArray, threshold: float, cost_ratio: float
) -> float:
    """Expected cost per Reading at ``threshold``, averaged over the CV repeats."""
    return float(np.mean([expected_cost(y, row >= threshold, cost_ratio) for row in repeat_scores]))


@dataclass(frozen=True)
class ThresholdEstimate:
    """A cost-minimising threshold at one ratio, with its two sources of uncertainty.

    ``per_repeat`` has one threshold per CV repeat; ``median`` is the one to ship and
    ``repeat_range`` its spread (procedure noise). ``interval`` is a bootstrap interval over
    Readings (sampling noise from the limited number of failures).
    """

    cost_ratio: float
    per_repeat: tuple[float, ...]
    median: float
    repeat_range: tuple[float, float]
    interval: tuple[float, float]


def estimate_threshold(
    y: Sequence[int] | NDArray[Any],
    repeat_scores: FloatArray,
    cost_ratio: float,
    *,
    n_resamples: int,
    seed: int,
) -> ThresholdEstimate:
    """Pick the threshold per CV repeat, take the median, and bootstrap its interval.

    Each bootstrap resample redraws failures and non-failures separately (so the failure count is
    kept) and picks the threshold on one repeat's scores, cycling through the repeats.
    """
    labels = _binary(y)
    per_repeat = per_repeat_thresholds(labels, repeat_scores, cost_ratio)
    rng = np.random.default_rng(seed)
    positives, negatives = np.flatnonzero(labels), np.flatnonzero(~labels)
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = np.concatenate(
            (
                rng.choice(positives, positives.size, replace=True),
                rng.choice(negatives, negatives.size, replace=True),
            )
        )
        boot[i] = pick_threshold(
            labels[idx], repeat_scores[i % len(repeat_scores)][idx], cost_ratio
        )
    tail = (1 - THRESHOLD_INTERVAL_LEVEL) / 2 * 100
    low, high = np.percentile(boot, [tail, 100 - tail])
    return ThresholdEstimate(
        cost_ratio=cost_ratio,
        per_repeat=per_repeat,
        median=float(np.median(per_repeat)),
        repeat_range=(min(per_repeat), max(per_repeat)),
        interval=(float(low), float(high)),
    )


def interval_cost_excess(
    y: Sequence[int] | NDArray[Any], repeat_scores: FloatArray, estimate: ThresholdEstimate
) -> tuple[float, float]:
    """Relative extra expected cost at the two ends of the threshold interval vs at the median."""
    base = mean_cost(y, repeat_scores, estimate.median, estimate.cost_ratio)
    return tuple(  # type: ignore[return-value]
        mean_cost(y, repeat_scores, end, estimate.cost_ratio) / base - 1
        for end in estimate.interval
    )


def is_flat_optimum(excess: tuple[float, float]) -> bool:
    """Whether the cost at both interval ends stays within ``FLAT_OPTIMUM_TOLERANCE``."""
    return max(excess) <= FLAT_OPTIMUM_TOLERANCE


@dataclass(frozen=True)
class OperatingMetrics:
    """What an operating threshold means in operational terms, at one cost ratio.

    Costs are per 1,000 Readings in false-alarm units. ``precision`` and ``false_alarms_per_caught``
    are NaN when nothing is flagged or no failure is caught.
    """

    threshold: float
    cost_ratio: float
    recall: float
    precision: float
    alerts_per_1000: float
    false_alarms_per_caught: float
    cost_per_1000: float
    flag_nothing_per_1000: float
    flag_everything_per_1000: float
    saving_vs_flag_nothing: float


def operating_metrics(
    y: Sequence[int] | NDArray[Any],
    scores: Sequence[float] | NDArray[Any],
    threshold: float,
    cost_ratio: float,
) -> OperatingMetrics:
    """Recall, precision, alert rate and expected cost of flagging ``score >= threshold``."""
    labels = _binary(y)
    flagged = _scores(labels, scores) >= threshold
    caught = int(np.sum(labels & flagged))
    false_alarms = int(np.sum(~labels & flagged))
    cost = expected_cost(labels, flagged, cost_ratio)
    nothing = flag_nothing_cost(labels, cost_ratio)
    nan = math.nan
    return OperatingMetrics(
        threshold=threshold,
        cost_ratio=cost_ratio,
        recall=caught / labels.sum() if labels.any() else nan,
        precision=caught / (caught + false_alarms) if caught + false_alarms else nan,
        alerts_per_1000=1000 * flagged.mean(),
        false_alarms_per_caught=false_alarms / caught if caught else nan,
        cost_per_1000=1000 * cost,
        flag_nothing_per_1000=1000 * nothing,
        flag_everything_per_1000=1000 * flag_everything_cost(labels),
        saving_vs_flag_nothing=1 - cost / nothing if nothing else nan,
    )


def cv_operating_metrics(
    y: Sequence[int] | NDArray[Any], repeat_scores: FloatArray, threshold: float, cost_ratio: float
) -> OperatingMetrics:
    """``operating_metrics`` averaged over the CV repeats (NaN-aware)."""
    per_repeat = [operating_metrics(y, row, threshold, cost_ratio) for row in repeat_scores]
    averaged = {
        f.name: float(np.nanmean([getattr(m, f.name) for m in per_repeat]))
        for f in fields(per_repeat[0])
    }
    return OperatingMetrics(**averaged)


def is_viable(metrics_by_ratio: Sequence[OperatingMetrics]) -> bool:
    """Whether a model beats flagging nothing by ``MIN_COST_SAVING`` at every cost ratio."""
    return all(m.saving_vs_flag_nothing >= MIN_COST_SAVING for m in metrics_by_ratio)


def ranking_metrics(
    y: Sequence[int] | NDArray[Any], scores: Sequence[float] | NDArray[Any]
) -> dict[str, float]:
    """PR-AUC (ranking), ROC-AUC (optimistic at low prevalence) and Brier (probability quality)."""
    labels = _binary(y)
    values = _scores(labels, scores)
    return {
        "pr_auc": float(average_precision_score(labels, values)),
        "roc_auc": float(roc_auc_score(labels, values)),
        "brier": float(brier_score_loss(labels, values)),
    }


def wilson_interval(failures: float, n: float, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for an observed failure rate (``(0, 1)`` if ``n`` is 0)."""
    if n <= 0:
        return (0.0, 1.0)
    p = failures / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Bands:
    """Edges of the three Risk bands: Low below ``elevated``, High from ``high`` upward.

    ``elevated`` is the operating threshold at the default cost ratio (so Elevated and High are
    flagged) and ``high`` the one at the smallest ratio considered (flagged under every ratio).
    """

    elevated: float
    high: float

    def __post_init__(self) -> None:
        """Reject edges that do not leave room for three bands."""
        if not self.elevated < self.high:
            raise ValueError(
                f"band edges must satisfy elevated < high, got {self.elevated} and {self.high}"
            )

    def assign(self, scores: Sequence[float] | NDArray[Any]) -> NDArray[np.str_]:
        """The band name of each score."""
        values = np.asarray(scores, dtype=np.float64)
        return np.where(
            values >= self.high,
            BAND_NAMES[2],
            np.where(values >= self.elevated, BAND_NAMES[1], BAND_NAMES[0]),
        ).astype(np.str_)


def band_table(
    y: Sequence[int] | NDArray[Any], repeat_scores: FloatArray, bands: Bands
) -> pl.DataFrame:
    """Observed failure rate per band, with a Wilson interval.

    ``readings`` and ``failures`` are means over the CV repeats (the repeats rescore the same
    Readings, so pooling them would overstate the evidence).
    """
    labels = _binary(y)
    assigned = [bands.assign(row) for row in repeat_scores]
    rows = []
    for name in BAND_NAMES:
        readings = float(np.mean([np.sum(a == name) for a in assigned]))
        failures = float(np.mean([np.sum((a == name) & labels) for a in assigned]))
        low, high = wilson_interval(failures, readings)
        rows.append(
            (name, readings, failures, failures / readings if readings else math.nan, low, high)
        )
    return pl.DataFrame(
        rows,
        schema=["band", "readings", "failures", "observed_rate", "wilson_low", "wilson_high"],
        orient="row",
    )


@dataclass(frozen=True)
class BandCheck:
    """Whether the three bands may ship, and why not if they may not."""

    ships: bool
    reason: str


def check_bands(table: pl.DataFrame) -> BandCheck:
    """Bands ship only if Elevated and High each hold ``MIN_BAND_FAILURES`` failures and the
    observed rate strictly increases from Low to Elevated to High."""
    failures = dict(zip(table["band"], table["failures"], strict=True))
    rates = [float(r) for r in table["observed_rate"]]
    thin = [b for b in BAND_NAMES[1:] if failures[b] < MIN_BAND_FAILURES]
    if thin:
        return BandCheck(False, f"fewer than {MIN_BAND_FAILURES} failures in {', '.join(thin)}")
    if any(math.isnan(r) for r in rates) or not rates[0] < rates[1] < rates[2]:
        return BandCheck(False, "observed failure rate does not strictly increase across the bands")
    return BandCheck(True, "")


def risk_floor(y: Sequence[int] | NDArray[Any], repeat_scores: FloatArray) -> float:
    """Lowest Failure risk the app may show as a number, for one model.

    The 95 % Wilson upper bound on the observed failure rate of the lowest-scoring
    ``FLOOR_LOW_FRACTION`` of Readings, rounded up to a whole percent and at least
    ``FLOOR_MINIMUM``. Below it the app says "under F %", because the model cannot resolve smaller
    risks. Use the maximum over the offered models so the app has one wording.
    """
    labels = _binary(y)
    counts, hits = [], []
    for row in repeat_scores:
        low = _scores(labels, row) <= np.quantile(row, FLOOR_LOW_FRACTION)
        counts.append(low.sum())
        hits.append((low & labels).sum())
    upper = wilson_interval(float(np.mean(hits)), float(np.mean(counts)))[1]
    return max(FLOOR_MINIMUM, math.ceil(round(upper * 100, 9)) / 100)


@dataclass(frozen=True)
class OperatingPoint:
    """Everything the app needs next to one model's Pipeline artifact (ADR 0007).

    ``thresholds`` maps each cost ratio to its operating threshold and ``default_cost_ratio`` picks
    the one that flags. ``bands`` is None when the three bands do not ship for this model.
    """

    model: str
    default_cost_ratio: float
    thresholds: Mapping[float, float]
    bands: Bands | None
    risk_floor: float

    def __post_init__(self) -> None:
        """Require a threshold at the default cost ratio."""
        if self.default_cost_ratio not in self.thresholds:
            raise ValueError("thresholds must include the default cost ratio")

    @property
    def threshold(self) -> float:
        """The operating threshold at the default cost ratio."""
        return self.thresholds[self.default_cost_ratio]

    def to_json(self) -> str:
        """Serialise to the sidecar JSON stored next to the model's Pipeline artifact."""
        return json.dumps(
            {
                "model": self.model,
                "default_cost_ratio": self.default_cost_ratio,
                "thresholds": {repr(r): t for r, t in sorted(self.thresholds.items())},
                "bands": None
                if self.bands is None
                else {"elevated": self.bands.elevated, "high": self.bands.high},
                "risk_floor": self.risk_floor,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "OperatingPoint":
        """Inverse of ``to_json``."""
        raw = json.loads(text)
        bands = raw["bands"]
        return cls(
            model=raw["model"],
            default_cost_ratio=float(raw["default_cost_ratio"]),
            thresholds={float(r): float(t) for r, t in raw["thresholds"].items()},
            bands=None if bands is None else Bands(bands["elevated"], bands["high"]),
            risk_floor=float(raw["risk_floor"]),
        )


def build_operating_point(
    model: str,
    y: Sequence[int] | NDArray[Any],
    repeat_scores: FloatArray,
    *,
    thresholds: Mapping[float, float],
    default_cost_ratio: float,
    floor: float,
) -> tuple[OperatingPoint, pl.DataFrame, BandCheck]:
    """Assemble a model's operating point, its band table and whether the bands ship.

    Band edges are the threshold at the default ratio and at the smallest ratio.

    Raises:
        ValueError: if those two thresholds are not strictly increasing, or the default ratio is
            not among ``thresholds``.
    """
    bands = Bands(elevated=thresholds[default_cost_ratio], high=thresholds[min(thresholds)])
    table = band_table(y, repeat_scores, bands)
    check = check_bands(table)
    point = OperatingPoint(
        model=model,
        default_cost_ratio=default_cost_ratio,
        thresholds=dict(thresholds),
        bands=bands if check.ships else None,
        risk_floor=floor,
    )
    return point, table, check


def fold_expected_costs(
    y: Sequence[int] | NDArray[Any],
    repeat_scores: FloatArray,
    folds: Sequence[Fold],
    *,
    n_splits: int,
    threshold: float,
    cost_ratio: float,
) -> FloatArray:
    """Expected cost per Reading on each CV fold's test rows at ``threshold``."""
    labels = _binary(y)
    return np.array(
        [
            expected_cost(labels[b], repeat_scores[i // n_splits][b] >= threshold, cost_ratio)
            for i, (_, b) in enumerate(folds)
        ]
    )


@dataclass(frozen=True)
class CostComparison:
    """Paired expected-cost comparison of a challenger against an incumbent on identical folds."""

    mean_saving: float
    relative_saving: float
    corrected_se: float
    cheaper: bool


def compare_costs(
    challenger: FloatArray, incumbent: FloatArray, *, n_cv_folds: int
) -> CostComparison:
    """Paired comparison of two models' per-fold costs (a saving is incumbent minus challenger).

    The challenger is *cheaper* if its mean saving beats the larger of ``COST_DIFFERENCE_MARGIN``
    times the incumbent's cost and the Nadeau-Bengio corrected SE. Informational: it never changes
    the default model.
    """
    saving = incumbent - challenger
    se = corrected_se(saving, n_cv_folds=n_cv_folds)
    mean_saving = float(saving.mean())
    base = float(incumbent.mean())
    return CostComparison(
        mean_saving=mean_saving,
        relative_saving=mean_saving / base if base else math.nan,
        corrected_se=se,
        cheaper=mean_saving > max(COST_DIFFERENCE_MARGIN * base, se),
    )


def bootstrap_metric_cis(
    y: Sequence[int] | NDArray[Any],
    scores: Sequence[float] | NDArray[Any],
    *,
    threshold: float,
    cost_ratio: float,
    n_resamples: int,
    seed: int,
    level: float = CI_LEVEL,
) -> pl.DataFrame:
    """Stratified percentile bootstrap intervals for the final metrics at a *frozen* threshold.

    Failures and non-failures are resampled separately so every resample keeps the failure count.
    The threshold is never re-chosen inside a resample. Columns: ``metric``, ``estimate``, ``low``,
    ``high``. Call it for the holdout once, for the single chosen model.
    """
    labels = _binary(y)
    values = _scores(labels, scores)
    positives, negatives = np.flatnonzero(labels), np.flatnonzero(~labels)

    def measure(idx: NDArray[np.intp]) -> dict[str, float]:
        sub_y, sub_s = labels[idx], values[idx]
        flagged = sub_s >= threshold
        caught = int(np.sum(sub_y & flagged))
        return {
            **ranking_metrics(sub_y, sub_s),
            "recall": caught / sub_y.sum(),
            "precision": caught / flagged.sum() if flagged.any() else math.nan,
            "expected_cost_per_1000": 1000 * expected_cost(sub_y, flagged, cost_ratio),
        }

    estimate = measure(np.arange(labels.size))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        idx = np.concatenate(
            (
                rng.choice(positives, positives.size, replace=True),
                rng.choice(negatives, negatives.size, replace=True),
            )
        )
        draws.append(measure(idx))
    tail = (1 - level) / 2 * 100
    rows = []
    for name, point in estimate.items():
        low, high = np.nanpercentile([d[name] for d in draws], [tail, 100 - tail])
        rows.append((name, point, float(low), float(high)))
    return pl.DataFrame(rows, schema=["metric", "estimate", "low", "high"], orient="row")
