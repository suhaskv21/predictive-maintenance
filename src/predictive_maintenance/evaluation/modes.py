"""Per-mode predictability check (ADR 0009, closing the open item of ADR 0004).

Can the five sensors and ``Type`` predict *which* Failure mode occurs, well enough to justify a
secondary output? One binary model per mode, on the train partition's repeated CV only (the holdout
is never passed in). Failure modes are outcomes, never features: only ``schema.FEATURE_COLUMNS`` are
given to a model.

Each repeat pools its test folds into one out-of-fold score per Reading, because a rare mode
(RNF has 13 positives in the train partition) is absent from many single folds. PR-AUC is computed
per repeat; the corrected standard error is the Nadeau-Bengio form applied to those per-repeat
values. It captures how fold assignment moves the figure, not the sampling noise of a handful of
positives, so it understates the uncertainty for the rarest modes.

The margins come from ``configs/default.toml`` (``[failure_modes]``), fixed before any number.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score

from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.compare import Fold, corrected_se, score_readings
from predictive_maintenance.training.candidates import Candidate

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ModeResult:
    """Predictability of one Failure mode from the Reading's inputs (train-partition CV).

    ``prevalence`` is the mode's share of train Readings, the PR-AUC of a model with no skill.
    ``lift`` is ``pr_auc / prevalence``. A mode is ``predictable`` only if all three checks pass:
    the PR-AUC floor, the lift floor, and PR-AUC minus one corrected SE above the prevalence.
    """

    mode: str
    positives: int
    prevalence: float
    pr_auc: float
    corrected_se: float
    meets_floor: bool
    meets_lift: bool
    beats_prevalence: bool

    @property
    def lift(self) -> float:
        """PR-AUC as a multiple of the prevalence."""
        return self.pr_auc / self.prevalence

    @property
    def predictable(self) -> bool:
        """Whether the pre-declared rule judges the mode predictable enough for a follow-up ADR."""
        return self.meets_floor and self.meets_lift and self.beats_prevalence


def _fit_score(
    candidate: Candidate,
    x: pl.DataFrame,
    y: NDArray[np.int_],
    a: NDArray[np.intp],
    b: NDArray[np.intp],
) -> FloatArray:
    return score_readings(candidate.make_pipeline().fit(x[a], y[a]), x[b])


def mode_repeat_scores(
    candidate: Candidate,
    train: pl.DataFrame,
    folds: Sequence[Fold],
    mode: str,
    *,
    n_splits: int,
    n_jobs: int = -1,
) -> FloatArray:
    """Out-of-fold probability of ``mode`` per Reading and CV repeat: shape ``(repeats, rows)``.

    ``folds`` must be the repeat-major folds of ``repeated_stratified_cv`` on the train partition.

    Raises:
        ValueError: if ``mode`` is not a Failure mode, has no positive or no negative Reading in
            ``train``, or ``folds`` is not a whole number of repeats covering each Reading once.
    """
    if mode not in schema.FAILURE_MODES:
        raise ValueError(f"{mode!r} is not a Failure mode {schema.FAILURE_MODES}")
    y = train[mode].to_numpy()
    if y.min() == y.max():
        raise ValueError(f"{mode} needs both positive and negative Readings in the train partition")
    if not folds or len(folds) % n_splits:
        raise ValueError(f"{len(folds)} folds are not a whole number of repeats of {n_splits}")
    x = train.select(schema.FEATURE_COLUMNS)
    results = Parallel(n_jobs=n_jobs)(delayed(_fit_score)(candidate, x, y, a, b) for a, b in folds)
    scores = np.full((len(folds) // n_splits, train.height), np.nan)
    seen = np.zeros(scores.shape, dtype=int)
    for i, ((_, b), fold_scores) in enumerate(zip(folds, results, strict=True)):
        scores[i // n_splits, b] = fold_scores
        seen[i // n_splits, b] += 1
    if not (seen == 1).all():
        raise ValueError("every repeat's test folds must cover each Reading exactly once")
    return scores


def judge_mode(
    mode: str,
    y: NDArray[np.int_],
    repeat_pr_auc: FloatArray,
    *,
    n_splits: int,
    min_pr_auc: float,
    min_lift: float,
) -> ModeResult:
    """Apply the pre-declared rule to one mode's per-repeat PR-AUC values.

    Raises:
        ValueError: if ``y`` has no positive Reading or fewer than two repeat values are given.
    """
    if len(repeat_pr_auc) < 2:
        raise ValueError("at least two repeats are needed for a standard error")
    positives = int(y.sum())
    if positives == 0:
        raise ValueError(f"{mode} has no positive Reading")
    prevalence = positives / len(y)
    pr_auc = float(repeat_pr_auc.mean())
    se = corrected_se(repeat_pr_auc, n_cv_folds=n_splits)
    return ModeResult(
        mode=mode,
        positives=positives,
        prevalence=prevalence,
        pr_auc=pr_auc,
        corrected_se=se,
        meets_floor=pr_auc >= min_pr_auc,
        meets_lift=pr_auc >= min_lift * prevalence,
        beats_prevalence=pr_auc - se > prevalence,
    )


def evaluate_modes(
    candidate: Candidate,
    train: pl.DataFrame,
    folds: Sequence[Fold],
    *,
    n_splits: int,
    min_pr_auc: float,
    min_lift: float,
    modes: Sequence[str] = schema.FAILURE_MODES,
    n_jobs: int = -1,
) -> list[ModeResult]:
    """Judge every Failure mode in ``modes`` on the train partition's repeated CV.

    Raises:
        ValueError: as ``mode_repeat_scores`` and ``judge_mode``.
    """
    results = []
    for mode in modes:
        scores = mode_repeat_scores(candidate, train, folds, mode, n_splits=n_splits, n_jobs=n_jobs)
        y = train[mode].to_numpy()
        repeat_pr_auc = np.array([average_precision_score(y, row) for row in scores])
        results.append(
            judge_mode(
                mode,
                y,
                repeat_pr_auc,
                n_splits=n_splits,
                min_pr_auc=min_pr_auc,
                min_lift=min_lift,
            )
        )
    return results
