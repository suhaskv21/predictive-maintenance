import numpy as np
import polars as pl
import pytest

from predictive_maintenance.data import schema
from predictive_maintenance.evaluation import modes as modes_module
from predictive_maintenance.evaluation.compare import cv_folds
from predictive_maintenance.evaluation.modes import (
    evaluate_modes,
    judge_mode,
    mode_repeat_scores,
)
from predictive_maintenance.evaluation.split import repeated_stratified_cv
from predictive_maintenance.training.candidates import build_candidates

N_SPLITS = 3


def readings(n: int = 800) -> pl.DataFrame:
    """HDF follows torque, RNF is random noise; the other modes are absent from the data."""
    rng = np.random.default_rng(0)
    torque = rng.normal(40, 10, n)
    hdf = np.zeros(n, dtype=int)
    hdf[np.argsort(-(torque + rng.normal(0, 1, n)))[:40]] = 1
    rnf = np.zeros(n, dtype=int)
    rnf[rng.choice(n, 12, replace=False)] = 1
    return pl.DataFrame(
        {
            "Type": rng.choice(list(schema.TYPE_LEVELS), n),
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 1, n),
            "Rotational speed [rpm]": rng.normal(1500, 100, n),
            "Torque [Nm]": torque,
            "Tool wear [min]": rng.uniform(0, 250, n),
            schema.TARGET: np.maximum(hdf, rnf),
            "TWF": np.zeros(n, dtype=int),
            "HDF": hdf,
            "PWF": np.zeros(n, dtype=int),
            "OSF": np.zeros(n, dtype=int),
            "RNF": rnf,
        }
    )


def folds_for(frame: pl.DataFrame, repeats: int = 2) -> list:
    cv = repeated_stratified_cv(n_splits=N_SPLITS, n_repeats=repeats, seed=1)
    return cv_folds(frame, cv)


def candidate():
    return build_candidates(0)["random_forest_unweighted"]


def judge(pr_auc: list[float], positives: int = 20, n: int = 1000, **margins: float):
    y = np.zeros(n, dtype=int)
    y[:positives] = 1
    settings = {"min_pr_auc": 0.30, "min_lift": 5.0, **margins}
    return judge_mode("HDF", y, np.array(pr_auc), n_splits=5, **settings)


def test_judge_mode_passes_a_mode_that_clears_all_three_checks() -> None:
    result = judge([0.60, 0.62, 0.58])
    assert result.predictable
    assert result.prevalence == 0.02
    assert result.lift == pytest.approx(30.0)
    assert result.positives == 20


def test_judge_mode_fails_below_the_pr_auc_floor() -> None:
    result = judge([0.20, 0.21, 0.19])
    assert result.meets_lift and result.beats_prevalence
    assert not result.meets_floor
    assert not result.predictable


def test_judge_mode_fails_below_the_lift_floor() -> None:
    result = judge([0.35, 0.36, 0.34], positives=100)  # prevalence 0.10, lift 3.5 < 5
    assert result.meets_floor and result.beats_prevalence
    assert not result.meets_lift
    assert not result.predictable


def test_judge_mode_fails_when_one_corrected_se_reaches_the_prevalence() -> None:
    # prevalence 0.2, mean 0.35, corrected SE about 0.21
    result = judge([0.05, 0.60, 0.40], positives=200, min_pr_auc=0.2, min_lift=1.5)
    assert result.meets_floor and result.meets_lift
    assert not result.beats_prevalence
    assert not result.predictable


def test_judge_mode_needs_a_positive_reading_and_two_repeats() -> None:
    with pytest.raises(ValueError, match="no positive"):
        judge_mode("RNF", np.zeros(50, dtype=int), np.array([0.1, 0.2]), n_splits=5, **_margins())
    with pytest.raises(ValueError, match="two repeats"):
        judge([0.5])


def _margins() -> dict[str, float]:
    return {"min_pr_auc": 0.30, "min_lift": 5.0}


def test_mode_repeat_scores_scores_every_reading_once_per_repeat() -> None:
    frame = readings()
    scores = mode_repeat_scores(
        candidate(), frame, folds_for(frame), "HDF", n_splits=N_SPLITS, n_jobs=1
    )
    assert scores.shape == (2, frame.height)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_mode_repeat_scores_gives_a_model_only_the_input_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = readings()
    seen: list[list[str]] = []

    def spy(_candidate, x, y, a, b):  # type: ignore[no-untyped-def]
        seen.append(x.columns)
        return np.zeros(len(b))

    monkeypatch.setattr(modes_module, "_fit_score", spy)
    mode_repeat_scores(candidate(), frame, folds_for(frame), "HDF", n_splits=N_SPLITS, n_jobs=1)
    assert seen
    assert all(columns == list(schema.FEATURE_COLUMNS) for columns in seen)


def test_mode_repeat_scores_rejects_a_column_that_is_not_a_failure_mode() -> None:
    frame = readings()
    with pytest.raises(ValueError, match="not a Failure mode"):
        mode_repeat_scores(
            candidate(), frame, folds_for(frame), schema.TARGET, n_splits=N_SPLITS, n_jobs=1
        )


def test_mode_repeat_scores_rejects_a_mode_that_never_occurs() -> None:
    frame = readings()
    with pytest.raises(ValueError, match="positive and negative"):
        mode_repeat_scores(candidate(), frame, folds_for(frame), "TWF", n_splits=N_SPLITS, n_jobs=1)


def test_mode_repeat_scores_rejects_folds_that_are_not_whole_repeats() -> None:
    frame = readings()
    with pytest.raises(ValueError, match="whole number of repeats"):
        mode_repeat_scores(
            candidate(), frame, folds_for(frame)[:-1], "HDF", n_splits=N_SPLITS, n_jobs=1
        )


def test_evaluate_modes_separates_a_mode_with_signal_from_noise() -> None:
    frame = readings()
    results = {
        r.mode: r
        for r in evaluate_modes(
            candidate(),
            frame,
            folds_for(frame),
            n_splits=N_SPLITS,
            min_pr_auc=0.30,
            min_lift=5.0,
            modes=("HDF", "RNF"),
            n_jobs=1,
        )
    }
    assert results["HDF"].predictable
    assert results["HDF"].pr_auc > 0.6
    assert not results["RNF"].predictable
    assert results["RNF"].positives == 12
