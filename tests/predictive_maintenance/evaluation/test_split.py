import numpy as np
import polars as pl
import pytest

from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.split import repeated_stratified_cv, stratified_holdout

N_READINGS = 1000
N_FAILURES = 50


def readings() -> pl.DataFrame:
    """1000 Readings with 50 failures, scattered so row order carries no signal."""
    target = np.zeros(N_READINGS, dtype=int)
    target[np.random.default_rng(0).choice(N_READINGS, N_FAILURES, replace=False)] = 1
    return pl.DataFrame({"UDI": np.arange(N_READINGS), schema.TARGET: target})


def test_holdout_partitions_every_reading_exactly_once() -> None:
    train, test = stratified_holdout(readings(), test_size=0.2, seed=1)
    assert sorted([*train["UDI"], *test["UDI"]]) == list(range(N_READINGS))


def test_holdout_sizes_follow_test_size() -> None:
    train, test = stratified_holdout(readings(), test_size=0.3, seed=1)
    assert (train.height, test.height) == (700, 300)


def test_holdout_keeps_all_columns_and_original_row_order() -> None:
    train, test = stratified_holdout(readings(), test_size=0.2, seed=1)
    assert train.columns == test.columns == readings().columns
    assert train["UDI"].is_sorted()
    assert test["UDI"].is_sorted()


@pytest.mark.parametrize("seed", range(20))
def test_holdout_stratifies_the_target(seed: int) -> None:
    train, test = stratified_holdout(readings(), test_size=0.2, seed=seed)
    assert test[schema.TARGET].sum() == 10
    assert train[schema.TARGET].sum() == 40


def test_holdout_is_reproducible_for_a_seed_and_varies_across_seeds() -> None:
    first = stratified_holdout(readings(), test_size=0.2, seed=7)
    again = stratified_holdout(readings(), test_size=0.2, seed=7)
    other = stratified_holdout(readings(), test_size=0.2, seed=8)
    assert first[1].equals(again[1])
    assert not first[1].equals(other[1])


def test_repeated_cv_yields_n_splits_times_n_repeats_folds() -> None:
    cv = repeated_stratified_cv(n_splits=5, n_repeats=3, seed=1)
    folds = list(cv.split(readings(), readings()[schema.TARGET]))
    assert len(folds) == 15
    assert cv.get_n_splits() == 15


def test_repeated_cv_test_folds_of_one_repeat_cover_every_reading_once() -> None:
    cv = repeated_stratified_cv(n_splits=5, n_repeats=2, seed=1)
    folds = list(cv.split(readings(), readings()[schema.TARGET]))
    for repeat in range(2):
        covered = np.concatenate([test for _, test in folds[repeat * 5 : (repeat + 1) * 5]])
        assert sorted(covered) == list(range(N_READINGS))


def test_repeated_cv_stratifies_every_fold() -> None:
    cv = repeated_stratified_cv(n_splits=5, n_repeats=2, seed=1)
    target = readings()[schema.TARGET].to_numpy()
    for train_idx, test_idx in cv.split(readings(), target):
        assert target[test_idx].sum() == 10
        assert target[train_idx].sum() == 40


def test_repeated_cv_is_reproducible_for_a_seed_and_varies_across_seeds() -> None:
    target = readings()[schema.TARGET]

    def first_test_fold(seed: int) -> list[int]:
        cv = repeated_stratified_cv(n_splits=5, n_repeats=1, seed=seed)
        return sorted(next(iter(cv.split(readings(), target)))[1])

    assert first_test_fold(3) == first_test_fold(3)
    assert first_test_fold(3) != first_test_fold(4)
