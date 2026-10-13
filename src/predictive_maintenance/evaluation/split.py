"""Seeded, stratified splitters: the train/test holdout and the repeated CV on the train partition.

Every function takes its seed from the caller (``configs/default.toml`` via ``load_config``) and the
ratios as required arguments, so the split strategy is decided in one place (ADR 0005), not here.
"""

import numpy as np
import polars as pl
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from predictive_maintenance.data import schema


def stratified_holdout(
    readings: pl.DataFrame, *, test_size: float, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split ``readings`` into train and test, stratified on ``Machine failure``.

    Both partitions keep the original row order and all columns. Split before fitting anything;
    fit preprocessing on the train partition only.
    """
    positions = np.arange(readings.height)
    train_idx, test_idx = train_test_split(
        positions,
        test_size=test_size,
        stratify=readings[schema.TARGET].to_numpy(),
        random_state=seed,
    )
    return readings[np.sort(train_idx)], readings[np.sort(test_idx)]


def repeated_stratified_cv(*, n_splits: int, n_repeats: int, seed: int) -> RepeatedStratifiedKFold:
    """Repeated stratified K-fold splitter (``n_splits * n_repeats`` folds) for the train data."""
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
