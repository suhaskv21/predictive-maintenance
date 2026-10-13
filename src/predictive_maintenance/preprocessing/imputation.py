"""Strategies for an Incomplete reading (ADR 0008): fill absent inputs before features are derived.

``InputImputer`` is the first step of the serialized ``Pipeline``. It works on the six raw inputs
(``Type`` and the five sensors), so engineered features are built from filled raw values and stay
consistent (power = torque x speed). Everything is learned from the training fold only, and a
complete Reading passes through unchanged, so scores of complete Readings do not move.

Strategies (``reject`` has no transformer: the inference layer refuses before scoring):
- ``median``: sensors by their training median, ``Type`` by its training mode.
- ``regression``: absent sensors predicted by a ridge regression on the sensors that are present
  (one model per set of absent sensors, fitted on the complete training rows); ``Type`` by mode.
- ``native``: leave absent values to the model (the boosters and the forest accept NaN); only
  ``Type`` needs the tolerant encoder in ``build_feature_transformer(tolerate_unknown_type=True)``.
  Trained on complete data, a model has never seen NaN, so where it sends it is not learned.
"""

from collections.abc import Iterator
from itertools import combinations
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge

from predictive_maintenance.data import schema

Strategy = Literal["reject", "median", "regression", "native"]
# Simplest first: the order of the selection walk (ADR 0008).
STRATEGIES: tuple[Strategy, ...] = ("reject", "median", "regression", "native")
IMPUTING_STRATEGIES = ("median", "regression", "native")
RIDGE_ALPHA = 1e-3  # near-ordinary least squares; only guards against collinear sensors

_SENSORS = list(schema.NUMERIC_FEATURES)


def _absent_sets() -> Iterator[tuple[str, ...]]:
    for size in range(1, len(_SENSORS) + 1):
        yield from combinations(_SENSORS, size)


class InputImputer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Fill absent raw inputs (nulls or NaN) from statistics of the training fold."""

    def __init__(self, strategy: str = "median") -> None:
        """Choose one of ``IMPUTING_STRATEGIES``."""
        self.strategy = strategy

    def fit(self, X: pl.DataFrame, y: object = None) -> "InputImputer":  # noqa: N803
        """Learn medians, the ``Type`` mode and (for ``regression``) the ridge models.

        Raises:
            ValueError: if the strategy is unknown or the training rows have absent inputs.
        """
        if self.strategy not in IMPUTING_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {IMPUTING_STRATEGIES}, got {self.strategy!r}"
            )
        frame = self._normalise(X)
        if frame.null_count().sum_horizontal().item() > 0:
            raise ValueError("imputers are fitted on complete training Readings only")
        self.medians_ = {c: float(frame[c].median()) for c in _SENSORS}  # type: ignore[arg-type]
        self.type_mode_ = str(frame[schema.CATEGORICAL_FEATURE].mode().sort()[0])
        self.models_: dict[tuple[str, ...], tuple[list[str], Ridge]] = {}
        if self.strategy == "regression":
            values = frame.select(_SENSORS).to_numpy()
            for absent in _absent_sets():
                present = [c for c in _SENSORS if c not in absent]
                if not present:
                    continue
                model = Ridge(alpha=RIDGE_ALPHA).fit(
                    values[:, [_SENSORS.index(c) for c in present]],
                    values[:, [_SENSORS.index(c) for c in absent]],
                )
                self.models_[absent] = (present, model)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:  # noqa: N803
        """Return the six inputs with absent values filled (``native`` leaves sensors absent)."""
        frame = self._normalise(X)
        if self.strategy == "native":
            return frame
        if self.strategy == "regression":
            frame = self._regress(frame)
        return frame.with_columns(
            pl.col(c).fill_null(self.medians_[c]) for c in _SENSORS
        ).with_columns(pl.col(schema.CATEGORICAL_FEATURE).fill_null(self.type_mode_))

    @staticmethod
    def _normalise(frame: pl.DataFrame) -> pl.DataFrame:
        """The six inputs in canonical order, sensors as Float64 with NaN treated as absent."""
        return frame.select(schema.FEATURE_COLUMNS).with_columns(
            pl.col(c).cast(pl.Float64).fill_nan(None) for c in _SENSORS
        )

    def _regress(self, frame: pl.DataFrame) -> pl.DataFrame:
        values = frame.select(_SENSORS).to_numpy().copy()
        absent_mask: NDArray[np.bool_] = np.isnan(values)
        keys = [
            tuple(c for c, gone in zip(_SENSORS, row, strict=True) if gone) for row in absent_mask
        ]
        for absent in set(keys):
            if absent not in self.models_:  # nothing absent, or every sensor absent
                continue
            rows = np.array([k == absent for k in keys])
            present, model = self.models_[absent]
            predicted = model.predict(values[np.ix_(rows, [_SENSORS.index(c) for c in present])])
            values[np.ix_(rows, [_SENSORS.index(c) for c in absent])] = predicted.reshape(
                int(rows.sum()), len(absent)
            )
        filled = pl.DataFrame(values, schema=_SENSORS, orient="row").with_columns(
            pl.col(c).fill_nan(None) for c in _SENSORS
        )
        return frame.with_columns(filled[c].alias(c) for c in _SENSORS)

    def get_feature_names_out(self, input_features: object = None) -> list[str]:
        """The six input names, in order."""
        return list(schema.FEATURE_COLUMNS)
