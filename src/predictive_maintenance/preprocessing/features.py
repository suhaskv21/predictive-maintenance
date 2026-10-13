"""Candidate feature transformer: raw features, optionally with engineered physics features.

The transformer is unfitted on creation; fit it on training readings only (inside a scikit-learn
``Pipeline`` so cross-validation refits it per fold). Columns other than the candidate
features (target, failure modes, identifiers) are dropped by construction.
"""

import math
from collections.abc import Collection, Sequence

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictive_maintenance.data import schema

POWER = "Power [W]"
TEMPERATURE_DELTA = "Temperature delta [K]"
WEAR_TORQUE = "Wear x torque [min Nm]"
ENGINEERED_FEATURES = (POWER, TEMPERATURE_DELTA, WEAR_TORQUE)

_RPM_TO_RAD_PER_S = 2 * math.pi / 60


def add_engineered_features(readings: pl.DataFrame) -> pl.DataFrame:
    """Append power, temperature delta and wear x torque to ``readings``."""
    torque = pl.col("Torque [Nm]")
    return readings.with_columns(
        (torque * pl.col("Rotational speed [rpm]") * _RPM_TO_RAD_PER_S).alias(POWER),
        (pl.col("Process temperature [K]") - pl.col("Air temperature [K]")).alias(
            TEMPERATURE_DELTA
        ),
        (pl.col("Tool wear [min]") * torque).alias(WEAR_TORQUE),
    )


class EngineeredFeatures(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Derive the requested engineered features from the raw sensor columns (stateless)."""

    def __init__(self, features: Sequence[str] = ENGINEERED_FEATURES) -> None:
        """Select which of ``ENGINEERED_FEATURES`` to derive."""
        self.features = features

    def fit(self, X: pl.DataFrame, y: object = None) -> "EngineeredFeatures":  # noqa: N803
        """Do nothing; the derivation has no learned state."""
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:  # noqa: N803
        """Return only the requested engineered features."""
        return add_engineered_features(X).select(self.features)

    def get_feature_names_out(self, input_features: object = None) -> list[str]:
        """Names of the derived features."""
        return list(self.features)


def build_feature_transformer(
    engineered: Collection[str] = (), *, tolerate_unknown_type: bool = False
) -> ColumnTransformer:
    """Build the unfitted feature transformer for one cell of the raw/engineered comparison.

    Output (polars): one-hot ``Type`` (L, M, H), the five standardized sensors and the
    standardized ``engineered`` features, in the canonical ``ENGINEERED_FEATURES`` order. With
    ``tolerate_unknown_type`` an absent or unknown ``Type`` encodes as all zeros instead of raising
    (the ``native`` missing-data strategy, ADR 0008).

    Raises:
        ValueError: if ``engineered`` names a feature outside ``ENGINEERED_FEATURES``.
    """
    unknown = set(engineered) - set(ENGINEERED_FEATURES)
    if unknown:
        raise ValueError(
            f"unknown engineered features {sorted(unknown)}; known: {ENGINEERED_FEATURES}"
        )
    chosen = [name for name in ENGINEERED_FEATURES if name in engineered]
    branches: list[tuple[str, object, list[str]]] = [
        (
            "type",
            OneHotEncoder(
                categories=[list(schema.TYPE_LEVELS)],
                sparse_output=False,
                handle_unknown="ignore" if tolerate_unknown_type else "error",
            ),
            [schema.CATEGORICAL_FEATURE],
        ),
        ("sensors", StandardScaler(), list(schema.NUMERIC_FEATURES)),
    ]
    if chosen:
        derived = Pipeline([("derive", EngineeredFeatures(chosen)), ("scale", StandardScaler())])
        branches.append(("engineered", derived, list(schema.NUMERIC_FEATURES)))
    transformer = ColumnTransformer(branches, remainder="drop", verbose_feature_names_out=False)
    transformer.set_output(transform="polars")
    return transformer
