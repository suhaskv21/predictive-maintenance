"""Candidate model families for the ADR 0006 comparison, each as one preprocessing + model Pipeline.

Settings are fixed and pre-declared (no tuning in the family decision). Every ``random_state`` is
the seed passed in, which comes from ``configs/default.toml``. Every supervised family has an
unweighted and a class-weighted (``balanced``) variant, named ``<family>_unweighted`` and
``<family>_weighted``. XGBoost and LightGBM are imported lazily when their model is built.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import OneClassSVM
from sklearn.utils.class_weight import compute_sample_weight

from predictive_maintenance.preprocessing.features import (
    ENGINEERED_FEATURES,
    build_feature_transformer,
)
from predictive_maintenance.preprocessing.imputation import (
    IMPUTING_STRATEGIES,
    STRATEGIES,
    InputImputer,
)

Role = Literal["baseline", "selectable", "reference"]

CALIBRATION_METHODS = ("sigmoid", "isotonic")
CALIBRATION_FOLDS = 5
ONE_CLASS_SVM_NU = 0.034  # the Machine failure prevalence

FAMILIES = ("logistic", "random_forest", "hist_gradient_boosting", "lightgbm", "xgboost")
TREE_FAMILIES = FAMILIES[1:]
# Families offered for selection once ADR 0007 has set a threshold for each. HistGradientBoosting is
# still compared but not offered: it is redundant with LightGBM (ADR 0006, post-review change).
SWITCHABLE_FAMILIES = ("random_forest", "lightgbm", "xgboost")
UNWEIGHTED = "unweighted"
WEIGHTED = "weighted"


def variant_names(family: str) -> tuple[str, str]:
    """The unweighted then weighted candidate names of ``family``."""
    return f"{family}_{UNWEIGHTED}", f"{family}_{WEIGHTED}"


# Simplest to most complex (ADR 0006); within a family unweighted comes first. The selection rule
# walks this order top down.
SELECTION_ORDER = tuple(name for family in FAMILIES for name in variant_names(family))


class NoveltyScorer(BaseEstimator):  # type: ignore[misc]
    """Fit an anomaly detector on non-failure Readings only; score = how anomalous a Reading is."""

    def __init__(self, detector: Any) -> None:
        """Wrap an unfitted detector exposing ``score_samples`` (higher = more normal)."""
        self.detector = detector

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int_]) -> "NoveltyScorer":  # noqa: N803
        """Fit a clone of the detector on the rows where the target is 0."""
        self.detector_ = clone(self.detector).fit(X[np.asarray(y) == 0])
        return self

    def decision_function(self, X: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: N803
        """Anomaly score: larger means more unlike the non-failure Readings."""
        scores: NDArray[np.float64] = -self.detector_.score_samples(X)
        return scores


class BalancedSampleWeights(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Fit a classifier with ``balanced`` sample weights computed from the y it is fitted on.

    For estimators with no ``class_weight`` argument (XGBoost). The weights come from the training
    rows of each fit, so cross-validation computes them inside the fold.
    """

    def __init__(self, estimator: Any) -> None:
        """Wrap an unfitted classifier whose ``fit`` accepts ``sample_weight``."""
        self.estimator = estimator

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int_]) -> "BalancedSampleWeights":  # noqa: N803
        """Fit a clone of the estimator with balanced sample weights."""
        weights = compute_sample_weight("balanced", y)
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=weights)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: N803
        """Class probabilities from the fitted estimator."""
        probabilities: NDArray[np.float64] = self.estimator_.predict_proba(X)
        return probabilities

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int_]:  # noqa: N803
        """Predicted class labels from the fitted estimator."""
        labels: NDArray[np.int_] = self.estimator_.predict(X)
        return labels


def _frame_to_array(frame: pl.DataFrame) -> NDArray[np.float64]:
    """Convert the transformed polars frame at the scikit-learn boundary (bracketed names)."""
    return frame.to_numpy()


@dataclass(frozen=True)
class Candidate:
    """One candidate model family: a name, its role in selection, and how to build it."""

    name: str
    role: Role
    make_model: Callable[[], Any]
    seed: int
    engineered: tuple[str, ...] = ENGINEERED_FEATURES

    def make_pipeline(
        self, calibration: str | None = None, *, missing_strategy: str | None = None
    ) -> Pipeline:
        """Unfitted feature transformer + model, optionally wrapped in probability calibration.

        The transformer derives the ``engineered`` features (all three of ADR 0004 by default;
        ``dataclasses.replace(candidate, engineered=())`` gives the raw-feature variant).
        ``missing_strategy`` (ADR 0008) adds an ``InputImputer`` step before the features:
        ``median``, ``regression`` or ``native``. ``None`` and ``reject`` give the plain pipeline,
        which cannot score an Incomplete reading (the inference layer refuses it first).

        Raises:
            ValueError: if ``calibration`` is not one of ``CALIBRATION_METHODS`` or
                ``missing_strategy`` is not a known strategy.
        """
        if missing_strategy not in (None, *STRATEGIES):
            raise ValueError(
                f"missing_strategy must be one of {STRATEGIES}, got {missing_strategy!r}"
            )
        model = self.make_model()
        if calibration is not None:
            if calibration not in CALIBRATION_METHODS:
                raise ValueError(
                    f"calibration must be one of {CALIBRATION_METHODS}, got {calibration!r}"
                )
            model = CalibratedClassifierCV(
                model,
                method=calibration,
                cv=StratifiedKFold(CALIBRATION_FOLDS, shuffle=True, random_state=self.seed),
            )
        steps: list[tuple[str, Any]] = []
        if missing_strategy in IMPUTING_STRATEGIES:
            steps.append(("impute", InputImputer(missing_strategy)))
        steps += [
            (
                "features",
                build_feature_transformer(
                    self.engineered, tolerate_unknown_type=missing_strategy == "native"
                ),
            ),
            ("array", FunctionTransformer(_frame_to_array)),
            ("model", model),
        ]
        return Pipeline(steps)


def build_candidates(seed: int) -> dict[str, Candidate]:
    """All candidates of ADR 0006, keyed by name, seeded from ``seed``."""

    def logistic(weight: str | None) -> Callable[[], Any]:
        return lambda: LogisticRegression(C=1.0, max_iter=1000, class_weight=weight)

    def forest(weight: str | None) -> Callable[[], Any]:
        return lambda: RandomForestClassifier(
            n_estimators=300, class_weight=weight, random_state=seed, n_jobs=1
        )

    def hist_boosting(weight: str | None) -> Callable[[], Any]:
        return lambda: HistGradientBoostingClassifier(class_weight=weight, random_state=seed)

    def lightgbm(weight: str | None) -> Callable[[], Any]:
        def make() -> Any:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(class_weight=weight, verbose=-1, n_jobs=1, random_state=seed)

        return make

    def xgboost(weight: str | None) -> Callable[[], Any]:
        def make() -> Any:
            from xgboost import XGBClassifier

            model = XGBClassifier(tree_method="hist", n_jobs=1, random_state=seed)
            return model if weight is None else BalancedSampleWeights(model)

        return make

    makers: list[tuple[str, Role, Callable[[], Any]]] = [
        ("dummy", "baseline", lambda: DummyClassifier(strategy="prior")),
        (
            "isolation_forest",
            "reference",
            lambda: NoveltyScorer(IsolationForest(n_estimators=300, random_state=seed, n_jobs=1)),
        ),
        (
            "one_class_svm",
            "reference",
            lambda: NoveltyScorer(OneClassSVM(nu=ONE_CLASS_SVM_NU, gamma="scale")),
        ),
    ]
    factories = {
        "logistic": logistic,
        "random_forest": forest,
        "hist_gradient_boosting": hist_boosting,
        "lightgbm": lightgbm,
        "xgboost": xgboost,
    }
    for family in FAMILIES:
        unweighted, weighted = variant_names(family)
        makers.append((unweighted, "selectable", factories[family](None)))
        makers.append((weighted, "selectable", factories[family]("balanced")))
    return {name: Candidate(name, role, maker, seed) for name, role, maker in makers}
