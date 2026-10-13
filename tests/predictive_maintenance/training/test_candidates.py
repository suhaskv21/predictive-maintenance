from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

from predictive_maintenance.data import schema
from predictive_maintenance.preprocessing.features import ENGINEERED_FEATURES
from predictive_maintenance.training.candidates import (
    FAMILIES,
    SELECTION_ORDER,
    SWITCHABLE_FAMILIES,
    TREE_FAMILIES,
    BalancedSampleWeights,
    Candidate,
    NoveltyScorer,
    build_candidates,
    variant_names,
)


def readings(n: int = 400, failures: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    torque = rng.normal(40, 10, n)
    target = np.zeros(n, dtype=int)
    target[np.argsort(-torque)[:failures]] = 1
    return pl.DataFrame(
        {
            "Type": rng.choice(list(schema.TYPE_LEVELS), n),
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 1, n),
            "Rotational speed [rpm]": rng.normal(1500, 100, n),
            "Torque [Nm]": torque,
            "Tool wear [min]": rng.uniform(0, 250, n),
            schema.TARGET: target,
        }
    )


class RecordingDetector(BaseEstimator):  # type: ignore[misc]
    def fit(self, X: np.ndarray, y: object = None) -> "RecordingDetector":  # noqa: N803
        self.n_fitted_ = len(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.zeros(len(X))


def test_candidate_set_matches_the_adr() -> None:
    candidates = build_candidates(seed=1)
    families = ("logistic", "random_forest", "hist_gradient_boosting", "lightgbm", "xgboost")
    expected = {"dummy", "isolation_forest", "one_class_svm"} | {
        f"{family}_{weighting}" for family in families for weighting in ("unweighted", "weighted")
    }
    assert set(candidates) == expected
    assert {n for n, c in candidates.items() if c.role == "reference"} == {
        "isolation_forest",
        "one_class_svm",
    }
    assert candidates["dummy"].role == "baseline"
    assert set(SELECTION_ORDER) == {n for n, c in candidates.items() if c.role == "selectable"}


def test_selection_order_goes_simplest_first_with_unweighted_before_weighted() -> None:
    assert SELECTION_ORDER[:4] == (
        "logistic_unweighted",
        "logistic_weighted",
        "random_forest_unweighted",
        "random_forest_weighted",
    )
    assert SELECTION_ORDER[-2:] == ("xgboost_unweighted", "xgboost_weighted")


def test_every_switchable_family_is_a_candidate_family() -> None:
    assert set(SWITCHABLE_FAMILIES) <= set(FAMILIES)
    assert "logistic" not in SWITCHABLE_FAMILIES


def test_hist_gradient_boosting_is_compared_but_not_offered() -> None:
    assert "hist_gradient_boosting" in TREE_FAMILIES
    assert "hist_gradient_boosting" not in SWITCHABLE_FAMILIES
    assert set(SWITCHABLE_FAMILIES) == {"random_forest", "lightgbm", "xgboost"}


@pytest.mark.parametrize("name", list(build_candidates(seed=1)))
def test_every_candidate_fits_and_scores_one_value_per_reading(name: str) -> None:
    data = readings()
    x, y = data.select(schema.FEATURE_COLUMNS), data[schema.TARGET].to_numpy()
    pipeline = build_candidates(seed=1)[name].make_pipeline().fit(x, y)
    scores = (
        pipeline.predict_proba(x)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else pipeline.decision_function(x)
    )
    assert scores.shape == (400,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    "family", ["random_forest", "hist_gradient_boosting", "lightgbm", "xgboost"]
)
def test_class_weighting_raises_predicted_risk_for_the_rare_class(family: str) -> None:
    data = readings()
    x, y = data.select(schema.FEATURE_COLUMNS), data[schema.TARGET].to_numpy()
    candidates = build_candidates(seed=1)
    plain, weighted = (
        candidates[name].make_pipeline().fit(x, y).predict_proba(x)[:, 1].mean()
        for name in variant_names(family)
    )
    assert weighted > plain


def test_balanced_sample_weights_use_only_the_rows_it_is_fitted_on() -> None:
    seen: list[np.ndarray] = []

    class RecordingClassifier(BaseEstimator):  # type: ignore[misc]
        def fit(
            self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
        ) -> "RecordingClassifier":
            seen.append(sample_weight)
            self.classes_ = np.unique(y)
            return self

    x = np.zeros((10, 2))
    BalancedSampleWeights(RecordingClassifier()).fit(x, np.array([0] * 8 + [1] * 2))
    weights = seen[0]
    assert weights[0] == pytest.approx(10 / (2 * 8))
    assert weights[-1] == pytest.approx(10 / (2 * 2))


def test_reference_candidates_have_no_probabilities() -> None:
    candidates = build_candidates(seed=1)
    for name in ("isolation_forest", "one_class_svm"):
        assert not hasattr(candidates[name].make_pipeline(), "predict_proba")


def test_novelty_scorer_sees_only_non_failures() -> None:
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([0] * 7 + [1] * 3)
    scorer = NoveltyScorer(RecordingDetector()).fit(x, y)
    assert scorer.detector_.n_fitted_ == 7


def test_random_forest_is_reproducible_for_a_seed() -> None:
    data = readings()
    x, y = data.select(schema.FEATURE_COLUMNS), data[schema.TARGET].to_numpy()
    first = build_candidates(1)["random_forest_unweighted"].make_pipeline().fit(x, y)
    again = build_candidates(1)["random_forest_unweighted"].make_pipeline().fit(x, y)
    assert np.array_equal(first.predict_proba(x), again.predict_proba(x))


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_wraps_the_model_and_keeps_probabilities_valid(method: str) -> None:
    data = readings()
    x, y = data.select(schema.FEATURE_COLUMNS), data[schema.TARGET].to_numpy()
    pipeline = build_candidates(1)["logistic_weighted"].make_pipeline(calibration=method)
    assert isinstance(pipeline.named_steps["model"], CalibratedClassifierCV)
    probabilities = pipeline.fit(x, y).predict_proba(x)[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_unknown_calibration_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="calibration"):
        build_candidates(1)["dummy"].make_pipeline(calibration="platt")


def test_pipeline_ignores_outcome_and_identifier_columns() -> None:
    data = readings().with_columns(TWF=pl.col(schema.TARGET), UDI=pl.int_range(pl.len()))
    pipeline = build_candidates(1)["logistic_unweighted"].make_pipeline()
    pipeline.fit(data.select(schema.FEATURE_COLUMNS), data[schema.TARGET].to_numpy())
    assert (
        pipeline.named_steps["array"]
        .transform(pipeline.named_steps["features"].transform(data.select(schema.FEATURE_COLUMNS)))
        .shape[1]
        == 3 + 5 + 3
    )


def test_engineered_features_default_to_all_three_and_can_be_dropped() -> None:
    data = readings()
    x, y = data.select(schema.FEATURE_COLUMNS), data[schema.TARGET].to_numpy()
    candidate = build_candidates(1)["logistic_unweighted"]

    def n_columns(c: Candidate) -> int:
        pipeline = c.make_pipeline().fit(x, y)
        return int(pipeline.named_steps["array"].transform(pipeline[0].transform(x)).shape[1])

    assert candidate.engineered == ENGINEERED_FEATURES
    assert n_columns(candidate) == 3 + 5 + 3
    assert n_columns(replace(candidate, engineered=())) == 3 + 5
