import numpy as np
import polars as pl
import pytest

from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.compare import (
    OperationalCost,
    calibration_decision,
    compare_candidates,
    corrected_se,
    cv_folds,
    derived_seeds,
    feature_check_verdict,
    measure_operational_cost,
    out_of_fold_scores,
    paired_deltas,
    reliability_table,
    select_model,
    select_variants,
    train_size_penalty,
)
from predictive_maintenance.evaluation.split import repeated_stratified_cv
from predictive_maintenance.training.candidates import build_candidates


def readings(n: int = 600, failures: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    torque = rng.normal(40, 10, n)
    target = np.zeros(n, dtype=int)
    target[np.argsort(-(torque + rng.normal(0, 4, n)))[:failures]] = 1
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


def table(**columns: list[float]) -> pl.DataFrame:
    """Scores table with one raw-variant row per fold for each named candidate."""
    rows = [
        (name, "raw", fold, value, None, 0.0)
        for name, values in columns.items()
        for fold, value in enumerate(values)
    ]
    return pl.DataFrame(
        rows,
        schema=["candidate", "variant", "fold", "pr_auc", "brier", "fit_seconds"],
        orient="row",
    )


def brier_table(**variants: float) -> pl.DataFrame:
    rows = [
        ("m", name, fold, 0.5, value, 0.0) for name, value in variants.items() for fold in range(3)
    ]
    return pl.DataFrame(
        rows,
        schema=["candidate", "variant", "fold", "pr_auc", "brier", "fit_seconds"],
        orient="row",
    )


def noisy(base: float, n: int = 25, seed: int = 0, spread: float = 0.001) -> list[float]:
    return list(base + np.random.default_rng(seed).normal(0, spread, n))


def test_compare_scores_every_candidate_on_every_fold() -> None:
    train = readings()
    folds = cv_folds(train, repeated_stratified_cv(n_splits=3, n_repeats=2, seed=1))
    candidates = build_candidates(1)
    chosen = [candidates[n] for n in ("dummy", "logistic_unweighted", "isolation_forest")]
    scores = compare_candidates(chosen, train, folds, n_jobs=1)
    assert scores.height == 3 * 6
    assert scores.columns == ["candidate", "variant", "fold", "pr_auc", "brier", "fit_seconds"]
    assert scores["pr_auc"].is_between(0, 1).all()
    reference = scores.filter(pl.col("candidate") == "isolation_forest")
    assert reference["brier"].null_count() == 6


def test_a_model_with_signal_beats_the_dummy_baseline() -> None:
    train = readings()
    folds = cv_folds(train, repeated_stratified_cv(n_splits=3, n_repeats=1, seed=1))
    candidates = build_candidates(1)
    scores = compare_candidates(
        [candidates["dummy"], candidates["logistic_unweighted"]], train, folds, n_jobs=1
    )
    assert paired_deltas(scores, "logistic_unweighted", "dummy").mean() > 0.2


def test_calibration_variants_get_their_own_rows() -> None:
    train = readings()
    folds = cv_folds(train, repeated_stratified_cv(n_splits=3, n_repeats=1, seed=1))
    scores = compare_candidates(
        [build_candidates(1)["logistic_weighted"]],
        train,
        folds,
        variants=(None, "sigmoid"),
        n_jobs=1,
    )
    assert sorted(scores["variant"].unique()) == ["raw", "sigmoid"]


def test_folds_outside_the_train_partition_are_rejected() -> None:
    train = readings()
    bad = [(np.arange(10), np.array([train.height]))]
    with pytest.raises(ValueError, match="outside the train partition"):
        compare_candidates([build_candidates(1)["dummy"]], train, bad, n_jobs=1)


def test_empty_folds_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        compare_candidates([build_candidates(1)["dummy"]], readings(), [], n_jobs=1)


def test_paired_deltas_align_on_fold() -> None:
    scores = table(a=[0.5, 0.6, 0.7], b=[0.4, 0.4, 0.4])
    assert paired_deltas(scores, "a", "b") == pytest.approx([0.1, 0.2, 0.3])


def test_corrected_se_adds_the_test_train_ratio() -> None:
    deltas = np.array([0.0, 0.2] * 12 + [0.1])
    naive = np.std(deltas, ddof=1) / np.sqrt(25)
    expected = np.sqrt((1 / 25 + 1 / 4) * np.var(deltas, ddof=1))
    assert corrected_se(deltas, n_cv_folds=5) == pytest.approx(expected)
    assert corrected_se(deltas, n_cv_folds=5) > 2 * naive


ORDER = ["simple", "middle", "complex"]


def test_simplest_model_stays_when_gain_is_below_the_margin() -> None:
    scores = table(simple=noisy(0.80), middle=noisy(0.81, seed=1), complex=noisy(0.815, seed=2))
    selection = select_model(scores, ORDER, n_cv_folds=5)
    assert selection.winner == "simple"
    assert [s.replaced for s in selection.steps] == [False, False]


def test_complex_model_replaces_incumbent_when_gain_exceeds_the_margin() -> None:
    scores = table(simple=noisy(0.50), middle=noisy(0.80, seed=1), complex=noisy(0.805, seed=2))
    selection = select_model(scores, ORDER, n_cv_folds=5)
    assert selection.winner == "middle"
    assert [s.challenger for s in selection.steps] == ["middle", "complex"]
    assert selection.steps[1].incumbent == "middle"


def test_noise_larger_than_the_margin_raises_the_threshold() -> None:
    noise = np.random.default_rng(3).normal(0, 0.3, 25)
    scores = table(simple=[0.5] * 25, middle=list(0.525 + noise))
    step = select_model(scores, ["simple", "middle"], n_cv_folds=5).steps[0]
    assert step.threshold == pytest.approx(step.corrected_se)
    assert step.threshold > 0.02


def test_select_model_needs_candidates() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_model(table(a=[0.5]), [], n_cv_folds=5)


def test_weighting_is_kept_per_family_only_when_it_beats_the_margin() -> None:
    scores = table(
        a_unweighted=noisy(0.50),
        a_weighted=noisy(0.60, seed=1),
        b_unweighted=noisy(0.80, seed=2),
        b_weighted=noisy(0.805, seed=3),
    )
    chosen = select_variants(scores, ["a", "b"], n_cv_folds=5)
    assert chosen["a"].winner == "a_weighted"
    assert chosen["b"].winner == "b_unweighted"


def test_calibration_skipped_when_no_method_gains_enough() -> None:
    decision = calibration_decision(brier_table(raw=0.030, sigmoid=0.0295, isotonic=0.0290), "m")
    assert decision.method is None


def test_sigmoid_preferred_when_isotonic_gains_nothing_more() -> None:
    decision = calibration_decision(brier_table(raw=0.030, sigmoid=0.020, isotonic=0.0195), "m")
    assert decision.method == "sigmoid"
    assert decision.relative_brier_gain["sigmoid"] == pytest.approx(1 / 3)


def test_isotonic_chosen_only_when_it_beats_sigmoid_by_the_margin() -> None:
    decision = calibration_decision(brier_table(raw=0.030, sigmoid=0.020, isotonic=0.017), "m")
    assert decision.method == "isotonic"


def test_reliability_table_bins_have_equal_counts_and_track_calibration() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < p).astype(int)
    result = reliability_table(y, p, n_bins=10)
    assert result["n"].to_list() == [2000] * 10
    assert (result["mean_predicted"] - result["observed_rate"]).abs().max() < 0.05  # type: ignore[operator]


def test_out_of_fold_scores_cover_every_train_reading() -> None:
    train = readings()
    folds = cv_folds(train, repeated_stratified_cv(n_splits=3, n_repeats=2, seed=1))
    scores = out_of_fold_scores(build_candidates(1)["logistic_unweighted"], train, folds)
    assert scores.shape == (train.height,)
    assert ((scores > 0) & (scores < 1)).all()


def test_operational_cost_is_measured_and_vetoes_apply() -> None:
    cost = measure_operational_cost(
        build_candidates(1)["logistic_unweighted"], readings(), n_calls=5
    )
    assert cost.fit_seconds > 0 and cost.latency_ms > 0 and cost.artifact_mb > 0
    assert OperationalCost(1.0, 99.0, 49.0).passes_vetoes()
    assert not OperationalCost(1.0, 101.0, 1.0).passes_vetoes()
    assert not OperationalCost(1.0, 1.0, 51.0).passes_vetoes()


def test_train_size_penalty_is_reproducible_and_uses_the_train_frame_only() -> None:
    train = readings()
    candidate = build_candidates(1)["logistic_unweighted"]
    seeds = derived_seeds(1, 3, 4)
    first = train_size_penalty(candidate, train, seeds, n_jobs=1)
    assert first.columns == ["seed", "pr_auc_full", "pr_auc_reduced", "penalty"]
    assert first["seed"].to_list() == seeds
    assert first.equals(train_size_penalty(candidate, train, seeds, n_jobs=1))


def test_derived_seeds_are_deterministic_and_stream_specific() -> None:
    assert derived_seeds(42, 1, 5) == derived_seeds(42, 1, 5)
    assert derived_seeds(42, 1, 5) != derived_seeds(42, 2, 5)
    assert len(set(derived_seeds(42, 1, 50))) == 50


@pytest.mark.parametrize(
    ("delta", "verdict"),
    [
        (0.08, "agrees"),
        (-0.005, "agrees"),
        (-0.0051, "noise"),
        (-0.0199, "noise"),
        (-0.02, "disagrees"),
        (-0.1, "disagrees"),
    ],
)
def test_feature_check_verdict_follows_the_pre_declared_margins(delta: float, verdict: str) -> None:
    assert feature_check_verdict(delta) == verdict
