import itertools
import math

import numpy as np
import polars as pl
import pytest

from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.compare import cv_folds, out_of_fold_scores
from predictive_maintenance.evaluation.split import repeated_stratified_cv
from predictive_maintenance.evaluation.threshold import (
    MIN_BAND_FAILURES,
    BandCheck,
    Bands,
    OperatingPoint,
    band_table,
    bootstrap_metric_cis,
    build_operating_point,
    check_bands,
    compare_costs,
    cv_operating_metrics,
    estimate_threshold,
    expected_cost,
    flag_everything_cost,
    flag_nothing_cost,
    fold_expected_costs,
    interval_cost_excess,
    is_flat_optimum,
    is_viable,
    mean_cost,
    operating_metrics,
    pick_threshold,
    ranking_metrics,
    repeat_scores,
    risk_floor,
    wilson_interval,
)
from predictive_maintenance.training.candidates import build_candidates


def readings(n: int = 300, failures: int = 30) -> pl.DataFrame:
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


def noisy_scores(n: int = 1000, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores where the failure rate rises with the score."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < scores**2).astype(int)
    return labels, scores


def test_expected_cost_charges_each_missed_failure_the_ratio_and_each_false_alarm_one() -> None:
    y = np.array([1, 0, 1, 0])
    flagged = np.array([True, True, False, False])
    assert expected_cost(y, flagged, 10) == pytest.approx((10 * 1 + 1) / 4)


def test_baseline_costs_for_flagging_nothing_and_everything() -> None:
    y = np.array([1, 0, 0, 0])
    assert flag_nothing_cost(y, 10) == pytest.approx(10 / 4)
    assert flag_everything_cost(y) == pytest.approx(3 / 4)


@pytest.mark.parametrize("bad", [[0, 2, 1], [], [[0, 1], [1, 0]]])
def test_labels_must_be_a_non_empty_one_dimensional_binary_array(bad: list[object]) -> None:
    with pytest.raises(ValueError, match="0/1"):
        flag_everything_cost(np.array(bad))


def test_scores_must_match_the_labels_in_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        pick_threshold(np.array([0, 1]), np.array([0.1, 0.2, 0.3]), 10)


def test_pick_threshold_sits_in_the_middle_of_the_gap_between_classes() -> None:
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert pick_threshold(y, scores, 10) == pytest.approx(0.55)


def test_pick_threshold_never_splits_tied_scores() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.9, 0.9, 0.1, 0.1])
    flagged = scores >= pick_threshold(y, scores, 10)
    assert flagged[0] == flagged[1]
    assert flagged[2] == flagged[3]
    assert flagged[4] == flagged[5]


def test_pick_threshold_without_failures_flags_only_the_highest_score() -> None:
    assert pick_threshold(np.zeros(3, dtype=int), np.array([0.1, 0.2, 0.3]), 10) == pytest.approx(
        0.25
    )


@pytest.mark.parametrize("ratio", [1, 5, 10, 20])
def test_pick_threshold_reaches_the_minimum_expected_cost(ratio: int) -> None:
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=200) < 0.2).astype(int)
    scores = np.round(np.clip(y * 0.3 + rng.uniform(0, 0.7, 200), 0, 1), 1)  # many ties
    chosen = expected_cost(y, scores >= pick_threshold(y, scores, ratio), ratio)
    best = min(expected_cost(y, scores >= t, ratio) for t in np.unique(scores))
    assert chosen == pytest.approx(best)


def test_a_higher_cost_ratio_never_raises_the_threshold() -> None:
    y, scores = noisy_scores()
    t5, t10, t20 = (pick_threshold(y, scores, r) for r in (5, 10, 20))
    assert t20 <= t10 <= t5


def test_estimate_threshold_reports_median_repeat_range_and_bootstrap_interval() -> None:
    y, a = noisy_scores()
    _, b = noisy_scores(seed=2)
    rows = np.vstack([a, b, a])
    estimate = estimate_threshold(y, rows, 10, n_resamples=50, seed=0)
    assert estimate.cost_ratio == 10
    assert len(estimate.per_repeat) == 3
    assert estimate.median == pytest.approx(float(np.median(estimate.per_repeat)))
    assert estimate.repeat_range == (min(estimate.per_repeat), max(estimate.per_repeat))
    assert estimate.interval[0] <= estimate.interval[1]


def test_estimate_threshold_is_reproducible_for_a_seed() -> None:
    y, scores = noisy_scores()
    rows = scores[None, :]
    first = estimate_threshold(y, rows, 10, n_resamples=30, seed=5)
    assert first == estimate_threshold(y, rows, 10, n_resamples=30, seed=5)
    assert first.interval != estimate_threshold(y, rows, 10, n_resamples=30, seed=6).interval


def test_mean_cost_averages_over_the_repeats() -> None:
    y = np.array([1, 0, 1, 0])
    rows = np.array([[0.9, 0.1, 0.9, 0.1], [0.9, 0.9, 0.1, 0.1]])
    assert mean_cost(y, rows, 0.5, 10) == pytest.approx((0 + (10 + 1) / 4) / 2)


def test_interval_cost_excess_is_zero_for_a_degenerate_interval_and_flat_flags_it() -> None:
    y, scores = noisy_scores()
    rows = scores[None, :]
    estimate = estimate_threshold(y, rows, 10, n_resamples=20, seed=0)
    degenerate = type(estimate)(
        cost_ratio=10,
        per_repeat=estimate.per_repeat,
        median=estimate.median,
        repeat_range=estimate.repeat_range,
        interval=(estimate.median, estimate.median),
    )
    assert interval_cost_excess(y, rows, degenerate) == (0.0, 0.0)
    assert is_flat_optimum((0.0, 0.05))
    assert not is_flat_optimum((0.0, 0.2))


def test_operating_metrics_in_operational_terms() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1])
    m = operating_metrics(y, scores, 0.5, 10)
    assert m.recall == pytest.approx(0.5)
    assert m.precision == pytest.approx(0.5)
    assert m.alerts_per_1000 == pytest.approx(1000 * 2 / 6)
    assert m.false_alarms_per_caught == pytest.approx(1.0)
    assert m.cost_per_1000 == pytest.approx(1000 * 11 / 6)
    assert m.flag_nothing_per_1000 == pytest.approx(1000 * 20 / 6)
    assert m.flag_everything_per_1000 == pytest.approx(1000 * 4 / 6)
    assert m.saving_vs_flag_nothing == pytest.approx(1 - 11 / 20)


def test_operating_metrics_are_nan_when_nothing_is_flagged() -> None:
    m = operating_metrics(np.array([1, 0]), np.array([0.2, 0.1]), 2.0, 10)
    assert m.recall == 0
    assert math.isnan(m.precision)
    assert math.isnan(m.false_alarms_per_caught)
    assert m.saving_vs_flag_nothing == pytest.approx(0.0)


def test_cv_operating_metrics_averages_the_repeats() -> None:
    y = np.array([1, 1, 0, 0])
    good = np.array([0.9, 0.9, 0.1, 0.1])
    bad = np.array([0.1, 0.9, 0.1, 0.1])
    m = cv_operating_metrics(y, np.vstack([good, bad]), 0.5, 10)
    assert m.recall == pytest.approx(0.75)
    assert m.threshold == 0.5


def test_a_model_is_viable_only_if_it_beats_flag_nothing_at_every_ratio() -> None:
    y = np.array([1] * 10 + [0] * 90)
    scores = np.where(y == 1, 0.9, 0.1)
    useful = [operating_metrics(y, scores, 0.5, r) for r in (5, 10, 20)]
    useless = [operating_metrics(y, np.zeros(100), 0.5, r) for r in (5, 10, 20)]
    assert is_viable(useful)
    assert not is_viable(useless)
    assert not is_viable([*useful, useless[0]])


def test_ranking_metrics_for_a_perfect_ranking() -> None:
    m = ranking_metrics(np.array([0, 0, 1, 1]), np.array([0.0, 0.1, 0.9, 1.0]))
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx((0.1**2 + 0.1**2) / 4 * 1)


def test_wilson_interval_matches_known_values() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(3.841459 / (10 + 3.841459), rel=1e-3)
    low, high = wilson_interval(5, 10)
    assert low + high == pytest.approx(1.0)


def test_bands_need_room_for_three_and_assign_at_the_edges() -> None:
    with pytest.raises(ValueError, match="elevated < high"):
        Bands(elevated=0.3, high=0.3)
    bands = Bands(elevated=0.1, high=0.5)
    labels = bands.assign(np.array([0.0, 0.1, 0.49, 0.5, 0.9]))
    assert list(labels) == ["Low", "Elevated", "Elevated", "High", "High"]


def test_band_table_reports_mean_counts_and_observed_rates() -> None:
    y = np.array([0, 0, 1, 0, 1, 1])
    rows = np.array([[0.0, 0.05, 0.2, 0.3, 0.9, 0.9], [0.0, 0.05, 0.2, 0.3, 0.9, 0.05]])
    table = band_table(y, rows, Bands(0.1, 0.5)).to_dict(as_series=False)
    assert table["band"] == ["Low", "Elevated", "High"]
    assert table["readings"] == [2.5, 2.0, 1.5]
    assert table["failures"] == [0.5, 1.0, 1.5]
    assert table["observed_rate"][0] == pytest.approx(0.5 / 2.5)
    assert all(
        lo <= r <= hi
        for lo, r, hi in zip(
            table["wilson_low"], table["observed_rate"], table["wilson_high"], strict=True
        )
    )


def make_table(failures: list[float], readings: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "band": ["Low", "Elevated", "High"],
            "readings": readings,
            "failures": failures,
            "observed_rate": [
                f / r if r else math.nan for f, r in zip(failures, readings, strict=True)
            ],
            "wilson_low": [0.0] * 3,
            "wilson_high": [1.0] * 3,
        }
    )


def test_bands_ship_when_rates_rise_and_flagged_bands_have_enough_failures() -> None:
    check = check_bands(make_table([5, 20, 100], [1000, 100, 120]))
    assert check == BandCheck(True, "")


def test_bands_do_not_ship_with_too_few_failures_in_a_flagged_band() -> None:
    check = check_bands(make_table([5, MIN_BAND_FAILURES - 1, 100], [1000, 100, 120]))
    assert not check.ships
    assert "Elevated" in check.reason


def test_bands_do_not_ship_when_the_rate_does_not_rise() -> None:
    check = check_bands(make_table([5, 20, 15], [1000, 100, 120]))
    assert not check.ships
    assert "increase" in check.reason


def test_risk_floor_is_the_rounded_up_wilson_bound_of_the_lowest_half() -> None:
    y = np.zeros(100, dtype=int)
    y[-2:] = 1
    rows = np.linspace(0.0, 1.0, 100)[None, :]
    assert risk_floor(y, rows) == pytest.approx(0.08)


def test_risk_floor_never_drops_below_one_percent() -> None:
    y = np.zeros(10_000, dtype=int)
    y[-1] = 1
    assert risk_floor(y, np.linspace(0.0, 1.0, 10_000)[None, :]) == pytest.approx(0.01)


def test_operating_point_round_trips_through_json() -> None:
    point = OperatingPoint(
        model="random_forest_unweighted",
        default_cost_ratio=10.0,
        thresholds={5.0: 0.3, 10.0: 0.15, 20.0: 0.07},
        bands=Bands(0.15, 0.3),
        risk_floor=0.02,
    )
    assert OperatingPoint.from_json(point.to_json()) == point
    assert point.threshold == 0.15
    without_bands = OperatingPoint("m", 10.0, {10.0: 0.1}, None, 0.01)
    assert OperatingPoint.from_json(without_bands.to_json()) == without_bands


def test_operating_point_requires_a_threshold_at_the_default_ratio() -> None:
    with pytest.raises(ValueError, match="default cost ratio"):
        OperatingPoint("m", 10.0, {5.0: 0.3}, None, 0.01)


def test_build_operating_point_uses_default_and_smallest_ratio_as_band_edges() -> None:
    y, scores = noisy_scores(2000)
    thresholds = {5.0: 0.6, 10.0: 0.3, 20.0: 0.1}
    point, table, check = build_operating_point(
        "m", y, scores[None, :], thresholds=thresholds, default_cost_ratio=10.0, floor=0.02
    )
    assert check.ships
    assert point.bands == Bands(elevated=0.3, high=0.6)
    assert table["band"].to_list() == ["Low", "Elevated", "High"]


def test_build_operating_point_drops_bands_that_fail_the_check() -> None:
    y = np.zeros(1000, dtype=int)
    y[:3] = 1
    scores = np.linspace(0, 1, 1000)
    point, _, check = build_operating_point(
        "m",
        y,
        scores[None, :],
        thresholds={5.0: 0.6, 10.0: 0.3},
        default_cost_ratio=10.0,
        floor=0.01,
    )
    assert not check.ships
    assert point.bands is None


def test_build_operating_point_fails_loudly_when_the_edges_collapse() -> None:
    y, scores = noisy_scores()
    with pytest.raises(ValueError, match="elevated < high"):
        build_operating_point(
            "m",
            y,
            scores[None, :],
            thresholds={5.0: 0.3, 10.0: 0.3},
            default_cost_ratio=10.0,
            floor=0.01,
        )


def test_fold_expected_costs_scores_each_fold_with_its_repeats_scores() -> None:
    y = np.array([1, 0, 1, 0])
    rows = np.array([[0.9, 0.1, 0.1, 0.1]])
    folds = [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]
    costs = fold_expected_costs(y, rows, folds, n_splits=2, threshold=0.5, cost_ratio=10)
    assert list(costs) == pytest.approx([0.0, 5.0])


def test_compare_costs_needs_more_than_the_margin_and_the_se() -> None:
    incumbent = np.full(25, 1.0)
    assert compare_costs(incumbent - 0.5, incumbent, n_cv_folds=5).cheaper
    assert not compare_costs(incumbent - 0.05, incumbent, n_cv_folds=5).cheaper
    savings = 0.2 + np.tile([-0.5, 0.5], 13)[:25]
    assert not compare_costs(incumbent - savings, incumbent, n_cv_folds=5).cheaper


def test_repeat_scores_gives_one_pooled_out_of_fold_score_per_reading_per_repeat() -> None:
    train = readings()
    folds = cv_folds(train, repeated_stratified_cv(n_splits=3, n_repeats=2, seed=0))
    candidate = build_candidates(0)["logistic_unweighted"]
    rows = repeat_scores(candidate, train, folds, n_splits=3, n_jobs=1)
    assert rows.shape == (2, train.height)
    assert ((rows >= 0) & (rows <= 1)).all()
    assert not np.allclose(rows[0], rows[1])
    averaged = out_of_fold_scores(candidate, train, folds)
    assert np.allclose(rows.mean(axis=0), averaged)


def test_repeat_scores_rejects_folds_that_are_not_whole_repeats() -> None:
    train = readings()
    folds = cv_folds(train, repeated_stratified_cv(n_splits=3, n_repeats=2, seed=0))
    candidate = build_candidates(0)["logistic_unweighted"]
    with pytest.raises(ValueError, match="whole number of repeats"):
        repeat_scores(candidate, train, folds[:-1], n_splits=3, n_jobs=1)


def test_repeat_scores_rejects_a_repeat_that_misses_readings() -> None:
    train = readings()
    folds = cv_folds(train, repeated_stratified_cv(n_splits=3, n_repeats=1, seed=0))
    candidate = build_candidates(0)["logistic_unweighted"]
    with pytest.raises(ValueError, match="exactly once"):
        repeat_scores(candidate, train, [folds[0], folds[0], folds[2]], n_splits=3, n_jobs=1)


def test_bootstrap_cis_cover_the_estimate_and_freeze_the_threshold() -> None:
    y, scores = noisy_scores(600)
    cis = bootstrap_metric_cis(y, scores, threshold=0.5, cost_ratio=10, n_resamples=100, seed=0)
    assert cis["metric"].to_list() == [
        "pr_auc",
        "roc_auc",
        "brier",
        "recall",
        "precision",
        "expected_cost_per_1000",
    ]
    for row in cis.iter_rows(named=True):
        assert row["low"] <= row["estimate"] <= row["high"] or row["metric"] == "precision"
    recall = cis.filter(pl.col("metric") == "recall").row(0, named=True)
    assert recall["estimate"] == pytest.approx(operating_metrics(y, scores, 0.5, 10).recall)


def test_bootstrap_cis_are_reproducible_for_a_seed_and_tight_for_a_perfect_model() -> None:
    y = np.array([0] * 40 + [1] * 10)
    perfect = np.where(y == 1, 0.9, 0.1)
    first = bootstrap_metric_cis(y, perfect, threshold=0.5, cost_ratio=10, n_resamples=40, seed=3)
    again = bootstrap_metric_cis(y, perfect, threshold=0.5, cost_ratio=10, n_resamples=40, seed=3)
    assert first.equals(again)
    pr = first.filter(pl.col("metric") == "pr_auc").row(0, named=True)
    assert (pr["low"], pr["estimate"], pr["high"]) == (1.0, 1.0, 1.0)
    y2, scores = noisy_scores(300)
    other = bootstrap_metric_cis(y2, scores, threshold=0.5, cost_ratio=10, n_resamples=40, seed=4)
    other2 = bootstrap_metric_cis(y2, scores, threshold=0.5, cost_ratio=10, n_resamples=40, seed=5)
    assert not other.equals(other2)


def test_every_threshold_pick_is_a_flag_set_the_curve_can_reach() -> None:
    y, scores = noisy_scores(200)
    scores = np.round(scores, 1)
    thresholds = [pick_threshold(y, scores, r) for r in (5, 10, 20)]
    reachable = {tuple(scores >= t) for t in itertools.chain(np.unique(scores), [2.0])}
    assert all(tuple(scores >= t) in reachable for t in thresholds)
