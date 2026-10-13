import math
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from predictive_maintenance.config import MissingDataConfig, load_config
from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.missingness import (
    ALL_ROWS,
    COMPLETE,
    INPUTS,
    RESULT_SCHEMA,
    SINGLE_SENSOR_PATTERNS,
    STRESS,
    TORQUE_PATTERN,
    MissingDataPolicy,
    Pattern,
    PatternDegradation,
    all_patterns,
    build_policy,
    choose_strategy,
    evaluate_patterns,
    incomplete_share,
    informative_loss,
    inject_absence,
    pattern_name,
    pattern_repeat_scores,
    pattern_verdicts,
    recoverable_cost,
    refused_share,
    scorable_patterns,
)
from predictive_maintenance.evaluation.threshold import operating_metrics
from predictive_maintenance.training.candidates import build_candidates

TORQUE = "Torque [Nm]"
AIR = "Air temperature [K]"
CFG = MissingDataConfig(
    min_retained_saving=0.9,
    max_recall_drop=0.1,
    flag_rate_ratio_min=0.5,
    flag_rate_ratio_max=2.0,
    challenger_margin=0.05,
    stress_speed_quantile=0.75,
    assumed_missing_rates=(0.05,),
)


def readings(n: int = 400, failures: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    torque = rng.normal(40, 10, n)
    target = np.zeros(n, dtype=int)
    target[np.argsort(-torque)[:failures]] = 1
    return pl.DataFrame(
        {
            "Type": rng.choice(list(schema.TYPE_LEVELS), n),
            AIR: rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 1, n),
            "Rotational speed [rpm]": rng.integers(1200, 2000, n),
            TORQUE: torque,
            "Tool wear [min]": rng.integers(0, 250, n),
            schema.TARGET: target,
        }
    )


def test_there_are_63_distinct_patterns_smallest_first_in_canonical_order() -> None:
    patterns = all_patterns()
    assert len(patterns) == len(set(patterns)) == 2 ** len(INPUTS) - 1 == 63
    assert [len(p) for p in patterns] == sorted(len(p) for p in patterns)
    assert patterns[0] == ("Type",)
    assert patterns[-1] == INPUTS
    assert all(list(p) == [c for c in INPUTS if c in p] for p in patterns)


def test_pattern_names_are_readable_and_distinct() -> None:
    assert pattern_name(()) == COMPLETE
    assert pattern_name((TORQUE, "Tool wear [min]")) == "Torque [Nm]+Tool wear [min]"
    assert len({pattern_name(p) for p in all_patterns()}) == 63


def test_inject_absence_nulls_only_the_pattern_and_leaves_the_input_alone() -> None:
    x = readings().select(schema.FEATURE_COLUMNS)
    out = inject_absence(x, (TORQUE, "Type"))
    assert out[TORQUE].null_count() == out["Type"].null_count() == x.height
    assert out[AIR].equals(x[AIR])
    assert x[TORQUE].null_count() == 0
    assert out.schema == x.schema


def test_inject_absence_can_be_restricted_to_some_rows() -> None:
    x = readings(10).select(schema.FEATURE_COLUMNS)
    rows = np.arange(10) >= 7
    out = inject_absence(x, TORQUE_PATTERN, rows)
    assert out[TORQUE].is_null().to_list() == rows.tolist()
    assert out[TORQUE][:7].equals(x[TORQUE][:7])


@pytest.mark.parametrize("bad", [("Pressure",), (TORQUE, "Pressure")])
def test_inject_absence_rejects_an_unknown_input(bad: Pattern) -> None:
    with pytest.raises(ValueError, match="unknown inputs"):
        inject_absence(readings().select(schema.FEATURE_COLUMNS), bad)


def test_inject_absence_rejects_a_row_mask_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="shape"):
        inject_absence(
            readings(10).select(schema.FEATURE_COLUMNS), TORQUE_PATTERN, np.ones(3, bool)
        )


@pytest.fixture(scope="module")
def evaluated() -> tuple[pl.DataFrame, pl.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    data = readings()
    idx = np.arange(data.height)
    folds = [(idx[idx % 3 != k], idx[idx % 3 == k]) for k in range(3)]
    results = evaluate_patterns(
        build_candidates(1)["random_forest_unweighted"],
        data,
        folds,
        threshold=0.3,
        cost_ratio=10,
        stress_quantile=0.75,
        patterns=[TORQUE_PATTERN, ("Type",)],
        n_jobs=1,
    )
    return data, results, folds


def test_results_have_one_row_per_fold_strategy_and_pattern_plus_baselines(
    evaluated: tuple[pl.DataFrame, pl.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    _, results, _ = evaluated
    assert dict(results.schema) == RESULT_SCHEMA
    baseline = results.filter(pl.col("strategy") == COMPLETE)
    assert baseline.height == 3 * 2  # per fold: all rows and the stress rows
    body = results.filter(pl.col("strategy") != COMPLETE)
    assert (
        body.filter(pl.col("scope") == ALL_ROWS).height == 3 * 3 * 2
    )  # folds x strategies x patterns
    assert body.filter(pl.col("scope") == STRESS).height == 3 * 3  # torque only
    assert set(body.filter(pl.col("scope") == STRESS)["pattern"]) == {pattern_name(TORQUE_PATTERN)}


def test_the_complete_baseline_is_what_the_plain_pipeline_scores_at_the_frozen_threshold(
    evaluated: tuple[pl.DataFrame, pl.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    data, results, folds = evaluated
    a, b = folds[0]
    x, y = data.select(schema.FEATURE_COLUMNS), data[schema.TARGET].to_numpy()
    plain = build_candidates(1)["random_forest_unweighted"].make_pipeline().fit(x[a], y[a])
    expected = operating_metrics(y[b], plain.predict_proba(x[b])[:, 1], 0.3, 10)
    row = results.filter(
        (pl.col("strategy") == COMPLETE) & (pl.col("scope") == ALL_ROWS) & (pl.col("fold") == 0)
    ).row(0, named=True)
    assert row["saving"] == pytest.approx(expected.saving_vs_flag_nothing)
    assert row["recall"] == pytest.approx(expected.recall)
    assert row["rows"] == len(b)


@pytest.mark.parametrize("strategy", ["median", "regression", "native"])
def test_swapping_the_imputer_equals_fitting_the_full_strategy_pipeline(
    strategy: str,
    evaluated: tuple[pl.DataFrame, pl.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    data, results, folds = evaluated
    a, b = folds[1]
    x, y = data.select(schema.FEATURE_COLUMNS), data[schema.TARGET].to_numpy()
    full = (
        build_candidates(1)["random_forest_unweighted"]
        .make_pipeline(missing_strategy=strategy)
        .fit(x[a], y[a])
    )
    scores = full.predict_proba(inject_absence(x[b], TORQUE_PATTERN))[:, 1]
    expected = operating_metrics(y[b], scores, 0.3, 10)
    row = results.filter(
        (pl.col("strategy") == strategy)
        & (pl.col("pattern") == pattern_name(TORQUE_PATTERN))
        & (pl.col("scope") == ALL_ROWS)
        & (pl.col("fold") == 1)
    ).row(0, named=True)
    assert row["alerts_per_1000"] == pytest.approx(expected.alerts_per_1000)
    assert row["cost_per_1000"] == pytest.approx(expected.cost_per_1000)


def test_the_stress_group_is_the_high_speed_rows_judged_against_themselves(
    evaluated: tuple[pl.DataFrame, pl.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    data, results, folds = evaluated
    a, b = folds[0]
    cutoff = np.quantile(data["Rotational speed [rpm]"].to_numpy()[a], 0.75)
    expected_rows = int((data["Rotational speed [rpm]"].to_numpy()[b] >= cutoff).sum())
    stress = results.filter((pl.col("scope") == STRESS) & (pl.col("fold") == 0))
    assert set(stress["rows"]) == {expected_rows}
    assert COMPLETE in set(stress["strategy"])


def test_the_input_data_is_not_modified(
    evaluated: tuple[pl.DataFrame, pl.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    data, _, _ = evaluated
    assert data.null_count().sum_horizontal().item() == 0


def test_evaluation_rejects_bad_folds_and_strategies() -> None:
    data = readings(60, 6)
    candidate = build_candidates(1)["random_forest_unweighted"]
    kwargs = {"threshold": 0.3, "cost_ratio": 10.0, "stress_quantile": 0.75, "n_jobs": 1}
    with pytest.raises(ValueError, match="folds"):
        evaluate_patterns(candidate, data, [], **kwargs)  # type: ignore[arg-type]
    outside = [(np.arange(5), np.arange(55, 61))]
    with pytest.raises(ValueError, match="folds"):
        evaluate_patterns(candidate, data, outside, **kwargs)  # type: ignore[arg-type]
    ok = [(np.arange(30), np.arange(30, 60))]
    with pytest.raises(ValueError, match="strategies"):
        evaluate_patterns(candidate, data, ok, strategies=["reject"], **kwargs)  # type: ignore[arg-type]


FOLDS = range(5)
BASE = {"saving": 0.80, "recall": 0.80, "alerts": 40.0}


def synthetic(
    spec: dict[tuple[str, Pattern], dict[str, float]],
    *,
    stress: dict[str, dict[str, float]] | None = None,
) -> pl.DataFrame:
    """Hand-built results: the complete baseline plus the given strategy/pattern outcomes."""

    def row(strategy: str, pattern: Pattern, scope: str, fold: int, m: dict[str, float]) -> tuple:
        return (
            strategy, pattern_name(pattern), len(pattern), scope, fold, 100, 10,
            m["recall"], m["alerts"], 100 * (1 - m["saving"]), m["saving"], 0.5,
        )  # fmt: skip

    rows = [row(COMPLETE, (), scope, f, BASE) for scope in (ALL_ROWS, STRESS) for f in FOLDS]
    for (strategy, pattern), metrics in spec.items():
        rows += [row(strategy, pattern, ALL_ROWS, f, metrics) for f in FOLDS]
    for strategy, metrics in (stress or {}).items():
        rows += [row(strategy, TORQUE_PATTERN, STRESS, f, metrics) for f in FOLDS]
    return pl.DataFrame(rows, schema=RESULT_SCHEMA, orient="row")


def verdict(frame: pl.DataFrame, strategy: str, pattern: Pattern) -> dict[str, object]:
    return (
        pattern_verdicts(frame, CFG, n_cv_folds=5)
        .filter((pl.col("strategy") == strategy) & (pl.col("pattern") == pattern_name(pattern)))
        .row(0, named=True)
    )


GOOD = {"saving": 0.78, "recall": 0.75, "alerts": 38.0}


def test_a_pattern_that_keeps_saving_recall_and_flag_rate_passes() -> None:
    v = verdict(synthetic({("median", ("Type",)): GOOD}), "median", ("Type",))
    assert v["passes"] and v["scorable"]
    assert v["retained_saving"] == pytest.approx(0.975)
    assert v["recall_drop"] == pytest.approx(0.05)
    assert v["flag_rate_ratio"] == pytest.approx(0.95)


@pytest.mark.parametrize(
    ("change", "failed"),
    [
        ({"saving": 0.70}, "passes_saving"),  # retains 0.875 < 0.90
        ({"recall": 0.69}, "passes_recall"),  # drop 0.11 > 0.10
        ({"alerts": 81.0}, "passes_flag"),  # ratio 2.025 > 2
        ({"alerts": 19.0}, "passes_flag"),  # ratio 0.475 < 0.5
    ],
)
def test_each_margin_alone_can_refuse_a_pattern(change: dict[str, float], failed: str) -> None:
    v = verdict(synthetic({("median", ("Type",)): GOOD | change}), "median", ("Type",))
    assert not v[failed]
    assert not v["passes"] and not v["scorable"]


def test_noise_across_folds_counts_against_the_pattern() -> None:
    frame = synthetic({("median", ("Type",)): GOOD})
    noisy = frame.with_columns(
        pl.when((pl.col("strategy") == "median") & (pl.col("fold") % 2 == 0))
        .then(pl.col("saving") - 0.10)
        .otherwise(pl.col("saving"))
        .alias("saving")
    )
    assert verdict(frame, "median", ("Type",))["passes_saving"]
    assert not verdict(noisy, "median", ("Type",))["passes_saving"]


def test_a_larger_pattern_is_not_scorable_unless_every_smaller_one_is() -> None:
    both = ("Type", AIR)
    frame = synthetic(
        {
            ("median", ("Type",)): GOOD,
            ("median", (AIR,)): GOOD | {"saving": 0.3},
            ("median", ("Type", AIR)): GOOD,
        }
    )
    pattern = tuple(c for c in INPUTS if c in both)
    v = verdict(frame, "median", pattern)
    assert v["passes"]
    assert not v["scorable"]


def test_a_pattern_is_scorable_when_the_whole_chain_below_it_is() -> None:
    frame = synthetic(
        {
            ("median", ("Type",)): GOOD,
            ("median", (AIR,)): GOOD,
            ("median", ("Type", AIR)): GOOD,
        }
    )
    assert verdict(frame, "median", ("Type", AIR))["scorable"]


def test_the_stress_can_veto_the_torque_pattern_and_what_contains_it() -> None:
    spec = {
        ("median", TORQUE_PATTERN): GOOD,
        ("median", ("Type",)): GOOD,
        ("median", ("Type", TORQUE)): GOOD,
    }
    bad_stress = {"median": GOOD | {"saving": 0.4}}
    ok_stress = {"median": GOOD}
    vetoed = synthetic(spec, stress=bad_stress)
    allowed = synthetic(spec, stress=ok_stress)
    assert verdict(allowed, "median", TORQUE_PATTERN)["scorable"]
    assert verdict(allowed, "median", ("Type", TORQUE))["scorable"]
    v = verdict(vetoed, "median", TORQUE_PATTERN)
    assert v["passes"] and not v["stress_ok"] and not v["scorable"]
    assert not verdict(vetoed, "median", ("Type", TORQUE))["scorable"]
    assert verdict(vetoed, "median", ("Type",))["scorable"]


def test_scorable_patterns_lists_what_the_policy_scores_and_reject_scores_nothing() -> None:
    frame = synthetic({("median", ("Type",)): GOOD, ("median", (AIR,)): GOOD | {"saving": 0.3}})
    verdicts = pattern_verdicts(frame, CFG, n_cv_folds=5)
    assert scorable_patterns(verdicts, "median") == [("Type",)]
    assert scorable_patterns(verdicts, "reject") == []


def sensor_spec(
    median: float, regression: float = 0.5, native: float = 0.5
) -> dict[tuple[str, Pattern], dict[str, float]]:
    savings = {"median": median, "regression": regression, "native": native}
    return {
        (strategy, p): GOOD | {"saving": saving}
        for strategy, saving in savings.items()
        for p in SINGLE_SENSOR_PATTERNS
    }


def test_choosing_a_strategy_needs_results_for_every_strategy() -> None:
    frame = synthetic({("median", p): GOOD for p in SINGLE_SENSOR_PATTERNS})
    verdicts = pattern_verdicts(frame, CFG, n_cv_folds=5)
    with pytest.raises(ValueError, match="regression"):
        choose_strategy(frame, verdicts, CFG, n_cv_folds=5)


def test_reject_is_kept_when_no_strategy_makes_any_pattern_scorable() -> None:
    frame = synthetic(sensor_spec(0.5))
    verdicts = pattern_verdicts(frame, CFG, n_cv_folds=5)
    winner, steps = choose_strategy(frame, verdicts, CFG, n_cv_folds=5)
    assert winner == "reject"
    assert steps[0].challenger == "median" and not steps[0].replaced
    assert steps[0].mean_delta > steps[0].threshold  # it gains saving, but nothing is scorable


def test_the_simplest_strategy_that_works_is_chosen() -> None:
    frame = synthetic(sensor_spec(0.78, regression=0.79))
    verdicts = pattern_verdicts(frame, CFG, n_cv_folds=5)
    winner, steps = choose_strategy(frame, verdicts, CFG, n_cv_folds=5)
    assert winner == "median"
    assert [s.replaced for s in steps] == [True, False, False]


def test_a_more_complex_strategy_must_beat_the_incumbent_by_the_margin() -> None:
    frame = synthetic(sensor_spec(0.70, regression=0.78))
    verdicts = pattern_verdicts(frame, CFG, n_cv_folds=5)
    # median retains 0.875 (fails), regression 0.975: the gain over reject exceeds the margin
    assert choose_strategy(frame, verdicts, CFG, n_cv_folds=5)[0] == "regression"
    tight = replace(CFG, challenger_margin=0.99)
    assert choose_strategy(frame, verdicts, tight, n_cv_folds=5)[0] == "reject"


def test_refused_share_is_zero_when_everything_is_scorable_and_one_minus_complete_for_reject() -> (
    None
):
    assert refused_share(0.05, all_patterns()) == pytest.approx(0.0)
    assert refused_share(0.05, []) == pytest.approx(1 - 0.95**6)
    assert incomplete_share(0.05) == pytest.approx(1 - 0.95**6)


def test_refused_share_counts_only_the_patterns_that_are_not_scorable() -> None:
    assert refused_share(0.1, [("Type",)]) == pytest.approx(1 - 0.9**6 - 0.1 * 0.9**5)
    assert refused_share(0.1, [(TORQUE, "Type")]) < refused_share(0.1, [])


def test_informative_loss_compares_the_dropped_group_with_the_whole_train_partition() -> None:
    data = readings(200, 20).with_columns(
        pl.Series("Rotational speed [rpm]", np.arange(200)),
        pl.Series(
            schema.TARGET,
            [
                1 if i in (100, 101, 102, 103, 104, 190, 191, 192, 193, 194) else 0
                for i in range(200)
            ],
        ),
    )
    loss = informative_loss(data, 0.75)
    assert loss.rows_lost == pytest.approx(0.25)
    assert loss.failures_lost == pytest.approx(0.5)
    assert loss.failure_rate_lost == pytest.approx(0.1)
    assert loss.failure_rate_overall == pytest.approx(0.05)
    assert math.isfinite(loss.failure_rate_lost)


def test_default_config_margins_are_the_ones_used_in_these_tests() -> None:
    assert load_config().missing_data == replace(CFG, assumed_missing_rates=(0.01, 0.05, 0.10))


def median_policy() -> MissingDataPolicy:
    frame = synthetic({("median", ("Type",)): GOOD, ("median", (AIR,)): GOOD | {"saving": 0.5}})
    verdicts = pattern_verdicts(frame, CFG, n_cv_folds=5)
    return build_policy("forest", "median", verdicts, cost_ratio=10.0)


def test_the_policy_lists_only_scorable_patterns_with_their_degradation() -> None:
    policy = median_policy()
    assert [d.pattern for d in policy.scorable] == [("Type",)]
    assert policy.degradation(["Type"]) is not None
    assert policy.degradation(["Type"]).retained_saving == pytest.approx(0.975)  # type: ignore[union-attr]
    assert policy.is_scorable([]) and policy.is_scorable(["Type"])
    assert not policy.is_scorable([AIR]) and not policy.is_scorable(["Type", AIR])


def test_the_policy_survives_a_json_round_trip() -> None:
    policy = median_policy()
    assert MissingDataPolicy.from_json(policy.to_json()) == policy


def test_reject_policy_scores_only_complete_readings() -> None:
    frame = synthetic({("median", ("Type",)): GOOD})
    policy = build_policy(
        "forest", "reject", pattern_verdicts(frame, CFG, n_cv_folds=5), cost_ratio=10.0
    )
    assert policy.scorable == () and policy.is_scorable([]) and not policy.is_scorable(["Type"])
    assert policy.measured == () and policy.recoverable(["Type"]) is None


def reject_with_reference() -> MissingDataPolicy:
    frame = synthetic({("median", ("Type",)): GOOD, ("median", (AIR,)): GOOD | {"saving": 0.5}})
    verdicts = pattern_verdicts(frame, CFG, n_cv_folds=5)
    return build_policy("forest", "reject", verdicts, cost_ratio=10.0, measured_strategy="median")


def test_a_rejected_pattern_is_recoverable_only_if_the_reference_strategy_scored_it() -> None:
    policy = reject_with_reference()
    assert policy.scorable == () and not policy.is_scorable(["Type"])  # reference never scores
    assert policy.recoverable(["Type"]) is not None
    assert policy.recoverable(["Type"]).retained_saving == pytest.approx(0.975)  # type: ignore[union-attr]
    assert policy.recoverable([AIR]) is None  # measured and failed: no imputation would fix it
    assert policy.recoverable([]) is None  # a complete Reading is not refused


def test_the_reference_survives_a_json_round_trip_and_old_sidecars_still_load() -> None:
    policy = reject_with_reference()
    assert MissingDataPolicy.from_json(policy.to_json()) == policy
    old = '{"model": "m", "strategy": "reject", "cost_ratio": 10.0, "scorable": []}'
    assert MissingDataPolicy.from_json(old).measured == ()


def test_policy_rejects_unknown_strategy_and_scorable_patterns_under_reject() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        MissingDataPolicy("forest", "magic", 10.0, ())
    degradation = PatternDegradation(("Type",), 1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="reject"):
        MissingDataPolicy("forest", "reject", 10.0, (degradation,))
    with pytest.raises(ValueError, match="imputing"):
        MissingDataPolicy("forest", "reject", 10.0, (), measured_strategy="reject")
    with pytest.raises(ValueError, match="measured_strategy"):
        MissingDataPolicy("forest", "reject", 10.0, (), None, (degradation,))


def test_pattern_scores_are_pooled_per_repeat_and_cover_every_reading_once() -> None:
    data = readings(120, 12)
    idx = np.arange(data.height)
    folds = [(idx[idx % 3 != k], idx[idx % 3 == k]) for k in range(3)] * 2
    candidate = build_candidates(1)["random_forest_unweighted"]
    pooled = pattern_repeat_scores(
        candidate,
        data,
        folds,
        n_splits=3,
        strategy="median",
        patterns=[TORQUE_PATTERN, ("Type",)],
        n_jobs=1,
    )
    assert set(pooled) == {TORQUE_PATTERN, ("Type",)}
    assert all(m.shape == (2, 120) and not np.isnan(m).any() for m in pooled.values())
    with pytest.raises(ValueError, match="whole number"):
        pattern_repeat_scores(
            candidate, data, folds[:4], n_splits=3, strategy="median", patterns=[("Type",)]
        )


def test_recoverable_cost_is_never_negative_and_zero_at_the_best_threshold() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([[0.1, 0.2, 0.3, 0.4, 0.8, 0.9]] * 2)
    best = recoverable_cost(y, scores, frozen=0.8, cost_ratio=10)
    assert best.recoverable == pytest.approx(0.0) and best.frozen_cost == 0.0
    worse = recoverable_cost(y, scores, frozen=0.95, cost_ratio=10)
    assert worse.recoverable > 0 and worse.repicked_cost == 0.0
    assert worse.frozen_cost == pytest.approx(20 / 6)
