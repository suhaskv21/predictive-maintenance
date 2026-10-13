import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from predictive_maintenance.config import load_config
    from predictive_maintenance.data import schema
    from predictive_maintenance.data.ingest import load_dataset
    from predictive_maintenance.evaluation.compare import cv_folds
    from predictive_maintenance.evaluation.missingness import (
        SINGLE_SENSOR_PATTERNS,
        build_policy,
        choose_strategy,
        evaluate_patterns,
        incomplete_share,
        informative_loss,
        pattern_name,
        pattern_repeat_scores,
        pattern_verdicts,
        recoverable_cost,
        refused_share,
        scorable_patterns,
    )
    from predictive_maintenance.evaluation.split import (
        repeated_stratified_cv,
        stratified_holdout,
    )
    from predictive_maintenance.evaluation.threshold import estimate_threshold, repeat_scores
    from predictive_maintenance.preprocessing.imputation import IMPUTING_STRATEGIES
    from predictive_maintenance.training.candidates import (
        SWITCHABLE_FAMILIES,
        UNWEIGHTED,
        build_candidates,
    )

    return (
        IMPUTING_STRATEGIES,
        SINGLE_SENSOR_PATTERNS,
        SWITCHABLE_FAMILIES,
        UNWEIGHTED,
        alt,
        build_candidates,
        build_policy,
        choose_strategy,
        cv_folds,
        estimate_threshold,
        evaluate_patterns,
        incomplete_share,
        informative_loss,
        load_config,
        load_dataset,
        mo,
        np,
        pattern_name,
        pattern_repeat_scores,
        pattern_verdicts,
        pl,
        recoverable_cost,
        refused_share,
        repeat_scores,
        repeated_stratified_cv,
        schema,
        scorable_patterns,
        stratified_holdout,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 04 · Missing-data policy (concludes ADR 0008)

    What does the app do with an **Incomplete reading**: one where any of the six inputs (`Type`
    and the five sensors) is absent or unusable? The raw data has no nulls, so nothing is cleaned
    at training time; the question is what a model fitted on complete Readings can still say when
    an input is missing at scoring time. Same three models, same repeated stratified 5×5 CV on the
    **train partition only**, preprocessing fitted inside each fold, identical folds and identical
    injected absences for every model and strategy. **The holdout is discarded at load time.**

    **Rules, written before the numbers** (margins live in `configs/default.toml`,
    `[missing_data]`; logic in `predictive_maintenance.evaluation.missingness`):

    1. **Scenarios.** All 63 non-empty patterns of absent inputs, injected on 100 % of the
       validation rows only (training stays complete). Plus one *informative stress*: torque
       absent only on validation Readings at or above the training fold's 0.75 speed quantile,
       judged on those rows against the same rows complete; it can only veto `Torque`.
    2. **Strategies, simplest first.** `reject`, `median` (median/mode), `regression` (a ridge
       per absent set, fitted on complete training rows), `native` (NaN passed to the model, which
       routes it arbitrarily because it never saw one). Imputation runs before the features are
       derived and is fitted on the training fold only. No indicators (nothing to learn them from).
    3. **Frozen Operating threshold** (ADR 0007, 10:1) applied unchanged. Re-derived below and
       checked against the ADR.
    4. **Scorable pattern**, paired with the complete Readings on identical folds, all three:
       retained saving R minus its Nadeau–Bengio corrected SE ≥ 0.90 (R = saving over *flag
       nothing* with the pattern absent ÷ saving with complete data); mean recall drop ≤ 0.10
       absolute; Flag rate (alerts per 1,000) within ×0.5 to ×2 of the complete rate.
       PR-AUC, Brier and the like are reported, not judged.
    5. **Closure.** A pattern is scorable only if every pattern with one fewer absent input is.
       Anything not measured or not scorable is refused: no Failure risk, no Flag.
    6. **Strategy per model**, simplest-first walk: `reject` is the incumbent (R = 0). A challenger
       replaces it only if it makes ≥ 1 pattern scorable and its paired mean gain in R over the
       five single-sensor patterns exceeds `max(0.05, corrected SE)`. Each model keeps its own.
       The walk is a measurement; **the adopted policy is `reject` for all three models** (the
       measured gain is not reliable enough to build on), and the walk's winner is kept as the
       reference for the pipeline-health monitor.
    7. **Informational only:** the cost a pattern-specific threshold could recover (single inputs)
       and the data lost to incompleteness. Neither changes the threshold or the rules above.
    """)
    return


@app.cell
def _(
    SWITCHABLE_FAMILIES,
    UNWEIGHTED,
    build_candidates,
    cv_folds,
    load_config,
    load_dataset,
    mo,
    repeated_stratified_cv,
    schema,
    stratified_holdout,
):
    config = load_config()
    seed = config.seed
    cfg = config.missing_data
    readings = load_dataset()
    train, _holdout = stratified_holdout(readings, test_size=config.split.test_size, seed=seed)
    del _holdout
    n_splits = config.split.cv_folds
    folds = cv_folds(
        train,
        repeated_stratified_cv(n_splits=n_splits, n_repeats=config.split.cv_repeats, seed=seed),
    )
    candidates = build_candidates(seed)
    offered = [f"{family}_{UNWEIGHTED}" for family in SWITCHABLE_FAMILIES]
    y = train[schema.TARGET].to_numpy()
    ratio = config.threshold.default_cost_ratio
    mo.md(
        f"Seed `{seed}`. Train partition: **{train.height:,} Readings, {int(y.sum())} Machine "
        f"failures**, {train.null_count().sum_horizontal().item()} nulls; {len(folds)} folds "
        f"({n_splits}×{config.split.cv_repeats}). Default Cost ratio {ratio:g}:1. Margins: "
        f"R − SE ≥ {cfg.min_retained_saving}, recall drop ≤ {cfg.max_recall_drop}, Flag-rate ratio "
        f"{cfg.flag_rate_ratio_min}–{cfg.flag_rate_ratio_max}, challenger margin "
        f"{cfg.challenger_margin}, stress quantile {cfg.stress_speed_quantile}."
    )
    return cfg, folds, n_splits, offered, ratio, seed, train, y, candidates, config


@app.cell
def _(
    candidates,
    estimate_threshold,
    folds,
    mo,
    n_splits,
    offered,
    pl,
    ratio,
    repeat_scores,
    seed,
    train,
    y,
):
    # Re-derive each frozen threshold exactly as notebook 03b does, and check it against ADR 0007.
    ADR_0007 = {
        "random_forest_unweighted": 0.218,
        "lightgbm_unweighted": 0.134,
        "xgboost_unweighted": 0.075,
    }
    frozen = {}
    for _name in offered:
        _matrix = repeat_scores(candidates[_name], train, folds, n_splits=n_splits)
        # The median over repeats does not depend on the bootstrap; two resamples are enough here.
        frozen[_name] = estimate_threshold(y, _matrix, ratio, n_resamples=2, seed=seed).median
    drift = {n: round(frozen[n], 3) - ADR_0007[n] for n in offered}
    mo.stop(
        any(abs(d) > 5e-4 for d in drift.values()),
        mo.md(f"**Frozen thresholds differ from ADR 0007: {frozen}. Stopped.**"),
    )
    mo.vstack(
        [
            mo.md("### Frozen Operating thresholds (re-derived, match ADR 0007)"),
            pl.DataFrame(
                {
                    "model": offered,
                    "threshold": [frozen[n] for n in offered],
                    "adr_0007": [ADR_0007[n] for n in offered],
                }
            ),
        ]
    )
    return (frozen,)


@app.cell
def _(
    candidates,
    cfg,
    choose_strategy,
    evaluate_patterns,
    folds,
    frozen,
    n_splits,
    offered,
    pattern_verdicts,
    ratio,
    train,
):
    # The expensive step: 63 patterns × 3 imputing strategies × 25 folds, per model.
    results, verdicts, chosen, walks = {}, {}, {}, {}
    for _name in offered:
        results[_name] = evaluate_patterns(
            candidates[_name],
            train,
            folds,
            threshold=frozen[_name],
            cost_ratio=ratio,
            stress_quantile=cfg.stress_speed_quantile,
        )
        verdicts[_name] = pattern_verdicts(results[_name], cfg, n_cv_folds=n_splits)
        chosen[_name], walks[_name] = choose_strategy(
            results[_name], verdicts[_name], cfg, n_cv_folds=n_splits
        )
    return chosen, results, verdicts, walks


@app.cell
def _(chosen, mo, offered, pl, walks):
    walk_table = pl.DataFrame(
        [
            (
                _name,
                _s.incumbent,
                _s.challenger,
                _s.mean_delta,
                _s.corrected_se,
                _s.threshold,
                _s.replaced,
            )
            for _name in offered
            for _s in walks[_name]
        ],
        schema=[
            "model",
            "incumbent",
            "challenger",
            "mean_gain_R",
            "corrected_se",
            "needed",
            "replaced",
        ],
        orient="row",
    )
    mo.vstack(
        [
            mo.md(
                "### Strategy walk (simplest first; gain = paired per-fold gain in retained saving R, "
                "mean over the 5 single-sensor patterns)\n"
                f"**Chosen: {chosen}**"
            ),
            walk_table,
        ]
    )
    return walk_table


@app.cell
def _(SINGLE_SENSOR_PATTERNS, alt, mo, offered, pattern_name, pl, verdicts):
    single_names = [pattern_name(p) for p in SINGLE_SENSOR_PATTERNS]
    single = pl.concat(
        [verdicts[_name].with_columns(pl.lit(_name).alias("model")) for _name in offered]
    ).filter(pl.col("pattern").is_in(single_names))
    single_table = single.select(
        "model",
        "strategy",
        "pattern",
        "retained_saving",
        "retained_se",
        "recall_drop",
        "flag_rate_ratio",
        "pr_auc_relative_drop",
        "passes_saving",
        "passes_recall",
        "passes_flag",
        "stress_ok",
        "scorable",
    ).sort("model", "strategy", "pattern")
    heat = (
        alt.Chart(single.to_pandas())
        .mark_rect()
        .encode(
            x=alt.X("pattern:N", title=None),
            y=alt.Y("strategy:N", title=None),
            color=alt.Color(
                "retained_saving:Q", scale=alt.Scale(domain=[0, 1], scheme="redyellowgreen")
            ),
            tooltip=["model", "strategy", "pattern", "retained_saving", "recall_drop"],
        )
        .properties(width=260, height=90)
        .facet(row="model:N")
    )
    mo.vstack(
        [
            mo.md(
                "### One input absent: retained saving R (pass needs R − SE ≥ 0.90), recall drop, "
                "Flag-rate ratio. `pr_auc_relative_drop` is reported only (the old plan's metric)."
            ),
            heat,
            single_table,
        ]
    )
    return single, single_table


@app.cell
def _(mo, offered, pl, verdicts):
    counts = (
        pl.concat([verdicts[_name].with_columns(pl.lit(_name).alias("model")) for _name in offered])
        .group_by("model", "strategy")
        .agg(
            pl.col("passes").sum().alias("patterns_passing_alone"),
            pl.col("scorable").sum().alias("patterns_scorable"),
            pl.col("scorable").filter(pl.col("n_absent") == 1).sum().alias("scorable_1_absent"),
            pl.col("scorable").filter(pl.col("n_absent") == 2).sum().alias("scorable_2_absent"),
            pl.col("retained_saving")
            .filter(pl.col("n_absent") == 2)
            .mean()
            .alias("mean_R_2_absent"),
        )
        .sort("model", "strategy")
    )
    mo.vstack(
        [
            mo.md(
                "### How many of the 63 patterns are scorable (after closure and the stress veto)?"
            ),
            counts,
        ]
    )
    return counts


@app.cell
def _(mo, offered, pl, results):
    stress = (
        pl.concat(
            [
                results[_name]
                .with_columns(pl.lit(_name).alias("model"))
                .filter((pl.col("scope") == "stress"))
                for _name in offered
            ]
        )
        .group_by("model", "strategy")
        .agg(
            pl.col("recall").mean(),
            pl.col("alerts_per_1000").mean(),
            pl.col("cost_per_1000").mean(),
            pl.col("saving").mean(),
            pl.col("failures").sum(),
        )
        .sort("model", "strategy")
    )
    mo.vstack(
        [
            mo.md(
                "### Informative stress: torque absent only at the top speed quartile\n"
                "Rows are the high-speed validation Readings only; `complete` is the same rows with "
                "every input present."
            ),
            stress,
        ]
    )
    return stress


@app.cell
def _(build_policy, chosen, mo, offered, pl, ratio, verdicts):
    # The walk's winner is a measurement. The adopted policy is a decision (ADR 0008): reject for
    # every model, because the gain is not reliable enough. The winner stays as the reference the
    # pipeline-health monitor uses to size what imputation would recover.
    ADOPTED = "reject"
    policies = {
        _name: build_policy(
            _name, ADOPTED, verdicts[_name], cost_ratio=ratio, measured_strategy=chosen[_name]
        )
        for _name in offered
    }
    allow = pl.DataFrame(
        [
            (
                _name,
                policies[_name].strategy,
                len(policies[_name].scorable),
                policies[_name].measured_strategy,
                [" + ".join(d.pattern) for d in policies[_name].measured],
            )
            for _name in offered
        ],
        schema=["model", "adopted", "n_scorable", "reference_strategy", "reference_patterns"],
        orient="row",
    )
    mo.vstack(
        [
            mo.md(
                "### Adopted policy per model (the `<model>.missing.json` content; nothing is written here)"
            ),
            allow,
            mo.md("```json\n" + policies[offered[0]].to_json() + "\n```"),
        ]
    )
    return allow, policies


@app.cell
def _(
    SINGLE_SENSOR_PATTERNS,
    candidates,
    folds,
    frozen,
    mo,
    n_splits,
    offered,
    pattern_repeat_scores,
    pattern_name,
    pl,
    ratio,
    recoverable_cost,
    train,
    y,
):
    # Informational: would a threshold re-picked on injected scores recover cost? Never shipped.
    single_and_type = [*SINGLE_SENSOR_PATTERNS, ("Type",)]
    recover_rows = []
    for _name in offered:
        _pooled = pattern_repeat_scores(
            candidates[_name],
            train,
            folds,
            n_splits=n_splits,
            strategy="median",
            patterns=single_and_type,
        )
        for _pattern, _matrix in _pooled.items():
            _r = recoverable_cost(y, _matrix, frozen=frozen[_name], cost_ratio=ratio)
            recover_rows.append(
                (
                    _name,
                    pattern_name(_pattern),
                    frozen[_name],
                    _r.repicked_threshold,
                    1000 * _r.frozen_cost,
                    1000 * _r.repicked_cost,
                    1000 * _r.recoverable,
                )
            )
    recoverable = pl.DataFrame(
        recover_rows,
        schema=[
            "model",
            "pattern",
            "frozen_threshold",
            "repicked_threshold",
            "cost_frozen_per_1000",
            "cost_repicked_per_1000",
            "recoverable_per_1000",
        ],
        orient="row",
    )
    mo.vstack(
        [
            mo.md(
                "### Informational: cost a pattern-specific threshold could recover (median imputation)\n"
                "Re-picked and scored on the same scores, so an optimistic upper bound. Costs per 1,000 "
                "Readings in false-alarm units. Not shipped; the threshold stays frozen."
            ),
            recoverable,
        ]
    )
    return recoverable


@app.cell
def _(
    IMPUTING_STRATEGIES,
    cfg,
    chosen,
    incomplete_share,
    informative_loss,
    mo,
    offered,
    pl,
    refused_share,
    scorable_patterns,
    train,
    verdicts,
):
    # Data-loss check: what does "refuse an Incomplete reading" cost, and what would the best
    # challenger win back? MCAR with independent rates is an assumption (there are no nulls to measure).
    loss_rows = []
    for _name in offered:
        _counts = {s: len(scorable_patterns(verdicts[_name], s)) for s in IMPUTING_STRATEGIES}
        _best = next(s for s in IMPUTING_STRATEGIES if _counts[s] == max(_counts.values()))
        for _rate in cfg.assumed_missing_rates:
            loss_rows.append(
                (
                    _name,
                    _rate,
                    incomplete_share(_rate),
                    refused_share(_rate, scorable_patterns(verdicts[_name], chosen[_name])),
                    _best,
                    refused_share(_rate, scorable_patterns(verdicts[_name], _best)),
                )
            )
    data_loss = pl.DataFrame(
        loss_rows,
        schema=[
            "model",
            "rate_per_input",
            "incomplete_share",
            "refused_chosen_policy",
            "best_challenger",
            "refused_best_challenger",
        ],
        orient="row",
    )
    top = informative_loss(train, cfg.stress_speed_quantile)
    mo.vstack(
        [
            mo.md(
                "### Data loss (per-input absence rate assumed, independent)\n"
                "`incomplete_share` is also the share of a complete-case retraining set that would be "
                "dropped. `refused_*` is the share of incoming Readings the app would refuse.\n\n"
                f"**Informative-missingness check.** If torque drops out at the top-speed quartile, "
                f"that group is {top.rows_lost:.1%} of Readings but {top.failures_lost:.1%} of the failures "
                f"(failure rate {top.failure_rate_lost:.1%} vs {top.failure_rate_overall:.1%} overall)."
            ),
            data_loss,
        ]
    )
    return data_loss, top


if __name__ == "__main__":
    app.run()
