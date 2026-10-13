import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from dataclasses import replace

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.metrics import precision_recall_curve

    from predictive_maintenance.config import load_config
    from predictive_maintenance.data import schema
    from predictive_maintenance.data.ingest import load_dataset
    from predictive_maintenance.evaluation.compare import (
        MARGIN,
        compare_candidates,
        corrected_se,
        cv_folds,
        derived_seeds,
        feature_check_verdict,
        paired_deltas,
    )
    from predictive_maintenance.evaluation.split import (
        repeated_stratified_cv,
        stratified_holdout,
    )
    from predictive_maintenance.evaluation.threshold import (
        BOOTSTRAP_STREAM,
        THRESHOLD_STREAM,
        build_operating_point,
        compare_costs,
        cv_operating_metrics,
        estimate_threshold,
        fold_expected_costs,
        interval_cost_excess,
        is_flat_optimum,
        is_viable,
        mean_cost,
        ranking_metrics,
        repeat_scores,
        risk_floor,
    )
    from predictive_maintenance.training.candidates import (
        SWITCHABLE_FAMILIES,
        UNWEIGHTED,
        build_candidates,
    )

    return (
        BOOTSTRAP_STREAM,
        MARGIN,
        SWITCHABLE_FAMILIES,
        THRESHOLD_STREAM,
        UNWEIGHTED,
        alt,
        build_candidates,
        build_operating_point,
        compare_candidates,
        compare_costs,
        corrected_se,
        cv_folds,
        cv_operating_metrics,
        derived_seeds,
        estimate_threshold,
        feature_check_verdict,
        fold_expected_costs,
        interval_cost_excess,
        is_flat_optimum,
        is_viable,
        load_config,
        load_dataset,
        mean_cost,
        mo,
        np,
        paired_deltas,
        pl,
        precision_recall_curve,
        ranking_metrics,
        repeat_scores,
        repeated_stratified_cv,
        replace,
        risk_floor,
        schema,
        stratified_holdout,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 03b · Metric and threshold (concludes ADR 0007)

    Where does a **Reading** become a **Flag**, and what does the engineer see? The three offered
    models (random forest, LightGBM, XGBoost; all unweighted, ADR 0006) each get their own
    **Operating threshold**, chosen on the **train partition only** with the repeated stratified 5×5
    CV of ADR 0005 (preprocessing fitted inside every fold, identical folds for every model).
    **The holdout is discarded at load time and never scored.**

    **Cost ratio.** An *assumption*, not a measurement: how many **False alarms** (one unnecessary
    inspection) one **Missed failure** (unplanned downtime) is worth. 5:1, 10:1 and 20:1 are shown;
    10:1 sets the Flag (`configs/default.toml`).

    **Rules, written before the numbers** (constants live in
    `predictive_maintenance.evaluation.threshold` and `.compare`):

    1. **Feature re-check first.** All three engineered features (ADR 0004) minus raw features, mean
       paired PR-AUC delta on identical folds, per offered model. Agrees at ≥ −0.005, disagrees at
       ≤ −0.02, noise in between. A disagreement stops the notebook: ADR 0004 is not touched here.
    2. **Threshold per CV repeat.** Each repeat pools its five test folds into one out-of-fold score
       per Reading. The threshold minimises expected cost per Reading
       (`(ratio × missed failures + false alarms) / Readings`). Candidates are the distinct observed
       scores (ties are never split); among exactly-tied minima the widest interval wins and its
       midpoint is used. The shipped threshold is the **median over repeats**, reported with the
       repeat range and a 90 % bootstrap interval (failures and non-failures resampled separately).
    3. **Viability.** A model's expected cost must beat *flag nothing* by ≥ 25 % at every ratio, or it
       is not offered.
    4. **Bands.** Low < t(10:1) ≤ Elevated < t(5:1) ≤ High. They ship only if Elevated and High each
       hold ≥ 10 failures (mean per repeat) and the observed failure rate strictly rises across the
       bands. If the two edges are not strictly ordered the build fails.
    5. **Risk floor.** The 95 % Wilson upper bound on the failure rate of the lowest-scoring half of
       Readings, rounded up to a whole percent, at least 1 %; the maximum over the three models.
    6. **Model cost comparison (informational).** A booster is *cheaper* than the forest if its paired
       per-fold expected-cost saving exceeds `max(10 % of the forest's cost, corrected SE)`. It never
       changes the default.
    7. **Flat optimum (informational).** The cost at both ends of the bootstrap interval within 10 %
       of the cost at the median threshold means the optimum is flat.
    8. **Primary metric at the operating point:** expected cost at the default ratio. Headline for
       engineers: recall, precision, alerts per 1,000 Readings. PR-AUC (ranking), ROC-AUC
       (optimistic at 3.4 % prevalence) and Brier (probability quality) are supporting.
    """)
    return


@app.cell
def _(
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
    readings = load_dataset()
    train, _holdout = stratified_holdout(readings, test_size=config.split.test_size, seed=seed)
    del _holdout
    folds = cv_folds(
        train,
        repeated_stratified_cv(
            n_splits=config.split.cv_folds, n_repeats=config.split.cv_repeats, seed=seed
        ),
    )
    candidates = build_candidates(seed)
    y = train[schema.TARGET].to_numpy()
    ratios = config.threshold.cost_ratios
    default_ratio = config.threshold.default_cost_ratio
    mo.md(
        f"Seed `{seed}`. Train partition: **{train.height:,} Readings, {int(y.sum())} Machine "
        f"failures**; {len(folds)} folds ({config.split.cv_folds}×{config.split.cv_repeats}). Cost "
        f"ratios {ratios}, default {default_ratio:g}:1; {config.evaluation.bootstrap_resamples} "
        "bootstrap resamples."
    )
    return candidates, config, default_ratio, folds, ratios, seed, train, y


@app.cell
def _(
    MARGIN,
    SWITCHABLE_FAMILIES,
    UNWEIGHTED,
    candidates,
    compare_candidates,
    corrected_se,
    feature_check_verdict,
    folds,
    mo,
    paired_deltas,
    pl,
    replace,
    config,
    train,
):
    # Re-check ADR 0004's engineered features for the offered models (all-three minus raw).
    offered = [f"{family}_{UNWEIGHTED}" for family in SWITCHABLE_FAMILIES]
    raw_names = {name: f"{name}__raw" for name in offered}
    pairs = [
        c
        for name in offered
        for c in (candidates[name], replace(candidates[name], name=raw_names[name], engineered=()))
    ]
    feature_scores = compare_candidates(pairs, train, folds)
    feature_rows = []
    for _name in offered:
        deltas = paired_deltas(feature_scores, _name, raw_names[_name])
        feature_rows.append(
            (
                _name,
                float(feature_scores.filter(pl.col("candidate") == raw_names[_name])["pr_auc"].mean()),
                float(feature_scores.filter(pl.col("candidate") == _name)["pr_auc"].mean()),
                float(deltas.mean()),
                corrected_se(deltas, n_cv_folds=config.split.cv_folds),
                feature_check_verdict(float(deltas.mean())),
            )
        )
    feature_table = pl.DataFrame(
        feature_rows,
        schema=["model", "pr_auc_raw", "pr_auc_all_three", "mean_delta", "corrected_se", "verdict"],
        orient="row",
    )
    verdicts = set(feature_table["verdict"])
    overall = (
        "disagrees"
        if "disagrees" in verdicts
        else "agrees"
        if verdicts == {"agrees"}
        else "noise"
    )
    mo.vstack(
        [
            mo.md(
                "### Feature re-check: all three engineered features minus raw (PR-AUC, 25 folds)\n"
                f"Margin for *disagrees*: −{MARGIN}. **Overall: {overall}.**"
            ),
            feature_table,
        ]
    )
    return offered, overall


@app.cell
def _(mo, overall):
    # A disagreement means: stop and ask before touching ADR 0004.
    mo.stop(
        overall == "disagrees",
        mo.md("**Feature re-check disagrees with ADR 0004. Stopped: ask before going on.**"),
    )
    return


@app.cell
def _(candidates, config, folds, offered, repeat_scores, train):
    # One row of pooled out-of-fold scores per CV repeat, per offered model.
    rows = {
        name: repeat_scores(candidates[name], train, folds, n_splits=config.split.cv_folds)
        for name in offered
    }
    return (rows,)


@app.cell
def _(mo, np, pl, ranking_metrics, rows, y):
    ranking = pl.DataFrame(
        [
            (
                name,
                *(
                    float(np.mean([ranking_metrics(y, r)[m] for r in matrix]))
                    for m in ("pr_auc", "roc_auc", "brier")
                ),
            )
            for name, matrix in rows.items()
        ],
        schema=["model", "pr_auc", "roc_auc", "brier"],
        orient="row",
    )
    mo.vstack(
        [
            mo.md("### Supporting metrics (pooled out-of-fold, mean over the 5 repeats)"),
            ranking,
        ]
    )
    return


@app.cell
def _(
    THRESHOLD_STREAM,
    config,
    derived_seeds,
    estimate_threshold,
    interval_cost_excess,
    is_flat_optimum,
    mo,
    pl,
    ratios,
    rows,
    seed,
    y,
):
    # Same bootstrap seed per ratio for every model, so the comparison is fair.
    ratio_seeds = dict(zip(ratios, derived_seeds(seed, THRESHOLD_STREAM, len(ratios)), strict=True))
    estimates = {
        (name, ratio): estimate_threshold(
            y, matrix, ratio, n_resamples=config.evaluation.bootstrap_resamples, seed=ratio_seeds[ratio]
        )
        for name, matrix in rows.items()
        for ratio in ratios
    }
    threshold_rows = []
    for (_name, _ratio), _est in estimates.items():
        _excess = interval_cost_excess(y, rows[_name], _est)
        threshold_rows.append(
            (
                _name,
                _ratio,
                _est.median,
                _est.repeat_range[0],
                _est.repeat_range[1],
                _est.interval[0],
                _est.interval[1],
                _excess[0],
                _excess[1],
                is_flat_optimum(_excess),
            )
        )
    threshold_table = pl.DataFrame(
        threshold_rows,
        schema=[
            "model",
            "cost_ratio",
            "threshold",
            "repeat_min",
            "repeat_max",
            "interval_low",
            "interval_high",
            "cost_excess_low",
            "cost_excess_high",
            "flat_optimum",
        ],
        orient="row",
    )
    mo.vstack(
        [
            mo.md(
                "### Operating thresholds (median over repeats; repeat range; 90 % bootstrap interval)"
            ),
            threshold_table,
        ]
    )
    return (estimates,)


@app.cell
def _(cv_operating_metrics, estimates, is_viable, mo, pl, ratios, rows, y):
    metric_rows, viable = [], {}
    for _name, _matrix in rows.items():
        _per_ratio = [
            cv_operating_metrics(y, _matrix, estimates[(_name, r)].median, r) for r in ratios
        ]
        viable[_name] = is_viable(_per_ratio)
        for _m in _per_ratio:
            metric_rows.append(
                (
                    _name,
                    _m.cost_ratio,
                    _m.threshold,
                    _m.recall,
                    _m.precision,
                    _m.alerts_per_1000,
                    _m.false_alarms_per_caught,
                    _m.cost_per_1000,
                    _m.flag_nothing_per_1000,
                    _m.flag_everything_per_1000,
                    _m.saving_vs_flag_nothing,
                )
            )
    metric_table = pl.DataFrame(
        metric_rows,
        schema=[
            "model",
            "cost_ratio",
            "threshold",
            "recall",
            "precision",
            "alerts_per_1000",
            "false_alarms_per_caught",
            "cost_per_1000",
            "flag_nothing_per_1000",
            "flag_everything_per_1000",
            "saving_vs_flag_nothing",
        ],
        orient="row",
    )
    mo.vstack(
        [
            mo.md(
                "### What each threshold means (CV, mean over repeats). Costs per 1,000 Readings "
                f"in false-alarm units. Viable (≥ 25 % saving at every ratio): {viable}"
            ),
            metric_table,
        ]
    )
    return (viable,)


@app.cell
def _(
    compare_costs,
    config,
    default_ratio,
    estimates,
    folds,
    fold_expected_costs,
    mo,
    pl,
    rows,
    y,
):
    # Informational: are the boosters cheaper than the forest, paired on identical folds?
    n_splits = config.split.cv_folds
    fold_costs = {
        name: fold_expected_costs(
            y,
            matrix,
            folds,
            n_splits=n_splits,
            threshold=estimates[(name, default_ratio)].median,
            cost_ratio=default_ratio,
        )
        for name, matrix in rows.items()
    }
    incumbent = "random_forest_unweighted"
    comparison_rows = []
    for _name, _costs in fold_costs.items():
        if _name == incumbent:
            continue
        _c = compare_costs(_costs, fold_costs[incumbent], n_cv_folds=n_splits)
        comparison_rows.append(
            (_name, 1000 * _c.mean_saving, _c.relative_saving, 1000 * _c.corrected_se, _c.cheaper)
        )
    mo.vstack(
        [
            mo.md(
                f"### Expected cost vs the forest at {default_ratio:g}:1 (positive saving = cheaper "
                "than the forest; informational, never changes the default)"
            ),
            pl.DataFrame(
                comparison_rows,
                schema=["model", "saving_per_1000", "relative_saving", "corrected_se_per_1000", "cheaper"],
                orient="row",
            ),
        ]
    )
    return


@app.cell
def _(
    build_operating_point,
    default_ratio,
    estimates,
    mo,
    pl,
    ratios,
    risk_floor,
    rows,
    y,
):
    floors = {name: max(risk_floor(y, matrix), 0.0) for name, matrix in rows.items()}
    shared_floor = max(floors.values())
    points, band_tables, band_notes = {}, [], {}
    for _name, _matrix in rows.items():
        try:
            _point, _table, _check = build_operating_point(
                _name,
                y,
                _matrix,
                thresholds={r: estimates[(_name, r)].median for r in ratios},
                default_cost_ratio=default_ratio,
                floor=shared_floor,
            )
        except ValueError as _error:
            band_notes[_name] = f"FAILED: {_error}"
            continue
        points[_name] = _point
        band_notes[_name] = "ships" if _check.ships else f"does not ship: {_check.reason}"
        band_tables.append(_table.with_columns(model=pl.lit(_name)))
    mo.vstack(
        [
            mo.md(
                f"### Risk floor per model: {floors}. **Shared floor: {shared_floor:.0%}.**\n\n"
                f"### Bands per model: {band_notes}"
            ),
            pl.concat(band_tables) if band_tables else mo.md("no bands"),
        ]
    )
    return (points,)


@app.cell
def _(mo, points):
    mo.vstack(
        [
            mo.md("### Operating point sidecars (JSON, written next to each Pipeline artifact)"),
            *[mo.md(f"`{name}`\n```json\n{p.to_json()}\n```") for name, p in points.items()],
        ]
    )
    return


@app.cell
def _(alt, cv_operating_metrics, default_ratio, estimates, mo, np, pl, precision_recall_curve, ratios, rows, y):
    # One curve per model from the out-of-fold scores of all 5 repeats pooled together.
    _labels = np.asarray(y)
    _curve_rows, _marker_rows = [], []
    for _name, _matrix in rows.items():
        _prec, _rec, _ = precision_recall_curve(np.tile(_labels, len(_matrix)), _matrix.ravel())
        _keep = np.linspace(0, len(_rec) - 1, min(len(_rec), 400)).astype(int)
        _curve_rows += [(_name, float(_rec[i]), float(_prec[i])) for i in _keep]
        for _r in ratios:
            _m = cv_operating_metrics(y, _matrix, estimates[(_name, _r)].median, _r)
            _marker_rows.append(
                (_name, f"{_r:g}:1", _r == default_ratio, _m.threshold, _m.recall, _m.precision)
            )
    pr_curve = pl.DataFrame(_curve_rows, schema=["model", "recall", "precision"], orient="row")
    pr_markers = pl.DataFrame(
        _marker_rows,
        schema=["model", "cost_ratio", "is_default", "threshold", "recall", "precision"],
        orient="row",
    )

    _x = alt.X("recall:Q", title="Recall (share of Machine failures Flagged)", scale=alt.Scale(domain=[0, 1]))
    _y = alt.Y("precision:Q", title="Precision (share of Flags that are real)", scale=alt.Scale(domain=[0, 1.02]))
    _lines = alt.Chart(pr_curve).mark_line().encode(x=_x, y=_y, color="model:N")
    _points = (
        alt.Chart(pr_markers)
        .mark_point(filled=True, stroke="black", strokeWidth=1)
        .encode(
            x=_x,
            y=_y,
            color="model:N",
            shape=alt.Shape("cost_ratio:N", title="Cost ratio"),
            size=alt.condition("datum.is_default", alt.value(260), alt.value(90)),
            tooltip=["model", "cost_ratio", "threshold", "recall", "precision"],
        )
    )
    _title = alt.TitleParams(
        f"Precision-recall curve with the chosen Operating thresholds (CV, {default_ratio:g}:1 is the default)",
        subtitle=[
            "How to read it: each line is one model. Moving right catches more Machine failures (recall) but",
            "more of the Flags are False alarms (precision falls). Higher and further right is better; the area",
            "under the line is PR-AUC. The marks are the chosen thresholds: the large one is the default ratio,",
            "the small ones are 5:1 and 20:1. A higher cost ratio moves the mark right (more Flags, fewer misses).",
        ],
        anchor="start",
        subtitleColor="gray",
    )
    mo.ui.altair_chart((_lines + _points).properties(title=_title, width=600, height=380))
    return pr_curve, pr_markers


@app.cell
def _(alt, default_ratio, estimates, mean_cost, mo, np, pl, rows, y):
    _grid = np.linspace(0.0, 0.8, 161)
    cost_curve = pl.DataFrame(
        [
            (name, float(t), 1000 * mean_cost(y, matrix, float(t), default_ratio))
            for name, matrix in rows.items()
            for t in _grid
        ],
        schema=["model", "threshold", "cost_per_1000"],
        orient="row",
    )
    chosen = pl.DataFrame(
        [
            (
                name,
                estimates[(name, default_ratio)].median,
                1000 * mean_cost(y, matrix, estimates[(name, default_ratio)].median, default_ratio),
            )
            for name, matrix in rows.items()
        ],
        schema=["model", "threshold", "cost_per_1000"],
        orient="row",
    )
    _x = alt.X("threshold:Q", title="Threshold (Failure risk at or above which a Reading is Flagged)")
    _y = alt.Y("cost_per_1000:Q", title="Expected cost per 1,000 Readings (False-alarm units)")
    _lines = alt.Chart(cost_curve).mark_line().encode(x=_x, y=_y, color="model:N")
    _rules = alt.Chart(chosen).mark_rule(strokeDash=[4, 4]).encode(x=_x, color="model:N")
    _points = (
        alt.Chart(chosen)
        .mark_point(filled=True, size=200, stroke="black", strokeWidth=1)
        .encode(x=_x, y=_y, color="model:N", tooltip=["model", "threshold", "cost_per_1000"])
    )
    _title = alt.TitleParams(
        f"Expected cost per 1,000 Readings at {default_ratio:g}:1 (CV), with the chosen thresholds",
        subtitle=[
            "How to read it: for each possible threshold, the cost of the Missed failures (each worth the cost",
            "ratio in False alarms) plus the False alarms it would cause. Lower is better. Far left flags",
            "almost everything (many False alarms); far right flags almost nothing (many Missed failures).",
            "The large mark and dashed line show each model's chosen threshold, at the bottom of its curve.",
            "A flat bottom means nearby thresholds cost about the same, so the exact value matters less.",
        ],
        anchor="start",
        subtitleColor="gray",
    )
    mo.ui.altair_chart((_lines + _rules + _points).properties(title=_title, width=600, height=380))
    return chosen, cost_curve


@app.cell
def _(mo):
    mo.md(r"""
    ## What this notebook did not do

    - The **holdout** was discarded at load time. The final holdout score, with bootstrap intervals
      (`bootstrap_metric_cis`), is produced once, later, for the single model finally chosen, with its
      threshold frozen beforehand.
    - Missing data (ADR 0008), app wording and layout (ADR 0009) and deployment (ADR 0010) are not
      decided here.
    """)
    return


if __name__ == "__main__":
    app.run()
