import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.metrics import average_precision_score

    from predictive_maintenance.config import load_config
    from predictive_maintenance.data import schema
    from predictive_maintenance.data.ingest import load_dataset
    from predictive_maintenance.evaluation.compare import (
        MARGIN,
        calibration_decision,
        compare_candidates,
        cv_folds,
        derived_seeds,
        measure_operational_cost,
        out_of_fold_scores,
        paired_deltas,
        reliability_table,
        score_readings,
        select_model,
        select_variants,
        train_size_penalty,
    )
    from predictive_maintenance.evaluation.split import (
        repeated_stratified_cv,
        stratified_holdout,
    )
    from predictive_maintenance.training.candidates import (
        SELECTION_ORDER,
        SWITCHABLE_FAMILIES,
        TREE_FAMILIES,
        build_candidates,
    )

    return (
        MARGIN,
        SELECTION_ORDER,
        SWITCHABLE_FAMILIES,
        TREE_FAMILIES,
        alt,
        average_precision_score,
        build_candidates,
        calibration_decision,
        compare_candidates,
        cv_folds,
        derived_seeds,
        load_config,
        load_dataset,
        measure_operational_cost,
        mo,
        np,
        out_of_fold_scores,
        paired_deltas,
        pl,
        reliability_table,
        repeated_stratified_cv,
        schema,
        score_readings,
        select_model,
        select_variants,
        stratified_holdout,
        train_size_penalty,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 03 · Model comparison (concludes ADR 0006)

    Which model family, and which models does the app offer? Thirteen candidates, scored on the
    **train partition only** with the repeated stratified 5×5 CV of ADR 0005 (preprocessing fitted
    inside every fold, identical folds for all candidates, so differences are paired). **The holdout
    is discarded at load time and never scored.**

    **Rules, written before the numbers** (they live in `predictive_maintenance.evaluation.compare`):

    1. Metric: mean PR-AUC over the 25 folds.
    2. Every supervised family (logistic, random forest, `HistGradientBoostingClassifier`, LightGBM,
       XGBoost) has an unweighted and a class-weighted (`balanced`) variant. Selectable candidates
       are walked simplest first (families in that order, unweighted before weighted). A more complex
       challenger replaces the incumbent only if its mean paired delta beats
       `max(0.02, corrected SE)`, with the Nadeau–Bengio SE (overlapping folds). The walk's winner is
       the **default model**.
    3. **Switchable models** (amended after the first results, see the ADR): random forest, LightGBM
       and XGBoost are offered to the app's model selector, each in its own variant chosen by the
       same rule (weighted replaces unweighted only beyond the margin). HistGradientBoosting stays
       in the comparison but is not offered (redundant with LightGBM; post-review scope change).
    4. Vetoes: single-Reading latency ≤ 100 ms, serialized pipeline ≤ 50 MB.
    5. Dummy is a baseline; Isolation Forest and One-Class SVM (fitted on non-failures only) are
       reference candidates and are not selectable (no probability).
    6. Calibrate a model if sigmoid or isotonic cuts its mean CV Brier by ≥ 10 % relative; isotonic
       only if it also beats sigmoid by 10 %. Applied to every tree family, because each shows its
       own probability.
    7. Revisit the 70/30 split if the default model's learning-curve penalty (training on 87.5 % of
       the rows, the 80/20 → 70/30 step) exceeds 0.01 PR-AUC; reported for every tree family.

    Disclosure: ADR 0004 had already shown logistic ≈ 0.50 and boosting ≈ 0.885 on this feature set,
    and the unweighted results of the first run were seen before rules 3 and the weighted tree
    variants were added. Defaults only, no tuning.
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
    mo.md(
        f"Seed `{seed}`. Train partition: **{train.height:,} Readings, "
        f"{int(train[schema.TARGET].sum())} Machine failures**; {len(folds)} folds "
        f"({config.split.cv_folds}×{config.split.cv_repeats})."
    )
    return candidates, config, folds, seed, train


@app.cell
def _(candidates, measure_operational_cost, pl, train):
    costs = {name: measure_operational_cost(c, train) for name, c in candidates.items()}
    cost_table = pl.DataFrame(
        [
            (n, c.fit_seconds, c.latency_ms, c.artifact_mb, c.passes_vetoes())
            for n, c in costs.items()
        ],
        schema=["candidate", "fit_seconds", "latency_ms", "artifact_mb", "passes_vetoes"],
        orient="row",
    )
    cost_table
    return cost_table, costs


@app.cell
def _(candidates, compare_candidates, folds, train):
    scores = compare_candidates(list(candidates.values()), train, folds)
    return (scores,)


@app.cell
def _(candidates, mo, paired_deltas, pl, scores):
    summary = (
        scores.group_by("candidate")
        .agg(
            pr_auc=pl.col("pr_auc").mean(),
            pr_auc_sd=pl.col("pr_auc").std(),
            brier=pl.col("brier").mean(),
            fit_seconds=pl.col("fit_seconds").mean(),
        )
        .with_columns(
            role=pl.col("candidate").replace_strict({n: c.role for n, c in candidates.items()}),
            delta_vs_dummy=pl.col("candidate").map_elements(
                lambda n: float(paired_deltas(scores, n, "dummy").mean()), return_dtype=pl.Float64
            ),
        )
        .sort("pr_auc", descending=True)
    )
    mo.vstack([mo.md("### Repeated-CV scores (25 folds, train partition)"), summary])
    return (summary,)


@app.cell
def _(alt, mo, scores):
    chart = (
        alt.Chart(scores.to_pandas())
        .mark_boxplot()
        .encode(
            x=alt.X("pr_auc:Q", title="PR-AUC per fold"),
            y=alt.Y("candidate:N", sort="-x"),
        )
        .properties(width=520, height=360)
    )
    mo.ui.altair_chart(chart)
    return


@app.cell
def _(
    MARGIN,
    SELECTION_ORDER,
    SWITCHABLE_FAMILIES,
    TREE_FAMILIES,
    config,
    cost_table,
    mo,
    paired_deltas,
    pl,
    scores,
    select_model,
    select_variants,
):
    passing = set(cost_table.filter(pl.col("passes_vetoes"))["candidate"])
    order = [n for n in SELECTION_ORDER if n in passing]
    selection = select_model(scores, order, n_cv_folds=config.split.cv_folds, margin=MARGIN)
    steps = pl.DataFrame(
        [
            (s.challenger, s.incumbent, s.mean_delta, s.corrected_se, s.threshold, s.replaced)
            for s in selection.steps
        ],
        schema=[
            "challenger",
            "incumbent",
            "mean_delta",
            "corrected_se",
            "threshold",
            "replaced",
        ],
        orient="row",
    )
    variant_choice = select_variants(
        scores, TREE_FAMILIES, n_cv_folds=config.split.cv_folds, margin=MARGIN
    )
    supported = {family: choice.winner for family, choice in variant_choice.items()}
    mean_pr_auc = scores.group_by("candidate").agg(pr_auc=pl.col("pr_auc").mean())
    supported_table = pl.DataFrame(
        [
            (
                family,
                model,
                float(mean_pr_auc.filter(pl.col("candidate") == model)["pr_auc"][0]),
                float(paired_deltas(scores, model, selection.winner).mean()),
                model in passing,
                family in SWITCHABLE_FAMILIES,
            )
            for family, model in supported.items()
        ],
        schema=["family", "model", "pr_auc", "delta_vs_default", "passes_vetoes", "offered"],
        orient="row",
    )
    weighting_steps = pl.DataFrame(
        [
            (
                family,
                s.challenger,
                s.incumbent,
                s.mean_delta,
                s.corrected_se,
                s.threshold,
                s.replaced,
            )
            for family, choice in variant_choice.items()
            for s in choice.steps
        ],
        schema=[
            "family",
            "challenger",
            "incumbent",
            "mean_delta",
            "corrected_se",
            "threshold",
            "replaced",
        ],
        orient="row",
    )
    mo.vstack(
        [
            mo.md(f"### Selection walk: default model **{selection.winner}**"),
            steps,
            mo.md("### Class weighting per tree family (weighted must beat unweighted)"),
            weighting_steps,
            mo.md(
                "### Tree families: each family's chosen variant "
                "(offered = switchable in the app; HistGradientBoosting is compared, not offered)"
            ),
            supported_table,
        ]
    )
    return selection, supported


@app.cell
def _(mo, paired_deltas, pl, scores, selection):
    reference = pl.DataFrame(
        [
            (n, float(paired_deltas(scores, n, selection.winner).mean()))
            for n in ("isolation_forest", "one_class_svm")
        ],
        schema=["reference_candidate", "mean_delta_vs_default"],
        orient="row",
    )
    mo.vstack([mo.md("### Anomaly-detection reference candidates vs the default"), reference])
    return


@app.cell
def _(
    build_candidates,
    calibration_decision,
    compare_candidates,
    folds,
    mo,
    out_of_fold_scores,
    pl,
    reliability_table,
    schema,
    scores,
    seed,
    supported,
    train,
):
    # Calibration check for every tree family: raw vs sigmoid vs isotonic, paired Brier.
    calibration_candidates = build_candidates(seed)
    y_train = train[schema.TARGET].to_numpy()
    decisions, brier_rows, reliability_rows = {}, [], []
    for model in supported.values():
        candidate = calibration_candidates[model]
        extra = compare_candidates([candidate], train, folds, variants=("sigmoid", "isotonic"))
        variants = pl.concat([scores.filter(pl.col("candidate") == model), extra])
        decisions[model] = calibration_decision(variants, model)
        brier_rows.append(
            variants.group_by("variant")
            .agg(brier=pl.col("brier").mean(), pr_auc=pl.col("pr_auc").mean())
            .with_columns(model=pl.lit(model))
        )
        chosen = [None] + ([decisions[model].method] if decisions[model].method else [])
        for method in chosen:
            oof = out_of_fold_scores(candidate, train, folds, variant=method)
            reliability_rows.append(
                reliability_table(y_train, oof).with_columns(
                    model=pl.lit(model), variant=pl.lit(method or "raw")
                )
            )
    decision_table = pl.DataFrame(
        [
            (
                m,
                d.method or "none",
                d.relative_brier_gain["sigmoid"],
                d.relative_brier_gain["isotonic"],
            )
            for m, d in decisions.items()
        ],
        schema=["model", "calibration", "sigmoid_brier_gain", "isotonic_brier_gain"],
        orient="row",
    )
    mo.vstack(
        [
            mo.md("### Calibration decision per tree family (rule: ≥ 10 % relative Brier gain)"),
            decision_table,
            mo.md("### Mean CV Brier and PR-AUC by calibration variant"),
            pl.concat(brier_rows)
            .select("model", "variant", "brier", "pr_auc")
            .sort("model", "variant"),
            mo.md("### Reliability (10 equal-count bins, out-of-fold, averaged over repeats)"),
            pl.concat(reliability_rows).select(
                "model", "variant", "bin", "n", "mean_predicted", "observed_rate"
            ),
        ]
    )
    return


@app.cell
def _(
    build_candidates,
    derived_seeds,
    mo,
    pl,
    seed,
    selection,
    supported,
    train,
    train_size_penalty,
):
    curve_candidates = build_candidates(seed)
    curve_seeds = derived_seeds(seed, 1, 60)
    probes = list(dict.fromkeys(["logistic_unweighted", selection.winner, *supported.values()]))
    curve = pl.concat(
        [
            train_size_penalty(curve_candidates[n], train, curve_seeds).with_columns(
                candidate=pl.lit(n)
            )
            for n in probes
        ]
    )
    curve_summary = curve.group_by("candidate", maintain_order=True).agg(
        pr_auc_full=pl.col("pr_auc_full").mean(),
        pr_auc_reduced=pl.col("pr_auc_reduced").mean(),
        mean_penalty=pl.col("penalty").mean(),
        penalty_sd=pl.col("penalty").std(),
    )
    mo.vstack(
        [
            mo.md(
                "### Learning curve: PR-AUC with 100 % vs 87.5 % of the training rows "
                "(60 throwaway splits inside the train partition; revisit 70/30 if the default "
                "model's penalty > 0.01)"
            ),
            curve_summary,
        ]
    )
    return


@app.cell
def _(
    average_precision_score,
    candidates,
    folds,
    mo,
    np,
    pl,
    schema,
    score_readings,
    seed,
    selection,
    train,
):
    # Permutation importance on the first fold's test part: drop in PR-AUC when one input is shuffled.
    train_idx, test_idx = folds[0]
    x_all = train.select(schema.FEATURE_COLUMNS)
    y_all = train[schema.TARGET].to_numpy()
    fitted = candidates[selection.winner].make_pipeline().fit(x_all[train_idx], y_all[train_idx])
    x_test, y_test = x_all[test_idx], y_all[test_idx]
    baseline = average_precision_score(y_test, score_readings(fitted, x_test))
    rng = np.random.default_rng(seed)
    importance = pl.DataFrame(
        [
            (
                column,
                float(
                    np.mean(
                        [
                            baseline
                            - average_precision_score(
                                y_test,
                                score_readings(
                                    fitted,
                                    x_test.with_columns(
                                        pl.Series(
                                            column, rng.permutation(x_test[column].to_numpy())
                                        )
                                    ),
                                ),
                            )
                            for _ in range(10)
                        ]
                    )
                ),
            )
            for column in schema.FEATURE_COLUMNS
        ],
        schema=["input", "pr_auc_drop"],
        orient="row",
    ).sort("pr_auc_drop", descending=True)
    mo.vstack([mo.md(f"### Permutation importance, {selection.winner} (fold 0)"), importance])
    return


if __name__ == "__main__":
    app.run()
