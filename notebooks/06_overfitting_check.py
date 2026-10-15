import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from dataclasses import replace
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from joblib import Parallel, delayed
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score

    from predictive_maintenance.config import load_config
    from predictive_maintenance.data import schema
    from predictive_maintenance.data.ingest import load_dataset
    from predictive_maintenance.evaluation.compare import cv_folds, score_readings
    from predictive_maintenance.evaluation.split import (
        repeated_stratified_cv,
        stratified_holdout,
    )
    from predictive_maintenance.training.candidates import build_candidates

    return (
        Parallel,
        Path,
        RandomForestClassifier,
        alt,
        average_precision_score,
        build_candidates,
        cv_folds,
        delayed,
        json,
        load_config,
        load_dataset,
        mo,
        np,
        pl,
        replace,
        repeated_stratified_cv,
        schema,
        score_readings,
        stratified_holdout,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 06 · Overfitting check for the random forest (no ADR; supports the README decision 7)

    **Question.** Does the default random forest (300 unweighted trees, unlimited depth) overfit? Is it
    sensitive to the number of trees or to the depth?

    **Method.** The **train partition only**, with the same repeated stratified 5×5 CV as ADR 0005
    (preprocessing fitted inside every fold). For each fold the pipeline is fitted and scored twice: on
    the rows it was fitted on (*in-sample*) and on the validation rows it never saw. The gap between
    the two is the overfitting signature. A forest always scores close to perfect in-sample, so the
    in-sample number alone says nothing; what matters is whether the **validation** PR-AUC improves
    when capacity is reduced.

    **The Holdout is not rescored.** It was scored once (ADR 0005). Its committed PR-AUC is only read
    from `reports/holdout-metrics.json` for comparison, and no choice below is made from it.

    **Decision rule, written before the numbers.** The default is vindicated if no constrained forest
    (depth or tree count) beats it on validation PR-AUC by more than the same `0.02` margin used for
    model selection in ADR 0006.
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
    base = build_candidates(seed)["random_forest_unweighted"]
    mo.md(
        f"Seed `{seed}`. Train partition: **{train.height:,} Readings, "
        f"{int(train[schema.TARGET].sum())} Machine failures**; {len(folds)} folds."
    )
    return base, config, folds, seed, train


@app.cell
def _(RandomForestClassifier, base, mo, replace, seed):
    # Default is 300 trees with no depth limit. One factor varies at a time.
    settings = (
        [("trees", n, {"n_estimators": n}) for n in (10, 50, 100, 300, 600)]
        + [("depth", d, {"max_depth": d}) for d in (3, 5, 8, 12)]
        + [("min leaf", m, {"min_samples_leaf": m}) for m in (2, 5, 10)]
    )

    def variant(kwargs):
        params = {"n_estimators": 300, "random_state": seed, "n_jobs": 1} | kwargs
        return replace(base, make_model=lambda: RandomForestClassifier(**params))

    variants = {f"{factor}={value}": variant(kw) for factor, value, kw in settings}
    variants["default"] = variant({})
    mo.md(
        f"{len(variants)} unweighted forests: " + ", ".join(f"`{k}`" for k in variants)
    )
    return (variants,)


@app.cell
def _(
    Parallel,
    average_precision_score,
    delayed,
    folds,
    pl,
    schema,
    score_readings,
    train,
    variants,
):
    x = train.select(schema.FEATURE_COLUMNS)
    y = train[schema.TARGET].to_numpy()

    def one_fit(name, candidate, fold, fit_idx, val_idx):
        pipeline = candidate.make_pipeline(None)
        pipeline.fit(x[fit_idx], y[fit_idx])
        return (
            name,
            fold,
            float(average_precision_score(y[fit_idx], score_readings(pipeline, x[fit_idx]))),
            float(average_precision_score(y[val_idx], score_readings(pipeline, x[val_idx]))),
        )

    rows = Parallel(n_jobs=-1)(
        delayed(one_fit)(name, cand, i, fit_idx, val_idx)
        for name, cand in variants.items()
        for i, (fit_idx, val_idx) in enumerate(folds)
    )
    scores = pl.DataFrame(
        rows, schema=["variant", "fold", "in_sample", "validation"], orient="row"
    ).with_columns(gap=pl.col("in_sample") - pl.col("validation"))
    return (scores,)


@app.cell
def _(mo, pl, scores):
    summary = (
        scores.group_by("variant")
        .agg(
            in_sample=pl.col("in_sample").mean(),
            validation=pl.col("validation").mean(),
            validation_sd=pl.col("validation").std(),
            gap=pl.col("gap").mean(),
        )
        .sort("validation", descending=True)
    )
    default_validation = summary.filter(pl.col("variant") == "default")["validation"][0]
    summary = summary.with_columns(delta_vs_default=pl.col("validation") - default_validation)
    mo.vstack([mo.md("### Mean PR-AUC over 25 folds (train partition)"), summary])
    return default_validation, summary


@app.cell
def _(alt, mo, summary):
    long = summary.unpivot(
        index="variant", on=["in_sample", "validation"], variable_name="scored_on", value_name="pr_auc"
    )
    chart = (
        alt.Chart(alt.Data(values=long.to_dicts()))
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X("pr_auc:Q", title="Mean PR-AUC", scale=alt.Scale(zero=False)),
            y=alt.Y("variant:N", sort=alt.EncodingSortField("pr_auc", op="min", order="descending")),
            color="scored_on:N",
        )
        .properties(width=520, height=320)
    )
    mo.ui.altair_chart(chart)
    return


@app.cell
def _(Path, json, mo, pl, summary):
    reported = json.loads(Path("reports/holdout-metrics.json").read_text())
    holdout_pr_auc = reported["metrics"]["pr_auc"]["estimate"]
    default = summary.filter(pl.col("variant") == "default").row(0, named=True)
    mo.md(
        f"""
        ### Default forest across the three scores

        | Scored on | PR-AUC |
        |---|---|
        | In-sample (fitted rows) | {default["in_sample"]:.3f} |
        | CV validation (train partition) | {default["validation"]:.3f} |
        | Holdout (read from the committed report, not rescored) | {holdout_pr_auc:.3f} |
        """
    )
    return default, holdout_pr_auc, reported


@app.cell
def _(default_validation, mo, summary):
    margin = 0.02
    best = summary.sort("validation", descending=True).row(0, named=True)
    beaten = best["validation"] - default_validation > margin
    mo.md(
        f"""
        ### Verdict

        Best validation PR-AUC: `{best["variant"]}` at {best["validation"]:.3f}, against the default's
        {default_validation:.3f} (margin {margin}). **{
            "A constrained forest beats the default by more than the margin: revisit ADR 0006."
            if beaten
            else "No variant beats the default by more than the margin: capacity is not the "
            "bottleneck and the default stays."
        }**
        """
    )
    return


if __name__ == "__main__":
    app.run()
