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
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from predictive_maintenance.config import load_config
    from predictive_maintenance.data import schema
    from predictive_maintenance.data.ingest import load_dataset
    from predictive_maintenance.evaluation.split import stratified_holdout
    from predictive_maintenance.preprocessing.features import add_engineered_features
    from predictive_maintenance.training.candidates import build_candidates

    return (
        StratifiedKFold,
        add_engineered_features,
        alt,
        build_candidates,
        cross_val_predict,
        load_config,
        load_dataset,
        mo,
        np,
        pl,
        replace,
        schema,
        stratified_holdout,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 07 · Is the failure signal non-linear? (supports the README decision on a non-linear model)

    **Question.** Would a linear model do, or is a non-linear one needed?

    **Method.** The **train partition only** (the Holdout is discarded at load time). Out-of-fold
    failure probabilities from a 5-fold stratified CV, for the unweighted logistic regression (the
    linear model of ADR 0006, on the same features as the forest), a logistic regression on the
    raw sensors only, and the default random forest. Each sensor is cut into ten equal-count bins.
    A **binned residual** is the observed failure rate in a bin minus the mean predicted
    probability there. A model whose log-odds are linear in the sensors, if that were the true
    shape, would leave residuals that scatter around zero. Residuals that are systematically
    positive at the ends and negative in the middle show a threshold shape the line cannot bend to.

    A raw residual plot is not used: for a 0/1 target, every residual sits on one of two bands and
    shows nothing.
    """)
    return


@app.cell
def _(
    StratifiedKFold,
    add_engineered_features,
    build_candidates,
    cross_val_predict,
    load_config,
    load_dataset,
    mo,
    replace,
    schema,
    stratified_holdout,
):
    config = load_config()
    readings = load_dataset()
    train, _holdout = stratified_holdout(readings, test_size=config.split.test_size, seed=config.seed)
    del _holdout
    y = train[schema.TARGET].to_numpy()
    x = train.select(schema.FEATURE_COLUMNS)
    candidates = build_candidates(config.seed)
    models = {
        "logistic (all features)": candidates["logistic_unweighted"],
        "logistic (raw sensors)": replace(candidates["logistic_unweighted"], engineered=()),
        "random forest": candidates["random_forest_unweighted"],
    }
    cv = StratifiedKFold(config.split.cv_folds, shuffle=True, random_state=config.seed)
    oof = {
        name: cross_val_predict(c.make_pipeline(None), x, y, cv=cv, method="predict_proba")[:, 1]
        for name, c in models.items()
    }
    frame = add_engineered_features(train)
    mo.md(
        f"Train partition: **{train.height:,} Readings, {int(y.sum())} Machine failures**, "
        f"{config.split.cv_folds}-fold out-of-fold predictions for {len(models)} models."
    )
    return frame, oof, y


@app.cell
def _(frame, np, oof, pl, y):
    SENSORS = [
        "Torque [Nm]",
        "Rotational speed [rpm]",
        "Power [W]",
        "Tool wear [min]",
        "Temperature delta [K]",
    ]

    def binned(feature: str) -> pl.DataFrame:
        values = frame[feature].to_numpy()
        edges = np.quantile(values, np.linspace(0, 1, 11))
        bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, 9)
        rows = []
        for k in range(10):
            mask = bins == k
            observed = float(y[mask].mean())
            for model, p in oof.items():
                rows.append(
                    (
                        feature,
                        k + 1,
                        float(edges[k]),
                        int(mask.sum()),
                        model,
                        observed,
                        float(p[mask].mean()),
                    )
                )
        return pl.DataFrame(
            rows,
            schema=["feature", "bin", "lower_edge", "n", "model", "observed", "predicted"],
            orient="row",
        ).with_columns(residual=pl.col("observed") - pl.col("predicted"))

    table = pl.concat([binned(f) for f in SENSORS])
    return SENSORS, table


@app.cell
def _(alt, mo, table):
    base = alt.Chart(alt.Data(values=table.to_dicts()))
    lines = base.mark_line(point=True).encode(
        x=alt.X("bin:O", title="Decile of the sensor (1 = lowest)"),
        y=alt.Y("residual:Q", title="Observed minus predicted failure rate"),
        color=alt.Color("model:N", title=None),
    )
    zero = base.mark_rule(color="grey", strokeDash=[3, 3]).encode(y=alt.datum(0))
    chart = (
        alt.layer(lines, zero)
        .facet(column=alt.Column("feature:N", title=None))
        .properties(title="Binned residuals by sensor decile (out-of-fold, train partition)")
    )
    mo.vstack([chart])
    return


@app.cell
def _(mo, pl, table):
    extremes = (
        table.filter(
            (pl.col("feature").is_in(["Tool wear [min]", "Temperature delta [K]"]))
            & (pl.col("bin").is_in([1, 5, 10]))
        )
        .select("feature", "bin", "lower_edge", "n", "model", "observed", "predicted", "residual")
        .sort("feature", "bin", "model")
    )
    mo.vstack([mo.md("### Tool wear and temperature delta, low, middle and top decile"), extremes])
    return


@app.cell
def _(mo, pl, table):
    fit = (
        table.group_by("model")
        .agg(
            mean_abs_residual=pl.col("residual").abs().mean(),
            worst_abs_residual=pl.col("residual").abs().max(),
        )
        .sort("mean_abs_residual")
    )
    mo.vstack([mo.md("### Size of the binned residuals, all five sensors"), fit])
    return


if __name__ == "__main__":
    app.run()
