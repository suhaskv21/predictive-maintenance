import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_score, recall_score
    from sklearn.model_selection import (
        RepeatedStratifiedKFold,
        cross_val_score,
        train_test_split,
    )
    from sklearn.pipeline import make_pipeline

    from predictive_maintenance.config import load_config
    from predictive_maintenance.data import schema
    from predictive_maintenance.data.ingest import load_dataset
    from predictive_maintenance.preprocessing.features import (
        ENGINEERED_FEATURES,
        build_feature_transformer,
    )

    return (
        ENGINEERED_FEATURES,
        HistGradientBoostingClassifier,
        LogisticRegression,
        RandomForestClassifier,
        RepeatedStratifiedKFold,
        alt,
        average_precision_score,
        build_feature_transformer,
        cross_val_score,
        load_config,
        load_dataset,
        make_pipeline,
        mo,
        np,
        pl,
        precision_score,
        recall_score,
        schema,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 01 · Target and features

    Concludes **ADR 0004** (status `Proposed`): what we predict, which columns are inputs,
    which are excluded, and whether engineered physics features earn a place.
    Plan: [`docs/design/decision-plan.md`](../docs/design/decision-plan.md).
    Terms follow [`GLOSSARY.md`](../GLOSSARY.md): a **Reading** is one snapshot of sensor values plus
    `Type`; **Machine failure** is the target; **Failure modes** are outcome fields, never inputs.

    **How to read this notebook.** Every model comparison uses the *training partition only*
    (stratified 80/20 split, seed from `configs/default.toml`) and repeated 5×5 stratified CV,
    scored by PR-AUC. The 20 % holdout is never evaluated here. The split strategy itself is
    decided in notebook 02, so the split below is provisional. All preprocessing is fitted
    inside each CV fold.
    """)
    return


@app.cell
def _(load_config, load_dataset, schema):
    df = load_dataset()
    seed = load_config().seed
    n_failures = int(df[schema.TARGET].sum())
    return df, n_failures, seed


@app.cell
def _(df):
    inspect_df = df
    inspect_df
    return


@app.cell
def _(df, mo, n_failures):
    mo.md(f"""
    ## 0 · Data and evaluation set-up

    `data/raw/ai4i2020.csv` passes the validation stage (required columns and dtypes, no nulls,
    unique `UDI`, known `Type` levels, binary flags): **{df.height:,} Readings**, {df.width} columns,
    **{n_failures} Machine failures ({n_failures / df.height:.2%})**.
    Class imbalance is why PR-AUC, not accuracy, drives every comparison below.
    """)
    return


@app.cell
def _(RepeatedStratifiedKFold, df, schema, seed, train_test_split):
    train_df, _holdout = train_test_split(
        df, test_size=0.2, stratify=df[schema.TARGET], random_state=seed
    )
    X_train = train_df.select(schema.FEATURE_COLUMNS)
    y_train = train_df[schema.TARGET].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    prevalence = float(y_train.mean())
    return X_train, cv, prevalence, train_df, y_train


@app.cell
def _(
    LogisticRegression,
    RandomForestClassifier,
    cross_val_score,
    cv,
    pl,
    seed,
):
    def cv_scores(estimator, X, y):
        """PR-AUC (average precision) per fold of the repeated 5x5 stratified CV."""
        return cross_val_score(estimator, X, y, cv=cv, scoring="average_precision")

    def markdown_table(frame):
        """Render a polars frame as a Markdown table."""
        with pl.Config(
            tbl_formatting="ASCII_MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_rows=100,
            tbl_cols=20,
            float_precision=3,
        ):
            return str(frame)

    # Untuned defaults: this notebook compares feature sets, it does not select a model (ADR 0006).
    nonlinear_model = "Random forest"
    models = {
        "Logistic regression": lambda: LogisticRegression(max_iter=1000),
        nonlinear_model: lambda: RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
    }
    return cv_scores, markdown_table, models, nonlinear_model


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Target

    The decision plan fixes a binary target. The table checks that choice against the data
    instead of assuming it.
    """)
    return


@app.cell
def _(df, mo, pl, schema):
    mode_counts = {name: int(df[name].sum()) for name in schema.FAILURE_MODES}
    any_mode = pl.sum_horizontal(schema.FAILURE_MODES) > 0
    failure_without_mode = df.filter((pl.col(schema.TARGET) == 1) & ~any_mode).height
    mode_without_failure = df.filter((pl.col(schema.TARGET) == 0) & any_mode).height
    smallest_mode = min(mode_counts, key=mode_counts.get)

    mo.md(f"""
    | Option | Positives available | Operational fit | Verdict |
    |---|---|---|---|
    | **Binary `Machine failure`** | {int(df[schema.TARGET].sum())} | Matches the spec default (§3) and the engineer's question: "does this Reading need attention?" | **Recommended** |
    | Multi-class failure mode (TWF/HDF/PWF/OSF/RNF) | {", ".join(f"{k} {v}" for k, v in mode_counts.items())} | Would say *why*, but {smallest_mode} has only {mode_counts[smallest_mode]} positives and modes overlap | **Deferred, not disproven**: per-mode predictability is checked in notebook 03 |
    | One binary model per mode | same as above, split five ways | Five models, five thresholds, five sets of caveats | **Deferred**, same check |

    The target and the modes are not interchangeable: **{failure_without_mode}** failures carry no
    mode flag and **{mode_without_failure}** Readings carry a mode flag without a recorded failure,
    so the modes are a separate, imperfect annotation of the outcome.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Leakage evidence

    Two kinds of columns could make a model look good for the wrong reason.
    """)
    return


@app.cell
def _(
    LogisticRegression,
    average_precision_score,
    cv_scores,
    mo,
    pl,
    precision_score,
    prevalence,
    recall_score,
    schema,
    train_df,
    y_train,
):
    flag_count = train_df.select(n=pl.sum_horizontal(schema.FAILURE_MODES))["n"].to_numpy()
    flag_pr_auc = average_precision_score(y_train, flag_count)
    flag_precision = precision_score(y_train, flag_count > 0)
    flag_recall = recall_score(y_train, flag_count > 0)
    flag_cv = cv_scores(
        LogisticRegression(max_iter=1000), train_df.select(schema.FAILURE_MODES), y_train
    )

    mo.md(f"""
    ### 2a · Failure-mode flags alone predict Machine failure almost perfectly

    The flags are *recorded causes of the failure*, so they encode the target.

    | Predictor of `Machine failure` | PR-AUC | Precision | Recall |
    |---|---|---|---|
    | "any mode flag set" rule (training partition) | {flag_pr_auc:.3f} | {flag_precision:.3f} | {flag_recall:.3f} |
    | Logistic regression on the 5 flags (repeated-CV, mean ± SD) | {flag_cv.mean():.3f} ± {flag_cv.std():.3f} | | |
    | Chance (prevalence) | {prevalence:.3f} | | |

    A Reading's mode flags exist only *after* a failure has been diagnosed, so they are
    unavailable when a Reading is assessed. **Excluded: `TWF, HDF, PWF, OSF, RNF`.**
    """)
    return


@app.cell
def _(
    HistGradientBoostingClassifier,
    cv_scores,
    df,
    markdown_table,
    mo,
    np,
    pl,
    prevalence,
    seed,
    train_df,
    y_train,
):
    id_predictors = train_df.select(
        pl.col("UDI"),
        pl.col("Product ID").str.slice(1).cast(pl.Int64).alias("Product ID number"),
        # Control: a seeded random column shows what "no signal" scores under this protocol.
        pl.Series(
            "Random noise (control)", np.random.default_rng(seed).uniform(size=train_df.height)
        ),
    )
    id_results = {}
    for id_name in id_predictors.columns:
        _scores = cv_scores(
            HistGradientBoostingClassifier(random_state=seed),
            id_predictors.select(id_name),
            y_train,
        )
        id_results[id_name] = (float(_scores.mean()), float(_scores.std()))

    noise_mean, noise_sd = id_results["Random noise (control)"]
    id_has_signal = any(
        m > noise_mean + 2 * noise_sd
        for name, (m, _) in id_results.items()
        if name != "Random noise (control)"
    )
    id_table = pl.DataFrame(
        {
            "single-column predictor": ["Chance (prevalence)", *id_results],
            "PR-AUC mean": [prevalence, *(m for m, _ in id_results.values())],
            "SD": [None, *(sd for _, sd in id_results.values())],
        }
    )

    id_unique = df["Product ID"].n_unique()
    prefix_is_type = bool((df["Product ID"].str.slice(0, 1) == df["Type"]).all())
    udi_blocks = (
        train_df.with_columns(udi_block=(pl.col("UDI") - 1) // 1000 * 1000 + 1)
        .group_by("udi_block")
        .agg(
            readings=pl.len(),
            failure_rate=pl.col("Machine failure").mean(),
            hdf_rate=pl.col("HDF").mean(),
        )
        .sort("udi_block")
    )
    peak = udi_blocks.sort("failure_rate", descending=True).row(0, named=True)
    rest = udi_blocks.filter(pl.col("udi_block") != peak["udi_block"])
    rest_rate = float((rest["failure_rate"] * rest["readings"]).sum() / rest["readings"].sum())

    mo.md(f"""
    ### 2b · Identifiers: a weak signal that is a dataset artifact, not machine physics

    - `Product ID` has **{id_unique:,} unique values for {df.height:,} Readings**: every machine appears
      once, so a held-out Reading's ID is never seen in training. There is no machine history to
      learn, and no group structure to leak across the split.
    - Its first letter equals `Type` in **{"every" if prefix_is_type else "NOT every"}** Reading
      (`Type` already supplies that information). The remainder is a serial number. `UDI` is the row index.

    {markdown_table(id_table)}

    {"The identifiers beat the noise control (by more than 2 SD of the control), so there *is* a signal." if id_has_signal else "Both identifiers stay within the noise of the control."}
    Where it comes from: failures cluster in one block of rows.

    {markdown_table(udi_blocks)}

    Rows `UDI` {peak["udi_block"]}–{peak["udi_block"] + 999} fail at **{peak["failure_rate"]:.1%}** (heat-dissipation
    flag `HDF` {peak["hdf_rate"]:.1%}) against **{rest_rate:.1%}** elsewhere. A model using the ID
    would learn *where a row sits in the file*, not how a machine behaves. A new Reading entered in
    the app has no such position, so the "gain" would not exist in use and would inflate offline scores.
    **Excluded: `UDI`, `Product ID`**, on those grounds (not on absence of signal).
    """)
    return id_has_signal, peak, rest_rate


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · What the sensors and `Type` look like

    Descriptive only, on the training partition.
    """)
    return


@app.cell
def _(alt, mo, pl, schema, train_df):
    by_type = train_df.group_by(schema.CATEGORICAL_FEATURE).agg(
        failure_rate=pl.col(schema.TARGET).mean(),
        readings=pl.len(),
    )
    type_chart = (
        alt.Chart(by_type)
        .mark_bar()
        .encode(
            x=alt.X("Type:N", sort=list(schema.TYPE_LEVELS)),
            y=alt.Y(
                "failure_rate:Q",
                title="Machine failure rate",
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                "Type",
                "readings",
                alt.Tooltip("failure_rate:Q", format=".2%"),
            ],
        )
        .properties(title="Failure rate by Type", width=220, height=220)
    )
    scatter = (
        alt.Chart(
            # Vega-Lite parses "[...]" in field names as nested access, so alias the columns.
            train_df.select(
                pl.col("Torque [Nm]").alias("Torque"),
                pl.col("Rotational speed [rpm]").alias("Rotational speed"),
                schema.TARGET,
            )
        )
        .mark_circle(size=18)
        .encode(
            x=alt.X(
                "Rotational speed:Q",
                title="Rotational speed [rpm]",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "Torque:Q",
                title="Torque [Nm]",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "Machine failure:N",
                scale=alt.Scale(domain=[0, 1], range=["#c8c8c8", "#d62728"]),
            ),
            order=alt.Order("Machine failure:Q"),
            opacity=alt.condition(
                alt.datum["Machine failure"] == 1,
                alt.value(0.9),
                alt.value(0.35),
            ),
        )
        .properties(title="Torque vs rotational speed", width=380, height=260)
    )
    mo.ui.altair_chart(type_chart | scatter)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Do engineered physics features earn a place?

    Candidates (all derivable from the five sensor inputs, so the app asks for nothing extra):

    - **Power** = torque × angular speed (W)
    - **Temperature delta** = process − air temperature (K)
    - **Wear × torque** (min·Nm): load accumulated on a worn tool

    **Decision rule (plan, amended 2026-10-12):** keep an engineered feature if it improves
    repeated-CV PR-AUC (mean paired Δ > 0) for the **non-linear** model; the linear model is not
    required to improve. A feature also stays if it is needed for interpretability.
    Each model is run on identical folds, so differences are *paired*.

    *History of the rule.* The plan first required a gain for **both** a linear and a non-linear
    model. After seeing these results the rule was relaxed: after standardisation, *Temperature
    delta* is an exact linear combination of two raw inputs, so a linear model has almost nothing
    to gain from it, and "both" rejects such features by construction. Because the rule was
    changed after the evidence was seen, the original per-model deltas stay visible below.
    """)
    return


@app.cell
def _(
    ENGINEERED_FEATURES,
    X_train,
    build_feature_transformer,
    cv_scores,
    make_pipeline,
    models,
    pl,
    y_train,
):
    candidate_sets = {
        "raw": (),
        **{f"+ {name}": (name,) for name in ENGINEERED_FEATURES},
        "+ all three": ENGINEERED_FEATURES,
    }
    score_rows = []
    for model_name, make_model in models.items():
        for set_name, engineered in candidate_sets.items():
            _pipeline = make_pipeline(build_feature_transformer(engineered), make_model())
            for fold, value in enumerate(cv_scores(_pipeline, X_train, y_train)):
                score_rows.append((model_name, set_name, fold, float(value)))
    fold_scores = pl.DataFrame(
        score_rows,
        schema=["model", "feature_set", "fold", "pr_auc"],
        orient="row",
    )
    return (fold_scores,)


@app.cell
def _(fold_scores, pl):
    def summarize(scores):
        """Mean PR-AUC per (model, feature set) with paired delta vs raw and its SE over folds."""
        raw = scores.filter(pl.col("feature_set") == "raw").select(
            "model", "fold", raw_pr_auc="pr_auc"
        )
        return (
            scores.join(raw, on=["model", "fold"])
            .with_columns(delta=pl.col("pr_auc") - pl.col("raw_pr_auc"))
            .group_by("model", "feature_set", maintain_order=True)
            .agg(
                pr_auc=pl.col("pr_auc").mean(),
                sd=pl.col("pr_auc").std(),
                delta=pl.col("delta").mean(),
                delta_se=pl.col("delta").std() / pl.col("delta").len().sqrt(),
            )
        )

    summary = summarize(fold_scores)
    return summarize, summary


@app.cell
def _(ENGINEERED_FEATURES, nonlinear_model, summary):
    def delta_of(model_name, feature_set):
        return summary.filter(
            (summary["model"] == model_name) & (summary["feature_set"] == feature_set)
        )["delta"].item()

    keep_set = tuple(
        name for name in ENGINEERED_FEATURES if delta_of(nonlinear_model, f"+ {name}") > 0
    )
    return delta_of, keep_set


@app.cell
def _(markdown_table, mo, pl, summary):
    grid = (
        summary.filter(pl.col("feature_set").is_in(["raw", "+ all three"]))
        .with_columns(cell=pl.format("{} ± {}", pl.col("pr_auc").round(3), pl.col("sd").round(3)))
        .pivot(on="feature_set", index="model", values="cell")
        .select("model", "raw", "+ all three")
    )

    mo.md(f"""
    ### 4a · The 2×2 (linear/non-linear × raw/engineered)

    Mean repeated-CV PR-AUC ± SD across the 25 folds.

    {markdown_table(grid)}
    """)
    return


@app.cell
def _(markdown_table, mo, pl, summary):
    per_feature = summary.filter(pl.col("feature_set") != "raw").select(
        "model",
        "feature_set",
        pr_auc=pl.col("pr_auc").round(3),
        delta=pl.col("delta").round(4),
        delta_se=pl.col("delta_se").round(4),
        within_1_se=pl.col("delta").abs() < pl.col("delta_se"),
    )
    mo.md(f"""
    ### 4b · Paired gain over raw, one engineered feature at a time

    `delta` is the mean paired difference in PR-AUC against the raw feature set on identical folds;
    `delta_se` its standard error over the 25 folds. Folds from repeated CV overlap, so the SE is
    optimistic. Treat `within_1_se = true` as "not clearly different from zero".

    {markdown_table(per_feature)}
    """)
    return


@app.cell
def _(alt, mo, pl, summary):
    deltas = summary.filter(pl.col("feature_set") != "raw").with_columns(
        lo=pl.col("delta") - pl.col("delta_se"),
        hi=pl.col("delta") + pl.col("delta_se"),
    )
    base = alt.Chart(deltas).encode(
        y=alt.Y("feature_set:N", title=None, sort=None),
        color=alt.Color("model:N", legend=None),
    )
    points = base.mark_circle(size=70).encode(
        x=alt.X("delta:Q", title="Paired gain in PR-AUC over raw (± 1 SE)")
    )
    bars = base.mark_errorbar().encode(
        x=alt.X("lo:Q", title="Paired gain in PR-AUC over raw (± 1 SE)"), x2="hi:Q"
    )
    delta_chart = (
        (bars + points)
        .properties(width=320, height=140)
        .facet(row=alt.Row("model:N", title=None))
        .properties(title="Engineered features vs raw")
    )
    mo.ui.altair_chart(delta_chart)
    # Static copy for the README. Needs `uv run --with vl-convert-python`; skipped otherwise.
    _figures = mo.notebook_dir().parent / "docs" / "figures"
    try:
        _figures.mkdir(parents=True, exist_ok=True)
        delta_chart.save(_figures / "engineered-vs-raw.png", scale_factor=2)
    except (ImportError, ValueError):
        pass
    return


@app.cell
def _(
    ENGINEERED_FEATURES,
    X_train,
    build_feature_transformer,
    cv_scores,
    fold_scores,
    keep_set,
    make_pipeline,
    models,
    pl,
    summarize,
    y_train,
):
    kept_rows = []
    for _model_name, _make_model in models.items():
        _pipeline = make_pipeline(build_feature_transformer(keep_set), _make_model())
        for _fold, _value in enumerate(cv_scores(_pipeline, X_train, y_train)):
            kept_rows.append((_model_name, "recommended set", _fold, float(_value)))
    kept_scores = pl.DataFrame(
        kept_rows, schema=["model", "feature_set", "fold", "pr_auc"], orient="row"
    )
    set_level = summarize(
        pl.concat(
            [
                fold_scores.filter(pl.col("feature_set") == "raw"),
                kept_scores,
                # Avoid a duplicate row when the kept set is all three features.
                *(
                    []
                    if keep_set == ENGINEERED_FEATURES
                    else [fold_scores.filter(pl.col("feature_set") == "+ all three")]
                ),
            ]
        )
    ).filter(pl.col("feature_set") != "raw")
    return (set_level,)


@app.cell(expand_output=True)
def _(
    ENGINEERED_FEATURES,
    delta_of,
    id_has_signal,
    keep_set,
    markdown_table,
    mo,
    models,
    nonlinear_model,
    peak,
    pl,
    rest_rate,
    schema,
    set_level,
    summary,
):
    verdicts = "\n".join(
        f"| {name} | "
        + " | ".join(f"{delta_of(m, f'+ {name}'):+.4f}" for m in models)
        + f" | {'**keep**' if name in keep_set else 'drop'} |"
        for name in ENGINEERED_FEATURES
    )
    model_headers = " | ".join(f"Δ PR-AUC, {m}" for m in models)
    set_check = set_level.select(
        "model",
        "feature_set",
        pr_auc=pl.col("pr_auc").round(3),
        delta=pl.col("delta").round(4),
        delta_se=pl.col("delta_se").round(4),
    ).sort("model", "feature_set", descending=[False, True])

    linear_flat = [
        f"{name} ({', '.join(f'{m} {delta_of(m, f"+ {name}"):+.4f}' for m in models if m != nonlinear_model)})"
        for name in keep_set
        if any(delta_of(m, f"+ {name}") <= 0 for m in models if m != nonlinear_model)
    ]
    linear_flat_text = (
        "; ".join(linear_flat)
        if linear_flat
        else "none: every kept feature also helps the linear model"
    )
    weak = [
        f"{row['model']} / {row['feature_set']}"
        for row in summary.filter(
            (pl.col("model") == nonlinear_model)
            & pl.col("feature_set").is_in([f"+ {name}" for name in keep_set])
            & (pl.col("delta").abs() < pl.col("delta_se"))
        ).iter_rows(named=True)
    ]
    weak_text = (
        "Kept features with a non-linear gain within 1 SE of zero: " + "; ".join(weak) + "."
        if weak
        else "Every kept feature's non-linear gain exceeds 1 SE of zero."
    )
    kept_text = ", ".join(f"`{name}`" for name in keep_set) if keep_set else "none"
    dropped_text = ", ".join(f"`{n}`" for n in ENGINEERED_FEATURES if n not in keep_set) or "none"
    modes_text = ", ".join(f"`{m}`" for m in schema.FAILURE_MODES)
    id_text = (
        "a signal exists but is a row-position artifact (§2b)"
        if id_has_signal
        else "no signal beyond the noise control (§2b)"
    )

    mo.md(f"""
    ## 5 · Recommendation (follows the amended rule)

    ### Engineered-feature trade-off

    | Feature | {model_headers} | Rule outcome (non-linear Δ > 0?) |
    |---|---|---|---|
    {verdicts}

    Interpretability is the rule's second arm. Each feature has a physical meaning an engineer
    recognises, but none is *needed* to explain a Reading, so that arm is not invoked: the outcome
    above rests on the CV evidence alone.

    ### Proposed decision for ADR 0004

    - **Target:** binary `Machine failure`. Multi-class failure mode is **deferred, not disproven**;
      notebook 03 measures per-mode predictability and ADR 0009 uses it to decide whether mode
      prediction becomes a secondary output.
    - **Excluded columns:** {modes_text} (outcome fields; leakage, §2a);
      `UDI` and `Product ID` (identifiers: their signal is a row-position artifact, rows
      `UDI` {peak["udi_block"]}–{peak["udi_block"] + 999} fail at {peak["failure_rate"]:.1%} vs {rest_rate:.1%}
      elsewhere, and a new Reading has no such position; §2b).
    - **Features:** `Type` + the 5 sensors, plus engineered: **{kept_text}**.
      Dropped by the rule: {dropped_text}.
    - Engineered features are computed *inside* the serialized scikit-learn `Pipeline`
      (`predictive_maintenance.preprocessing.features`), so the app never reimplements them.

    Kept set against raw, same folds:

    {markdown_table(set_check)}

    ### Caveats for the reviewer

    - **The rule was amended after the evidence was seen** (previously: gain required for both
      models). The earlier outcome is the table above read on the linear column.
    - Kept without a linear-model gain: {linear_flat_text}. If ADR 0006 selects the linear model,
      revisit: these features then add inputs and complexity for no measured benefit.
    - {weak_text}
    - Models use untuned defaults; this compares feature sets, not model families (ADR 0006).
    - The 80/20 split used here is provisional until ADR 0005; the holdout was never evaluated.
    - Identifier check: {id_text}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decision: Suhas decision for ADR 0004
    1. Target: keep the binary target for the ADR, but record the multi-class exclusion as "deferred, not disproven". Then add a short check to
    notebook 01, or a later one, that measures per-mode predictability from the sensors with PR-AUC per mode. The result would decide whether mode
    prediction becomes a secondary output in ADR 0009. I haven't added that check, since step 1 was scoped to the plan as written.
    2. Excluded columns: TWF, HDF, PWF, OSF, RNF (outcome fields; leakage, §2a); UDI and Product ID (identifiers: their signal is a row-position artifact, rows UDI 4001–5000 fail at 13.1% vs 2.3% elsewhere, and a new Reading has no such position; §2b).
    3. Features: Lets amend the plan such that "Decision rule (plan): keep an engineered feature only if it improves CV PR-AUC for both a linear and a non-linear model, or is needed for interpretability." to include features even if only improves CV PR-AUC on non-linear model only. so lets keep Type + the 5 sensors, plus all engineered features
    """)
    return


if __name__ == "__main__":
    app.run()
