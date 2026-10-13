import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from joblib import Parallel, delayed
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline

    from predictive_maintenance.config import load_config
    from predictive_maintenance.data import schema
    from predictive_maintenance.data.ingest import load_dataset
    from predictive_maintenance.evaluation.split import (
        repeated_stratified_cv,
        stratified_holdout,
    )
    from predictive_maintenance.preprocessing.features import (
        ENGINEERED_FEATURES,
        build_feature_transformer,
    )

    return (
        ENGINEERED_FEATURES,
        HistGradientBoostingClassifier,
        LogisticRegression,
        Parallel,
        alt,
        average_precision_score,
        build_feature_transformer,
        delayed,
        load_config,
        load_dataset,
        make_pipeline,
        mo,
        np,
        pl,
        repeated_stratified_cv,
        schema,
        stratified_holdout,
        time,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 02 · Split strategy and validation

    Concludes **ADR 0005** (status `Proposed`): how the data is split, and how models are compared and
    thresholds chosen on the training partition. Plan: [`docs/design/decision-plan.md`](../docs/design/decision-plan.md).
    Terms follow [`GLOSSARY.md`](../GLOSSARY.md).

    **Taken as given from ADR 0004 (not reopened):** binary `Machine failure`; features `Type` + the
    5 sensors + all three engineered features (power, temperature delta, wear × torque); failure
    modes and identifiers excluded.

    **Default under test:** stratified 80/20 holdout, touched once at the end, plus repeated 5×5
    stratified CV on the train partition for every comparison and threshold choice.
    **Evidence wins over the default.** Each comparison below states its decision rule *before*
    the numbers, and the rule can overturn the default.

    **The probe.** Logistic regression on the ADR 0004 feature set (cheap, deterministic, and with a
    clear probability scale). This notebook does **not** choose the model family (notebook 03). In
    comparison 3 a `HistGradientBoostingClassifier` joins as a second probe, because its learning
    curve is steeper and so it exposes the cost of a smaller training partition.

    **The held-out test partition is never used for selection.** The real holdout (seed from
    `configs/default.toml`) is set aside in §0 and never scored. Comparisons 1 and 3 *re-split the full
    data* many times and score those throwaway test sets purely to measure how much a split lottery
    moves the numbers. Comparison 2 runs on the train partition only.

    *Runtime: about 5 minutes on 12 cores (loops run in parallel with joblib); all seeds derive from
    the config seed, so reruns reproduce the numbers.*
    """)
    return


@app.cell
def _(load_config, load_dataset, np, schema, stratified_holdout):
    df = load_dataset()
    seed = load_config().seed
    n_failures = int(df[schema.TARGET].sum())

    def derived_seeds(stream: int, n: int) -> list[int]:
        """n reproducible seeds derived from the single config seed, one stream per comparison."""
        return [int(s) for s in np.random.SeedSequence([seed, stream]).generate_state(n)]

    # The real holdout is discarded here: nothing in this notebook scores it.
    train_df, _holdout = stratified_holdout(df, test_size=0.2, seed=seed)
    return derived_seeds, df, n_failures, seed, train_df


@app.cell
def _(df, mo, n_failures, seed, train_df):
    mo.md(f"""
    ## 0 · Set-up

    {df.height:,} Readings, **{n_failures} Machine failures ({n_failures / df.height:.2%})**, seed `{seed}`
    from `configs/default.toml`. The provisional 80/20 stratified partition used by comparison 2
    has {train_df.height:,} train Readings ({int(train_df["Machine failure"].sum())} failures).
    With so few positives, the test set of a 80/20 split holds only about {round(0.2 * n_failures)} of
    them, which is why split variance is the concern here.
    """)
    return


@app.cell
def _(
    ENGINEERED_FEATURES,
    HistGradientBoostingClassifier,
    LogisticRegression,
    Parallel,
    average_precision_score,
    build_feature_transformer,
    delayed,
    make_pipeline,
    np,
    pl,
    seed,
):
    N_HOLDOUT_SEEDS = 200
    N_SELECTION_REPLICATES = 30
    N_RATIO_SEEDS = 60
    N_BOOTSTRAP = 1000
    COST_RATIOS = (5, 10, 20)  # FN:FP cost ratios; 10:1 is the plan's default assumption
    DEFAULT_COST_RATIO = 10
    LOGISTIC = "Logistic regression"
    BOOSTED = "HistGradientBoosting"

    # Close-call candidates for the ranking-stability test (all one family: no family choice here).
    CANDIDATES = {
        f"C={c:g}{' balanced' if balanced else ''}": {
            "C": c,
            **({"class_weight": "balanced"} if balanced else {}),
        }
        for c in (0.01, 0.1, 1.0, 10.0)
        for balanced in (False, True)
    }
    CANDIDATE_NAMES = list(CANDIDATES)
    PROBE = "C=1"  # the plain logistic regression whose threshold is tracked
    PROBE_INDEX = CANDIDATE_NAMES.index(PROBE)

    # Expected-cost curve over a threshold grid. A placeholder rule to probe stability only;
    # the metric and threshold are decided in ADR 0007, not here.
    GRID = np.round(np.arange(0.01, 0.91, 0.01), 2)

    def cost_curve(y, p, ratio):
        """Mean cost per Reading at each grid threshold: `ratio` per missed failure, 1 per false alarm."""
        flagged = p[:, None] >= GRID
        positive = (y == 1)[:, None]
        misses = (~flagged & positive).sum(axis=0)
        false_alarms = (flagged & ~positive).sum(axis=0)
        return (ratio * misses + false_alarms) / len(y)

    def pick_threshold(curve):
        """Centre of the minimum-cost plateau (ties are common with few positives)."""
        return float(GRID[np.isclose(curve, curve.min())].mean())

    def prauc(y, p):
        """PR-AUC (average precision)."""
        return float(average_precision_score(y, p))

    def make_probe(kind):
        """Unfitted ADR 0004 pipeline (all engineered features) around the chosen probe model."""
        model = (
            LogisticRegression(max_iter=1000)
            if kind == LOGISTIC
            else HistGradientBoostingClassifier(random_state=seed)
        )
        return make_pipeline(build_feature_transformer(ENGINEERED_FEATURES), model)

    def run_parallel(function, tasks):
        """Run `function(*task)` for every task on all cores, in order."""
        return Parallel(n_jobs=-1)(delayed(function)(*task) for task in tasks)

    def markdown_table(frame):
        """Render a polars frame as a Markdown table."""
        with pl.Config(
            tbl_formatting="ASCII_MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_rows=100,
            tbl_cols=20,
            tbl_width_chars=400,
            fmt_str_lengths=200,
            float_precision=3,
        ):
            return str(frame)

    return (
        BOOSTED,
        CANDIDATES,
        CANDIDATE_NAMES,
        COST_RATIOS,
        LOGISTIC,
        N_BOOTSTRAP,
        N_HOLDOUT_SEEDS,
        N_RATIO_SEEDS,
        N_SELECTION_REPLICATES,
        PROBE_INDEX,
        cost_curve,
        make_probe,
        markdown_table,
        pick_threshold,
        prauc,
        run_parallel,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Plain vs stratified 80/20 holdout

    **Question.** Does stratifying the holdout on `Machine failure` stabilise the test set, or is a
    plain random split good enough?

    **Method.** About 200 seeds (derived from the config seed). For each seed split the full data
    80/20 both ways, fit the probe on the train part, record the number of **test positives** and the
    **test PR-AUC**. These test sets are throwaway draws to measure variance; nothing is selected from them.

    **Decision rule (stated before the numbers).** Keep **stratified** (the default) unless it fails
    to help: stratified is kept if its range of test positives is ≤ 1 *and* its seed-to-seed SD of
    test PR-AUC is no more than 10 % above the plain split's. Otherwise the plain split wins.
    Whether stratification also *reduces* PR-AUC spread by ≥ 10 % is reported separately, because
    even without that gain it guarantees the test set the positives it needs.
    """)
    return


@app.cell
def _(
    N_HOLDOUT_SEEDS,
    derived_seeds,
    df,
    make_probe,
    pl,
    prauc,
    run_parallel,
    schema,
    stratified_holdout,
    train_test_split,
):
    holdout_seeds = derived_seeds(1, N_HOLDOUT_SEEDS)

    def holdout_run(scheme, s):
        """One 80/20 split of the full data; probe PR-AUC on that split's throwaway test set."""
        if scheme == "plain":
            train, test = train_test_split(df, test_size=0.2, random_state=s)
        else:
            train, test = stratified_holdout(df, test_size=0.2, seed=s)
        model = make_probe("Logistic regression").fit(
            train.select(schema.FEATURE_COLUMNS), train[schema.TARGET].to_numpy()
        )
        y_test = test[schema.TARGET].to_numpy()
        scores = model.predict_proba(test.select(schema.FEATURE_COLUMNS))[:, 1]
        return scheme, s, int(y_test.sum()), prauc(y_test, scores)

    holdout_runs = pl.DataFrame(
        run_parallel(
            holdout_run, [(scheme, s) for scheme in ("plain", "stratified") for s in holdout_seeds]
        ),
        schema=["scheme", "seed", "test_positives", "test_pr_auc"],
        orient="row",
    )
    return (holdout_runs,)


@app.cell
def _(df, holdout_runs, np, pl, schema):
    expected_positives = 0.2 * df[schema.TARGET].sum()
    spread = (
        holdout_runs.group_by("scheme", maintain_order=True)
        .agg(
            pos_mean=pl.col("test_positives").mean(),
            pos_min=pl.col("test_positives").min(),
            pos_max=pl.col("test_positives").max(),
            pos_sd=pl.col("test_positives").std(),
            off_by_10pct=(
                (pl.col("test_positives") - expected_positives).abs() > 0.1 * expected_positives
            ).mean(),
            pr_auc_mean=pl.col("test_pr_auc").mean(),
            pr_auc_sd=pl.col("test_pr_auc").std(),
            pr_auc_p5=pl.col("test_pr_auc").quantile(0.05),
            pr_auc_p95=pl.col("test_pr_auc").quantile(0.95),
        )
        .with_columns(pr_auc_range_90=pl.col("pr_auc_p95") - pl.col("pr_auc_p5"))
    )
    _plain = holdout_runs.filter(pl.col("scheme") == "plain")["test_pr_auc"].to_numpy()
    _strat = holdout_runs.filter(pl.col("scheme") == "stratified")["test_pr_auc"].to_numpy()
    _rng = np.random.default_rng(0)
    _ratios = [
        _strat[_rng.integers(0, len(_strat), len(_strat))].std(ddof=1)
        / _plain[_rng.integers(0, len(_plain), len(_plain))].std(ddof=1)
        for _ in range(2000)
    ]
    sd_ratio = float(_strat.std(ddof=1) / _plain.std(ddof=1))
    sd_ratio_lo, sd_ratio_hi = (float(x) for x in np.quantile(_ratios, [0.025, 0.975]))
    strat_pos = holdout_runs.filter(pl.col("scheme") == "stratified")["test_positives"]
    stratified_wins = bool(strat_pos.max() - strat_pos.min() <= 1 and sd_ratio <= 1.10)
    gain_in_spread = sd_ratio <= 0.90
    return (
        expected_positives,
        gain_in_spread,
        sd_ratio,
        sd_ratio_hi,
        sd_ratio_lo,
        spread,
        stratified_wins,
    )


@app.cell
def _(
    N_HOLDOUT_SEEDS,
    expected_positives,
    gain_in_spread,
    markdown_table,
    mo,
    sd_ratio,
    sd_ratio_hi,
    sd_ratio_lo,
    spread,
    stratified_wins,
):
    mo.md(f"""
    ### 1 · Result ({N_HOLDOUT_SEEDS} seeds per scheme)

    Expected test positives: **{expected_positives:.1f}**. `off_by_10pct` is the share of seeds whose
    test set has more than 10 % fewer or more positives than expected.

    {markdown_table(spread)}

    SD ratio, test PR-AUC (stratified / plain): **{sd_ratio:.2f}**
    (bootstrap 95 % interval {sd_ratio_lo:.2f}–{sd_ratio_hi:.2f}).

    **Verdict by the rule: {"stratified kept" if stratified_wins else "PLAIN WINS"}.**
    Stratification {"also reduces" if gain_in_spread else "does not measurably reduce"} the seed-to-seed
    spread of test PR-AUC by ≥ 10 %; its guaranteed benefit is the test-positive count.
    """)
    return


@app.cell
def _(alt, holdout_runs, mo):
    positives_chart = (
        alt.Chart(holdout_runs)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "test_positives:Q",
                bin=alt.Bin(step=2),
                title="Test positives (80/20 split)",
            ),
            y=alt.Y("count()", stack=None, title="Seeds"),
            color=alt.Color("scheme:N", title=None),
        )
        .properties(width=330, height=200, title="Test positives per seed")
    )
    pr_auc_chart = (
        alt.Chart(holdout_runs)
        .mark_boxplot(extent="min-max")
        .encode(
            y=alt.Y("scheme:N", title=None),
            x=alt.X(
                "test_pr_auc:Q",
                title="Test PR-AUC (logistic probe)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("scheme:N", legend=None),
        )
        .properties(width=330, height=120, title="Test PR-AUC per seed")
    )
    mo.hstack(
        [
            mo.ui.altair_chart(positives_chart),
            mo.ui.altair_chart(pr_auc_chart),
        ],
        justify="start",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Single 80/20 split vs repeated stratified 5×5 CV (on the train partition)

    **Question.** For *comparing models and choosing a threshold* inside the train partition, does
    repeated CV give a more stable answer than a single 80/20 validation split?

    **Method.** Only the train partition of §0 is used. Each *replicate* re-runs the whole
    selection procedure with a fresh seed, in two ways:

    - **Single split:** one stratified 80/20 split of the train partition, fit on 80 %, score on 20 %.
    - **Repeated CV:** repeated stratified 5×5 CV (25 folds); mean fold PR-AUC; threshold from the
      out-of-fold predictions of the 5 repeats.

    Each procedure ranks the **8 close-call logistic candidates** (C ∈ {0.01, 0.1, 1, 10}, plain or
    class-balanced) by PR-AUC and picks a cost-minimising threshold for the plain `C=1` probe at
    FN:FP = 5:1, 10:1 and 20:1 (a placeholder; ADR 0007 decides the real rule). The same data is
    used every time, so any disagreement between replicates is **procedure noise**: what a different
    seed alone would change. (Which model is "best" is not the point.)

    **Stability measures.** Ranking: mean pairwise Spearman ρ between replicates' rankings
    (disagreement = 1 − ρ), and the share of replicates that agree on the modal winner. Threshold:
    SD across replicates.

    **Decision rule (stated before the numbers).** Repeated CV is adopted only if it cuts **both**
    ranking disagreement (1 − ρ) **and** the 10:1 threshold SD by **≥ 25 %** relative to the single
    split. If either gain is smaller, the single split wins: it costs 25× fewer fits and is as good
    as the evidence can show.
    """)
    return


@app.cell
def _(
    CANDIDATES,
    COST_RATIOS,
    ENGINEERED_FEATURES,
    LogisticRegression,
    N_SELECTION_REPLICATES,
    PROBE_INDEX,
    build_feature_transformer,
    cost_curve,
    derived_seeds,
    np,
    pick_threshold,
    pl,
    prauc,
    repeated_stratified_cv,
    run_parallel,
    schema,
    stratified_holdout,
    time,
    train_df,
):
    selection_seeds = derived_seeds(2, N_SELECTION_REPLICATES)
    X_dev = train_df.select(schema.FEATURE_COLUMNS)
    y_dev = train_df[schema.TARGET].to_numpy()

    def predict_candidates(train_idx, eval_idx):
        """Fit the transformer once on the train rows, then every candidate; predict the eval rows."""
        transformer = build_feature_transformer(ENGINEERED_FEATURES).fit(X_dev[train_idx])
        z_train = transformer.transform(X_dev[train_idx]).to_numpy()
        z_eval = transformer.transform(X_dev[eval_idx]).to_numpy()
        return np.array(
            [
                LogisticRegression(max_iter=1000, **kwargs)
                .fit(z_train, y_dev[train_idx])
                .predict_proba(z_eval)[:, 1]
                for kwargs in CANDIDATES.values()
            ]
        )

    def single_split_replicate(s):
        """One stratified 80/20 split of the train partition: scores and thresholds."""
        started = time.perf_counter()
        positions = np.arange(train_df.height)
        inner_train, inner_val = stratified_holdout(
            train_df.with_columns(position=pl.Series(positions)), test_size=0.2, seed=s
        )
        val_idx = inner_val["position"].to_numpy()
        preds = predict_candidates(inner_train["position"].to_numpy(), val_idx)
        scores = [prauc(y_dev[val_idx], p) for p in preds]
        thresholds = {
            r: pick_threshold(cost_curve(y_dev[val_idx], preds[PROBE_INDEX], r))
            for r in COST_RATIOS
        }
        return "single 80/20", s, scores, thresholds, time.perf_counter() - started

    def repeated_cv_replicate(s):
        """Repeated stratified 5x5 CV on the train partition: scores and thresholds."""
        started = time.perf_counter()
        n_splits, n_repeats = 5, 5
        cv = repeated_stratified_cv(n_splits=n_splits, n_repeats=n_repeats, seed=s)
        oof = np.zeros((n_repeats, len(CANDIDATES), train_df.height))
        fold_scores = []
        for k, (train_idx, test_idx) in enumerate(cv.split(X_dev, y_dev)):
            preds = predict_candidates(train_idx, test_idx)
            oof[k // n_splits][:, test_idx] = preds
            fold_scores.append([prauc(y_dev[test_idx], p) for p in preds])
        scores = np.mean(fold_scores, axis=0).tolist()
        thresholds = {
            r: pick_threshold(
                np.mean(
                    [cost_curve(y_dev, oof[i][PROBE_INDEX], r) for i in range(n_repeats)], axis=0
                )
            )
            for r in COST_RATIOS
        }
        return "repeated 5x5 CV", s, scores, thresholds, time.perf_counter() - started

    selection_runs = run_parallel(
        lambda kind, s: single_split_replicate(s) if kind == "single" else repeated_cv_replicate(s),
        [(kind, s) for kind in ("single", "cv") for s in selection_seeds],
    )
    return (selection_runs,)


@app.cell
def _(CANDIDATE_NAMES, COST_RATIOS, np, pl, selection_runs):
    method_order = ["single 80/20", "repeated 5x5 CV"]
    candidate_scores = pl.DataFrame(
        [
            (method, s, name, score)
            for method, s, scores, _thresholds, _seconds in selection_runs
            for name, score in zip(CANDIDATE_NAMES, scores, strict=True)
        ],
        schema=["method", "replicate_seed", "candidate", "pr_auc"],
        orient="row",
    )
    selected_thresholds = pl.DataFrame(
        [
            (method, s, ratio, threshold)
            for method, s, _scores, thresholds, _seconds in selection_runs
            for ratio, threshold in thresholds.items()
        ],
        schema=["method", "replicate_seed", "cost_ratio", "threshold"],
        orient="row",
    )

    def rank_disagreement(score_matrix):
        """1 minus the mean pairwise Spearman rho between the rankings of different replicates."""
        ranks = (-score_matrix).argsort(axis=1).argsort(axis=1)
        rho = np.corrcoef(ranks)
        upper = rho[np.triu_indices_from(rho, k=1)]
        return float(1 - upper.mean())

    stability_rows = []
    for _method in method_order:
        _runs = [run for run in selection_runs if run[0] == _method]
        _matrix = np.array([run[2] for run in _runs])
        _winners = np.bincount(_matrix.argmax(axis=1), minlength=len(CANDIDATE_NAMES))
        _row = {
            "method": _method,
            "rank_disagreement": rank_disagreement(_matrix),
            "modal_winner": CANDIDATE_NAMES[int(_winners.argmax())],
            "modal_winner_share": float(_winners.max() / len(_runs)),
            "distinct_winners": int((_winners > 0).sum()),
            "candidate_pr_auc_sd": float(_matrix.std(axis=0, ddof=1).mean()),
        }
        for _ratio in COST_RATIOS:
            _t = np.array([run[3][_ratio] for run in _runs])
            _row[f"threshold_sd_{_ratio}to1"] = float(_t.std(ddof=1))
        _row["fits_per_replicate"] = 8 if _method == "single 80/20" else 8 * 25
        _row["seconds_per_replicate"] = float(np.mean([run[4] for run in _runs]))
        stability_rows.append(_row)
    stability = pl.DataFrame(stability_rows)

    single, cv_row = (stability.row(i, named=True) for i in range(2))
    ranking_gain = 1 - cv_row["rank_disagreement"] / single["rank_disagreement"]
    threshold_gain = 1 - cv_row["threshold_sd_10to1"] / single["threshold_sd_10to1"]
    repeated_cv_wins = bool(ranking_gain >= 0.25 and threshold_gain >= 0.25)
    return (
        candidate_scores,
        ranking_gain,
        repeated_cv_wins,
        selected_thresholds,
        stability,
        threshold_gain,
    )


@app.cell
def _(
    N_SELECTION_REPLICATES,
    candidate_scores,
    markdown_table,
    mo,
    pl,
    ranking_gain,
    repeated_cv_wins,
    stability,
    threshold_gain,
):
    by_candidate = (
        candidate_scores.group_by("method", "candidate", maintain_order=True)
        .agg(mean=pl.col("pr_auc").mean(), sd=pl.col("pr_auc").std())
        .with_columns(cell=pl.format("{} ± {}", pl.col("mean").round(3), pl.col("sd").round(3)))
        .pivot(on="method", index="candidate", values="cell")
    )
    mo.md(f"""
    ### 2 · Result ({N_SELECTION_REPLICATES} replicates per method)

    {markdown_table(stability)}

    PR-AUC per candidate across replicates (mean ± SD). A single split's SD is the noise a
    comparison between close candidates has to beat.

    {markdown_table(by_candidate)}

    Relative gain of repeated CV over the single split at 10:1: ranking disagreement
    **{ranking_gain:.0%}** lower, threshold SD **{threshold_gain:.0%}** lower (rule: both ≥ 25 %).

    **Verdict by the rule: {"repeated 5×5 CV adopted" if repeated_cv_wins else "SINGLE SPLIT WINS"}.**
    """)
    return


@app.cell
def _(alt, mo, selected_thresholds):
    threshold_chart = (
        alt.Chart(selected_thresholds)
        .mark_circle(size=45, opacity=0.6)
        .encode(
            x=alt.X("threshold:Q", title="Selected threshold"),
            y=alt.Y("method:N", title=None),
            color=alt.Color("method:N", legend=None),
            yOffset="jitter:Q",
        )
        .transform_calculate(jitter="random()")
        .properties(width=360, height=110)
        .facet(row=alt.Row("cost_ratio:O", title="FN:FP cost ratio"))
        .properties(title="Selected threshold per replicate")
    )
    mo.ui.altair_chart(threshold_chart)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · 80/20 vs 70/30 vs 90/10

    **Question.** How much does the split ratio change what the test set can tell us, and what does
    it cost the training side?

    **Method.** About 60 seeds per ratio. Each seed: stratified split of the full data; fit the probe
    (logistic regression, plus `HistGradientBoostingClassifier` as a steeper-learning-curve
    sensitivity probe) on the train part. Record the **test positives**, the test PR-AUC with a
    **1000-resample bootstrap 95 % CI**, and the width of that CI. For the logistic probe also run
    repeated 5×5 CV on the *train* part and record the **between-fold SD of PR-AUC**, which is the
    noise of the selection side: a smaller train partition makes it larger. The mean test PR-AUC
    by ratio shows the **train-size penalty** (the same probe, less data).
    These test sets are throwaway draws to measure uncertainty; nothing is selected from them.

    **Decision rule (stated before the numbers).** 80/20 stays unless another ratio clears **all**
    of the following for **both** probes, against 80/20:

    - **70/30 replaces 80/20** if the mean bootstrap CI width is **≥ 15 % narrower**, the mean test
      PR-AUC drops by **≤ 0.01** (train-size penalty), and the logistic probe's between-fold
      PR-AUC SD rises by **≤ 15 %** (selection side not much noisier).
    - **90/10 replaces 80/20** if its CI width is **≤ 15 % wider** *and* its mean test PR-AUC is
      **≥ 0.01 higher** (the extra training data clearly pays).

    The 15 % and 0.01 margins are judgement calls, stated here so they can be argued with.
    """)
    return


@app.cell
def _(
    BOOSTED,
    ENGINEERED_FEATURES,
    LOGISTIC,
    LogisticRegression,
    N_BOOTSTRAP,
    N_RATIO_SEEDS,
    build_feature_transformer,
    derived_seeds,
    df,
    make_probe,
    np,
    pl,
    prauc,
    repeated_stratified_cv,
    run_parallel,
    schema,
    stratified_holdout,
):
    ratio_seeds = derived_seeds(3, N_RATIO_SEEDS)
    ratio_splits = {"90/10": 0.10, "80/20": 0.20, "70/30": 0.30}

    def fold_sd_of_cv(train, s):
        """Between-fold SD of PR-AUC, logistic probe, repeated 5x5 CV on the train part."""
        x, y = train.select(schema.FEATURE_COLUMNS), train[schema.TARGET].to_numpy()
        scores = []
        for train_idx, test_idx in repeated_stratified_cv(n_splits=5, n_repeats=5, seed=s).split(
            x, y
        ):
            transformer = build_feature_transformer(ENGINEERED_FEATURES).fit(x[train_idx])
            model = LogisticRegression(max_iter=1000).fit(
                transformer.transform(x[train_idx]).to_numpy(), y[train_idx]
            )
            scores.append(
                prauc(
                    y[test_idx],
                    model.predict_proba(transformer.transform(x[test_idx]).to_numpy())[:, 1],
                )
            )
        return float(np.std(scores, ddof=1))

    def ratio_run(split, probe, s):
        """One stratified split: probe test PR-AUC, its bootstrap 95% CI, test positives."""
        train, test = stratified_holdout(df, test_size=ratio_splits[split], seed=s)
        model = make_probe(probe).fit(
            train.select(schema.FEATURE_COLUMNS), train[schema.TARGET].to_numpy()
        )
        y_test = test[schema.TARGET].to_numpy()
        p = model.predict_proba(test.select(schema.FEATURE_COLUMNS))[:, 1]
        rng = np.random.default_rng(s)
        resamples = rng.integers(0, len(y_test), (N_BOOTSTRAP, len(y_test)))
        boot = [prauc(y_test[i], p[i]) for i in resamples]
        lo, hi = np.quantile(boot, [0.025, 0.975])
        fold_sd = fold_sd_of_cv(train, s) if probe == LOGISTIC else None
        return (
            split,
            probe,
            s,
            int(y_test.sum()),
            int(train[schema.TARGET].sum()),
            prauc(y_test, p),
            float(hi - lo),
            fold_sd,
        )

    ratio_runs = pl.DataFrame(
        run_parallel(
            ratio_run,
            [
                (split, probe, s)
                for split in ratio_splits
                for probe in (LOGISTIC, BOOSTED)
                for s in ratio_seeds
            ],
        ),
        schema=[
            "split",
            "probe",
            "seed",
            "test_positives",
            "train_positives",
            "test_pr_auc",
            "ci_width",
            "cv_fold_sd",
        ],
        orient="row",
    )
    return ratio_runs, ratio_splits


@app.cell
def _(BOOSTED, LOGISTIC, pl, ratio_runs, ratio_splits):
    by_ratio = (
        ratio_runs.group_by("split", "probe", maintain_order=True)
        .agg(
            test_positives=pl.col("test_positives").mean(),
            train_positives=pl.col("train_positives").mean(),
            test_pr_auc=pl.col("test_pr_auc").mean(),
            test_pr_auc_sd=pl.col("test_pr_auc").std(),
            ci_width=pl.col("ci_width").mean(),
            cv_fold_sd=pl.col("cv_fold_sd").mean(),
        )
        .sort(
            pl.col("probe").replace_strict({LOGISTIC: 0, BOOSTED: 1}),
            pl.col("split").replace_strict({s: i for i, s in enumerate(ratio_splits)}),
        )
    )

    def compare_to_default(frame, probe, split):
        """Relative CI width, PR-AUC difference and fold-SD ratio of `split` vs 80/20."""
        row = frame.filter((pl.col("probe") == probe) & (pl.col("split") == split)).row(
            0, named=True
        )
        base = frame.filter((pl.col("probe") == probe) & (pl.col("split") == "80/20")).row(
            0, named=True
        )
        return {
            "width_change": row["ci_width"] / base["ci_width"] - 1,
            "pr_auc_change": row["test_pr_auc"] - base["test_pr_auc"],
            "fold_sd_change": (
                None if row["cv_fold_sd"] is None else row["cv_fold_sd"] / base["cv_fold_sd"] - 1
            ),
        }

    checks = {
        split: {probe: compare_to_default(by_ratio, probe, split) for probe in (LOGISTIC, BOOSTED)}
        for split in ("70/30", "90/10")
    }
    narrower_70 = all(checks["70/30"][p]["width_change"] <= -0.15 for p in checks["70/30"])
    penalty_ok_70 = all(checks["70/30"][p]["pr_auc_change"] >= -0.01 for p in checks["70/30"])
    selection_ok_70 = checks["70/30"][LOGISTIC]["fold_sd_change"] <= 0.15
    seventy_wins = bool(narrower_70 and penalty_ok_70 and selection_ok_70)
    ninety_wins = bool(
        all(
            checks["90/10"][p]["width_change"] <= 0.15
            and checks["90/10"][p]["pr_auc_change"] >= 0.01
            for p in checks["90/10"]
        )
    )
    chosen_ratio = "70/30" if seventy_wins else "90/10" if ninety_wins else "80/20"
    return (
        by_ratio,
        chosen_ratio,
        narrower_70,
        ninety_wins,
        penalty_ok_70,
        selection_ok_70,
        seventy_wins,
    )


@app.cell
def _(
    BOOSTED,
    LOGISTIC,
    by_ratio,
    chosen_ratio,
    markdown_table,
    mo,
    narrower_70,
    ninety_wins,
    penalty_ok_70,
    pl,
    selection_ok_70,
    seventy_wins,
):
    # Per-probe relative widths: the baseline differs between probes.
    relative = by_ratio.with_columns(
        ci_width_vs_80_20=pl.col("ci_width")
        / pl.col("ci_width").filter(pl.col("split") == "80/20").first().over("probe")
        - 1,
        test_pr_auc_vs_80_20=pl.col("test_pr_auc")
        - pl.col("test_pr_auc").filter(pl.col("split") == "80/20").first().over("probe"),
    )
    mo.md(f"""
    ### 3 · Result

    `ci_width` is the mean width of the 95 % bootstrap CI of test PR-AUC; `test_pr_auc_sd` the
    seed-to-seed SD of the point estimate; `cv_fold_sd` (logistic only) the between-fold PR-AUC SD
    of repeated 5×5 CV on the train part.

    {markdown_table(relative)}

    70/30 against the rule: CI ≥ 15 % narrower on both probes: **{narrower_70}**;
    train-size penalty ≤ 0.01 on both probes: **{penalty_ok_70}**; selection side ≤ 15 % noisier
    (`{LOGISTIC}`): **{selection_ok_70}**. → **{"70/30 replaces 80/20" if seventy_wins else "70/30 does not replace 80/20"}**.
    90/10 against the rule: **{"replaces 80/20" if ninety_wins else "does not replace 80/20"}**
    (both probes, `{LOGISTIC}` and `{BOOSTED}`).

    **Verdict by the rule: {chosen_ratio}.**

    *Reading the CI column:* a CI's width scales as 1/√(test positives), so 70/30 is expected to be
    about 1 − √(68/102) = 18 % narrower and 90/10 about 41 % wider **by arithmetic alone**. The
    measured widths are close to that. The CI gain is therefore guaranteed, not a finding; the empirical
    question is whether the train-size penalty and the noisier selection side are small enough
    to pay for it.
    """)
    return


@app.cell
def _(alt, mo, ratio_runs):
    width_chart = (
        alt.Chart(ratio_runs)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("split:N", title=None, sort=["90/10", "80/20", "70/30"]),
            y=alt.Y("ci_width:Q", title="Width of 95% bootstrap CI (test PR-AUC)"),
            color=alt.Color("split:N", legend=None),
        )
        .properties(width=170, height=200)
        .facet(column=alt.Column("probe:N", title=None))
        .properties(title="Test-metric uncertainty by split ratio")
    )
    mo.ui.altair_chart(width_chart)
    return


@app.cell
def _(
    LOGISTIC,
    by_ratio,
    chosen_ratio,
    markdown_table,
    mo,
    pl,
    repeated_cv_wins,
    sd_ratio,
    stability,
    stratified_wins,
):
    ratio_logistic = by_ratio.filter(pl.col("probe") == LOGISTIC)
    row80 = ratio_logistic.filter(pl.col("split") == "80/20").row(0, named=True)
    cv_row_final = stability.filter(pl.col("method") == "repeated 5x5 CV").row(0, named=True)
    single_row_final = stability.filter(pl.col("method") == "single 80/20").row(0, named=True)
    tradeoff = pl.DataFrame(
        [
            (
                "Holdout: stratified vs plain",
                f"Stratified: test positives fixed; PR-AUC spread SD ratio {sd_ratio:.2f} vs plain",
                "Plain: no stratification code; varying positives",
                "stratified" if stratified_wins else "plain",
            ),
            (
                "Selection: repeated 5x5 CV vs single split",
                f"CV: rank disagreement {cv_row_final['rank_disagreement']:.2f} vs "
                f"{single_row_final['rank_disagreement']:.2f}; 10:1 threshold SD "
                f"{cv_row_final['threshold_sd_10to1']:.3f} vs {single_row_final['threshold_sd_10to1']:.3f}",
                f"Single: {single_row_final['fits_per_replicate']} vs "
                f"{cv_row_final['fits_per_replicate']} fits per candidate set",
                "repeated 5x5 CV" if repeated_cv_wins else "single split",
            ),
            (
                "Ratio: 70/30 vs 80/20 vs 90/10",
                "Larger test set: narrower test CI",
                "Smaller train set: train-size penalty, noisier selection",
                chosen_ratio,
            ),
        ],
        schema=["Decision", "Evidence for the option", "Cost", "Winner by the rule"],
        orient="row",
    )
    summary_text = (
        f"stratified {chosen_ratio} holdout"
        f" + {'repeated stratified 5×5 CV' if repeated_cv_wins else 'a single stratified 80/20 validation split'}"
        " on the train partition"
    )
    mo.md(f"""
    ## 4 · Trade-offs and recommendation

    {markdown_table(tradeoff)}

    ### Recommended for ADR 0005 (follows the rules above)

    **{summary_text}.** The default (stratified 80/20 + repeated 5×5 CV) is {
        "confirmed"
        if (stratified_wins and repeated_cv_wins and chosen_ratio == "80/20")
        else "overturned in at least one part"
    } by the evidence.

    - Test positives at 80/20: **{row80["test_positives"]:.0f}** of {
        row80["test_positives"] + row80["train_positives"]:.0f}
      failures, so every test metric needs a confidence interval (width ≈ {
        row80["ci_width"]:.2f} PR-AUC
      for the logistic probe).
    - The holdout is stratified, split before fitting anything, and scored **once**, at the end.
    - Model comparison and threshold selection use the train partition only.

    ### Limits of this evidence

    - The cost-minimising threshold and its 5:1 / 10:1 / 20:1 ratios are a **probe**, not the ADR 0007 decision.
    - Candidates in comparison 2 are one family on purpose; real candidate gaps (notebook 03) may be
      larger or smaller than the noise measured here.
    - Repeated-CV folds overlap, so SEs from them are optimistic (as in ADR 0004).
    - The 15 % CI margin sits close to the 18 % that 70/30 gets from arithmetic alone, so the ratio
      verdict really rests on the penalty criteria and on how much a smaller train partition matters.
      A reviewer who weighs that differently can reasonably keep 80/20.
    - The learning-curve sensitivity uses two default models. A different final model could weigh the
      train-size penalty differently; ADR 0006 may revisit the ratio.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Consequence check: ADR 0004 feature deltas at 70/30

    ADR 0004's CV figures were computed on the provisional **80/20** train partition. ADR 0005 moves
    the holdout to **70/30**, so this section recomputes ADR 0004's feature-set comparison on the new
    train partition. It does not reopen ADR 0004.

    **Method.** The split settings come from `configs/default.toml` (`[split]`), via the package
    splitter. Train partition only; the holdout is not scored. The same 5×5 stratified CV folds are
    reused for every feature set and both models, so deltas are paired on identical folds. Models are
    untuned defaults, as in ADR 0004.

    **Decision rule (stated before the numbers).** ADR 0004's amended keep-rule, unchanged: an
    engineered feature stays if its mean paired Δ PR-AUC over raw is **> 0 for the non-linear model**.
    If any of the three kept features fails that rule at 70/30, ADR 0004 must be revisited (by a
    superseding ADR, since it is Accepted). Otherwise its conclusion stands.
    """)
    return


@app.cell
def _(
    ENGINEERED_FEATURES,
    BOOSTED,
    HistGradientBoostingClassifier,
    LOGISTIC,
    LogisticRegression,
    build_feature_transformer,
    df,
    load_config,
    make_pipeline,
    np,
    pl,
    prauc,
    repeated_stratified_cv,
    run_parallel,
    schema,
    seed,
    stratified_holdout,
):
    split_settings = load_config().split
    recheck_train, _unused_holdout = stratified_holdout(
        df, test_size=split_settings.test_size, seed=seed
    )
    recheck_x = recheck_train.select(schema.FEATURE_COLUMNS)
    recheck_y = recheck_train[schema.TARGET].to_numpy()
    recheck_folds = list(
        repeated_stratified_cv(
            n_splits=split_settings.cv_folds, n_repeats=split_settings.cv_repeats, seed=seed
        ).split(recheck_x, recheck_y)
    )
    recheck_sets = {
        "raw": (),
        "+ Power": (ENGINEERED_FEATURES[0],),
        "+ Temperature delta": (ENGINEERED_FEATURES[1],),
        "+ Wear x torque": (ENGINEERED_FEATURES[2],),
        "+ all three": ENGINEERED_FEATURES,
    }

    def recheck_fold(model_name, set_name, fold):
        """PR-AUC of one model on one feature set for one CV fold (transformer fitted in-fold)."""
        train_idx, test_idx = recheck_folds[fold]
        model = (
            LogisticRegression(max_iter=1000)
            if model_name == LOGISTIC
            else HistGradientBoostingClassifier(random_state=seed)
        )
        pipeline = make_pipeline(build_feature_transformer(recheck_sets[set_name]), model)
        pipeline.fit(recheck_x[train_idx], recheck_y[train_idx])
        return prauc(recheck_y[test_idx], pipeline.predict_proba(recheck_x[test_idx])[:, 1])

    recheck_tasks = [
        (model_name, set_name, fold)
        for model_name in (LOGISTIC, BOOSTED)
        for set_name in recheck_sets
        for fold in range(len(recheck_folds))
    ]
    recheck_scores = pl.DataFrame(
        [
            (*task, score)
            for task, score in zip(
                recheck_tasks, run_parallel(recheck_fold, recheck_tasks), strict=True
            )
        ],
        schema=["model", "feature_set", "fold", "pr_auc"],
        orient="row",
    )
    recheck_raw = recheck_scores.filter(pl.col("feature_set") == "raw").select(
        "model", "fold", raw_pr_auc="pr_auc"
    )
    recheck_summary = (
        recheck_scores.join(recheck_raw, on=["model", "fold"])
        .with_columns(delta=pl.col("pr_auc") - pl.col("raw_pr_auc"))
        .group_by("model", "feature_set", maintain_order=True)
        .agg(
            pr_auc=pl.col("pr_auc").mean(),
            delta=pl.col("delta").mean(),
            delta_se=pl.col("delta").std() / np.sqrt(pl.len()),
        )
    )
    return recheck_sets, recheck_summary, recheck_train, split_settings


@app.cell
def _(
    BOOSTED,
    LOGISTIC,
    markdown_table,
    mo,
    pl,
    recheck_sets,
    recheck_summary,
    recheck_train,
    schema,
    split_settings,
):
    # Deltas recorded in ADR 0004 on the provisional 80/20 train partition.
    adr_0004_delta = {
        (LOGISTIC, "+ Power"): 0.036,
        (LOGISTIC, "+ Temperature delta"): -0.001,
        (LOGISTIC, "+ Wear x torque"): -0.000,
        (LOGISTIC, "+ all three"): 0.035,
        (BOOSTED, "+ Power"): 0.034,
        (BOOSTED, "+ Temperature delta"): 0.030,
        (BOOSTED, "+ Wear x torque"): 0.019,
        (BOOSTED, "+ all three"): 0.077,
    }
    recheck_table = recheck_summary.with_columns(
        adr_0004_delta_80_20=pl.struct("model", "feature_set").map_elements(
            lambda row: adr_0004_delta.get((row["model"], row["feature_set"])),
            return_dtype=pl.Float64,
        ),
        pr_auc=pl.col("pr_auc").round(3),
        delta=pl.col("delta").round(3),
        delta_se=pl.col("delta_se").round(4),
    ).rename({"delta": "delta_70_30"})
    kept_still_pass = all(
        recheck_summary.filter((pl.col("model") == BOOSTED) & (pl.col("feature_set") == label))[
            "delta"
        ].item()
        > 0
        for label in recheck_sets
        if label not in ("raw", "+ all three")
    )
    mo.md(f"""
    ### 5 · Result

    Train partition at `test_size = {split_settings.test_size}`: **{recheck_train.height:,} Readings,
    {int(recheck_train[schema.TARGET].sum())} failures**. `delta_70_30` is the mean paired Δ PR-AUC over
    raw on identical folds (± `delta_se`, optimistic because repeated-CV folds overlap);
    `adr_0004_delta_80_20` is the value recorded in ADR 0004.

    {markdown_table(recheck_table)}

    **Verdict by the rule: {"every kept feature still improves the non-linear model; ADR 0004's conclusion stands" if kept_still_pass else "A KEPT FEATURE NO LONGER IMPROVES THE NON-LINEAR MODEL; ADR 0004 must be revisited"}.**
    The linear model's pattern is unchanged: only Power helps it, as ADR 0004 already notes.
    """)
    return


if __name__ == "__main__":
    app.run()
