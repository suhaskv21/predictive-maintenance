import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from predictive_maintenance.config import load_config
    from predictive_maintenance.data import schema
    from predictive_maintenance.data.ingest import load_dataset
    from predictive_maintenance.evaluation.compare import cv_folds
    from predictive_maintenance.evaluation.modes import evaluate_modes
    from predictive_maintenance.evaluation.split import (
        repeated_stratified_cv,
        stratified_holdout,
    )
    from predictive_maintenance.training.candidates import build_candidates

    return (
        build_candidates,
        cv_folds,
        evaluate_modes,
        load_config,
        load_dataset,
        mo,
        pl,
        repeated_stratified_cv,
        schema,
        stratified_holdout,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 05 · Which Failure mode? (feeds ADR 0009)

    ADR 0004 and 0006 deferred one question to the app-output ADR: can the six inputs predict *which*
    **Failure mode** (TWF, HDF, PWF, OSF, RNF) occurs, well enough to justify showing it beside the
    Flag? A mode is never an input to the `Machine failure` model; here each mode is itself the
    target, and **only the six inputs are features**.

    **Setup.** Train partition only, the repeated stratified 5×5 CV of ADR 0005, identical folds for
    every mode, preprocessing fitted inside each fold. **The holdout is discarded at load time and
    never scored.** One binary model per mode: the default random forest (unweighted, 300 trees) with
    ADR 0004's three engineered features (chosen for `Machine failure`, not re-tested per mode).
    Each CV repeat pools its test folds into one out-of-fold score per Reading, because a rare mode is
    absent from many single folds; PR-AUC is computed per repeat.

    **Rule, written before the numbers** (margins in `configs/default.toml`, `[failure_modes]`). A
    mode is *predictable enough to justify a follow-up ADR* only if all three hold:

    1. mean PR-AUC ≥ `min_pr_auc` (0.30);
    2. mean PR-AUC ≥ `min_lift` (5) × the mode's prevalence (the PR-AUC of no skill);
    3. mean PR-AUC − one corrected standard error > the prevalence.

    The corrected SE is the Nadeau–Bengio form on the per-repeat PR-AUC values. It captures how fold
    assignment moves the figure, not the sampling noise of a handful of positives, so for RNF it
    understates the uncertainty.

    **What a pass means.** Nothing ships in ADR 0009 either way: a mode output needs one model per
    mode, its own Operating threshold (ADR 0007's scope), sidecars and a missing-data check. A pass
    only says a follow-up ADR is worth writing.
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
    readings = load_dataset()
    train, _holdout = stratified_holdout(
        readings, test_size=config.split.test_size, seed=config.seed
    )
    del _holdout
    folds = cv_folds(
        train,
        repeated_stratified_cv(
            n_splits=config.split.cv_folds, n_repeats=config.split.cv_repeats, seed=config.seed
        ),
    )
    forest = build_candidates(config.seed)["random_forest_unweighted"]
    counts = ", ".join(f"{m} {int(train[m].sum())}" for m in schema.FAILURE_MODES)
    mo.md(
        f"Seed `{config.seed}`. Train partition: **{train.height:,} Readings**, "
        f"{int(train[schema.TARGET].sum())} Machine failures; mode positives: {counts}. "
        f"{len(folds)} folds ({config.split.cv_folds}×{config.split.cv_repeats})."
    )
    return config, folds, forest, train


@app.cell
def _(config, evaluate_modes, folds, forest, train):
    results = evaluate_modes(
        forest,
        train,
        folds,
        n_splits=config.split.cv_folds,
        min_pr_auc=config.failure_modes.min_pr_auc,
        min_lift=config.failure_modes.min_lift,
    )
    return (results,)


@app.cell
def _(mo, pl, results):
    table = pl.DataFrame(
        [
            {
                "mode": r.mode,
                "positives": r.positives,
                "prevalence": r.prevalence,
                "pr_auc": r.pr_auc,
                "corrected_se": r.corrected_se,
                "lift": r.lift,
                "floor": r.meets_floor,
                "lift_ok": r.meets_lift,
                "beats_prevalence": r.beats_prevalence,
                "predictable": r.predictable,
            }
            for r in results
        ]
    )
    passed = [r.mode for r in results if r.predictable]
    print(table)
    print("predictable:", passed or "none")
    mo.vstack(
        [
            mo.md(
                "### PR-AUC per Failure mode against its prevalence (train-partition CV)\n"
                f"**Predictable under the pre-declared rule: {', '.join(passed) or 'none'}.**"
            ),
            table,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
