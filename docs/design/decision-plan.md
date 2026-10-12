# Decision plan (grill session outcome)

Date: 2026-10-12. Source of requirements: [`../../TECHNICAL_SPEC.md`](../../TECHNICAL_SPEC.md).
Vocabulary: [`../../GLOSSARY.md`](../../GLOSSARY.md). This file is a plan, not an ADR: nothing here is `Accepted`.
Each decision below becomes a `Proposed` ADR backed by a notebook; a human moves it to `Accepted`.

## Facts established

- `data/raw/ai4i2020.csv`: 10,000 rows, 14 columns, **0 nulls**.
- Failures: 339 (3.39%). Mode flags: TWF 46, HDF 115, PWF 95, OSF 98, RNF 19.
- 9 failures have no mode flag; 18 rows have a mode flag but no failure.
- 10,000 unique `Product ID`s: no repeated machines, so no group leakage and no group-aware split.
- Rows have no time ordering, so a temporal split is not meaningful.
- Missing data therefore means inference-time (`Incomplete reading`) and simulated missingness, not training-time cleaning.

## Decisions

| Area | Decision |
|---|---|
| Target and features | Binary `Machine failure`. Excluded: `TWF, HDF, PWF, OSF, RNF`, `UDI`, `Product ID`. Features: `Type` + 5 numeric sensors. Multi-class failure type: deferred, not disproven (see notebook 03). Excluded `UDI` and `Product ID` because their signal is a row-position artifact (notebook 01 §2b). Engineered physics features (power = torque x speed, temperature delta, wear x torque) compared as a 2x2: {linear, non-linear} x {raw, engineered}. |
| Split and validation | Stratified 80/20 holdout, touched once at the end. Repeated 5x5 stratified CV on train for all comparisons and threshold selection. Seed from `configs/default.toml` (single source). Notebook compares plain vs stratified over many seeds, single split vs repeated CV, and 80/20 vs 70/30 vs 90/10. Evidence wins over this default. |
| Metric and threshold | Select models on PR-AUC. Supporting: ROC-AUC, recall, precision, Brier. Operating threshold by minimizing expected cost at FN:FP = 5:1, 10:1, 20:1 (shown as a table). Default ratio 10:1, an explicit assumption, configurable in `configs/default.toml`. Bootstrap CIs on test metrics (about 68 test positives). |
| Model family | Logistic regression (L2, weighted vs unweighted) vs `HistGradientBoostingClassifier` and `RandomForestClassifier`. Rule: choose the simpler model unless the non-linear gain in repeated-CV PR-AUC exceeds max(0.02 absolute, 1 SE across folds). Per-reading explainability is a trade-off criterion; no `shap`. Calibration checked here. |
| Feature rule | Keep an engineered feature if it improves repeated-CV PR-AUC (mean paired delta > 0 on identical folds) for the non-linear model; the linear model is not required to improve. Also keep it if needed for interpretability. Amended 2026-10-12 after notebook 01 results, which showed the original "both models" rule rejects linear combinations of inputs by construction; the original rule required a gain for both linear and non-linear models. |
| Missing data | Train on complete data; inject nulls into validation folds only. Scenarios: MCAR in 1 sensor, MCAR in 2 sensors, MAR (torque missing at high speed), full sensor outage, `Type` missing. Strategies: reject, median/mode imputation, imputation + missing-indicators, native NaN handling (HistGradientBoosting only). Rule: adopt a strategy only if CV PR-AUC drops < 10% relative with 1 sensor missing; otherwise the app refuses to score. The ADR must say how the app signals reduced reliability. |
| App output | Three bands (Low / Elevated / High) plus the estimated probability and a caveat. Band edges from the cost analysis. Wording: "estimated risk based on historical readings", never "will fail". Calibrate only if notebook 03 shows it is needed. |
| Dependencies | Declare `altair` in the dev group (already installed transitively via marimo). No other additions. |
| Code boundary | Minimal tested enabling layer in the package (CSV load and validation, `configs/default.toml`, stratified split, candidate feature transformer), with mirrored tests, per ADR 0002 scope. Experiment-only code stays in the notebooks: model comparison loops, missing-data injection harness, plots. Losing candidates stay notebook-only. |

## Notebooks, in order (each gated on the previous ADR being Accepted)

| Notebook | Concludes ADR |
|---|---|
| `notebooks/01_target_and_features.py` | 0004 target, features, exclusions, engineered features. Includes leakage evidence: mode flags alone vs `Machine failure`, `Product ID` as predictor. |
| `notebooks/02_split_strategy.py` | 0005 split and validation |
| `notebooks/03_model_family.py` | 0006 model family; 0007 metric and threshold. Also a per-mode predictability check (repeated-CV PR-AUC per Failure mode from the sensors, against each mode's prevalence); its result feeds ADR 0009 (whether mode prediction becomes a secondary output). |
| `notebooks/04_missing_data.py` | 0008 missing-data policy (runs on the models shortlisted in 03) |

ADR numbering (renumbers the pending list in 0001, per the follow-up in 0002):
0003 dataset source (use supplied file; no notebook), 0004 target and features, 0005 split and validation,
0006 model family, 0007 metric and threshold, 0008 missing-data policy, 0009 app output, 0010 deployment.
0009 and 0010 have no notebook and come after 0008. 0009 is informed by the per-mode check in notebook 03.

## Workflow rules

- One notebook at a time. Build notebook, write `Proposed` ADR, human reviews and sets `Accepted`, then the next notebook.
- One concern per commit: each notebook and each ADR in its own commit.
- Notebooks import from the package and are never required by the pipeline.
- Do not edit an `Accepted` ADR; supersede it.
- `just check` must pass, with the result shown, before any step is called done.

## Tickets

- Location: `docs/tickets/NNN-short-title.md`.
- Fields: Context (ADR and notebook links), Scope, Acceptance criteria, Dependencies (ticket IDs), Out of scope.
- Each ticket is a vertical slice with tests that passes `just check` on its own.
- Drafted only after the ADRs they depend on are `Accepted`; the list is reviewed with the human before files are written.
- Also ticketed, though no notebook feeds them: deployment (spec §4.3), monitoring and production-readiness critique (§4.4), system-design diagram, README.
