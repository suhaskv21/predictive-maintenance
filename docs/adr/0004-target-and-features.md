# 0004. Target and features

- Status: Accepted
- Date: 2026-10-12
- Deciders: Suhas (accepted 2026-10-12)

## Context
The spec (§3, §8) requires an explicit target, input features, excluded columns, and a
leakage-safe treatment of the failure-mode indicators and identifiers. Evidence is in
[`notebooks/01_target_and_features.py`](../../notebooks/01_target_and_features.py); the plan is
[`decision-plan.md`](../design/decision-plan.md). Vocabulary follows [`GLOSSARY.md`](../../GLOSSARY.md).

Facts (10,000 Readings, 0 nulls): 339 Machine failures (3.39 %). 9 failures carry no Failure-mode
flag and 18 Readings carry a flag without a failure, so the modes annotate the outcome imperfectly.
All figures below come from the training partition of a provisional stratified 80/20 split (seed from
`configs/default.toml`) using repeated 5×5 stratified CV scored by PR-AUC. The holdout was not
evaluated. Models are untuned defaults (logistic regression, `HistGradientBoostingClassifier`); this
ADR compares feature sets, not model families.

## Options considered
**Target**
1. Binary `Machine failure` — matches the spec default and the engineer's question; 339 positives.
2. Multi-class failure mode, or one binary model per mode — says *why*, but RNF has 19 positives,
   modes overlap, and it multiplies thresholds and caveats. **Deferred, not disproven**: whether the
   sensors predict each mode is untested (see Open items).

**Excluded columns**
- Failure modes (`TWF, HDF, PWF, OSF, RNF`): "any flag set" scores PR-AUC 0.93 and a logistic
  regression on the five flags 0.975 (chance 0.034). They are recorded causes of the failure, known
  only after diagnosis.
- `UDI`, `Product ID`: every `Product ID` is unique, so no machine is ever seen twice (no machine
  history, no group leakage). The ID's letter equals `Type`. Both identifiers nonetheless beat a
  random-noise control (PR-AUC 0.085 vs 0.035). The cause is a dataset artifact: rows `UDI`
  4001–5000 fail at 13.1 % (heat-dissipation flag HDF 10.1 %) against 2.3 % elsewhere. A model would
  learn row position, which does not exist for a new Reading and would inflate offline scores.

**Engineered features** (all derivable from the five sensor inputs), 2×2 of {linear, non-linear} ×
{raw, engineered}. Rule (plan, **amended 2026-10-12 by Suhas after these results were seen**): keep an
engineered feature if it improves CV PR-AUC (mean paired Δ > 0) for the **non-linear** model, or is
needed for interpretability; the linear model is not required to improve. The original rule required
a gain for *both* a linear and a non-linear model and would have kept only Power. The amendment
rationale: after standardisation, Temperature delta is an exact linear combination of two inputs, so
"both" rejects it by construction. The original per-model deltas stay in the table below.

| Feature set | Logistic regression PR-AUC | HistGradientBoosting PR-AUC |
|---|---|---|
| Raw (`Type` + 5 sensors) | 0.466 | 0.808 |
| + Power (torque × angular speed) | 0.502 (Δ +0.036) | 0.842 (Δ +0.034) |
| + Temperature delta | 0.464 (Δ −0.001) | 0.837 (Δ +0.030) |
| + Wear × torque | 0.466 (Δ −0.000) | 0.827 (Δ +0.019) |
| + all three | 0.501 (Δ +0.035) | 0.885 (Δ +0.077) |

Δ are paired differences on identical folds; standard errors are about 0.001–0.006 and are
optimistic because repeated-CV folds overlap.

## Decision
Accepted (decisions by Suhas, 2026-10-12):
- **Target:** binary `Machine failure`. Multi-class failure mode is **deferred, not disproven**.
- **Excluded:** `TWF, HDF, PWF, OSF, RNF` (outcome fields; leakage, notebook §2a); `UDI`,
  `Product ID` (identifiers: their signal is a row-position artifact, since `UDI` 4001–5000 fail at
  13.1 % vs 2.3 % elsewhere and a new Reading has no such position; notebook §2b).
- **Features:** `Type` + `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`,
  `Torque [Nm]`, `Tool wear [min]`, plus **all three** engineered features: Power (torque × angular
  speed, W), Temperature delta [K], and Wear × torque [min Nm]. Each improves the non-linear model's
  CV PR-AUC (+0.034, +0.030, +0.019); the interpretability arm of the rule is not invoked.
- Engineered features are computed inside the serialized scikit-learn `Pipeline`
  (`predictive_maintenance.preprocessing.features`), never in the app.

## Consequences
- The feature set is small, derivable from five user-entered sensor values plus `Type`, and free of
  outcome columns. The failure-mode exclusion is enforced in code (`data/schema.py`) and tested.
- The rule was relaxed after the evidence was seen. Under the original rule only Power would be kept.
  The all-three set lifts HistGradientBoosting PR-AUC from 0.808 to 0.885.
- **If ADR 0006 selects the linear model, revisit the engineered features.** The linear model gains
  only from Power (+0.036); Temperature delta (−0.001) and Wear × torque (−0.000) add inputs without
  measured benefit there.
- Revisit if the split strategy (ADR 0005) or model family (ADR 0006) changes the comparison, or if
  the dataset is replaced by data where identifiers are meaningful.
- The UDI block artifact suggests the dataset's HDF failures are concentrated; Readings from other
  sources may not follow it. Operational wording in the app must not promise coverage beyond the
  historical data.

## Open items
- **Per-mode predictability (notebook 03, model family).** Measure PR-AUC per Failure mode from the
  sensors (training partition, repeated CV, against each mode's prevalence; RNF has 19 positives and
  will be very noisy). The result decides whether mode prediction becomes a secondary output in
  ADR 0009. Until then multi-class is deferred, not rejected.
