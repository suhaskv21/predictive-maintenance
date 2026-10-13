# 0006. Model family

- Status: Accepted
- Date: 2026-10-13
- Deciders: Suhas (rules agreed in a grill session 2026-10-13, amended after review; accepted 2026-10-13); drafted by an agent
- Post-review changes (2026-10-13): (1) boosters kept as switchable models and class weighting extended to every family; (2) HistGradientBoosting dropped from the offered set (see Decision)

## Context
The spec asks for a justified model choice and an interpretability/performance trade-off (§4.1.6, §8).
Fixed by accepted ADRs: target `Machine failure`; `Type` + 5 sensors + 3 engineered features, failure
modes and identifiers excluded ([0004](0004-target-and-features.md)); stratified 70/30 holdout scored
once, and repeated stratified 5×5 CV on the train partition (237 failures) for all comparisons, with
preprocessing fitted in-fold ([0005](0005-split-and-validation.md)). Preprocessing and model ship as
one scikit-learn `Pipeline`. Seeds come from `configs/default.toml` (42). Vocabulary follows
[`GLOSSARY.md`](../../GLOSSARY.md). The threshold and cost metric are **not** decided here (ADR 0007).

Evidence: [`notebooks/03_model_comparison.py`](../../notebooks/03_model_comparison.py), built on
`predictive_maintenance.evaluation.compare` and `predictive_maintenance.training.candidates` (tested).
**The holdout was discarded at load time and never scored**; every number below is train-partition CV.

Disclosures:
- ADR 0004 had already shown logistic ≈ 0.50 and HistGradientBoosting ≈ 0.885 PR-AUC on this feature
  set, so linear vs boosting was not a blind test.
- **Rules 5 and 8 below were amended after the first (unweighted, single-winner) results were seen.**
  The reviewer asked to keep the boosters as switchable candidates and to weight every supervised
  family. The amended rules were still written before the second set of numbers, but the first
  results shaped them.
- All candidates use pre-declared defaults, no tuning. The margins (0.02, 10 %, 0.01, 100 ms,
  50 MB) are judgement calls, written before the numbers so they can be challenged.

## Options considered
Candidates (13): Dummy; Isolation Forest and One-Class SVM (reference); and, each **unweighted and
weighted**, logistic regression, random forest (300 trees), HistGradientBoosting, LightGBM, XGBoost.

Rules, fixed before the numbers they judge:
1. Metric: mean PR-AUC over the 25 folds, paired on identical folds.
2. Default model: candidates are walked simplest first (logistic, random forest, HistGradientBoosting,
   LightGBM, XGBoost; unweighted before weighted within a family). A more complex challenger replaces
   the incumbent only if its mean paired delta exceeds `max(0.02, corrected SE)`. The SE is
   Nadeau–Bengio (`sqrt((1/25 + 1/4) · var(delta))`), because overlapping folds make the naive SE
   too small.
3. Vetoes before the rule: single-Reading `predict_proba` latency ≤ 100 ms, serialized pipeline
   ≤ 50 MB. Fit time and install size are recorded, not capped.
4. Dummy (prior) is a baseline. Isolation Forest and One-Class SVM (fitted on non-failures only,
   `nu` = 0.034) give no probability, so they are reference candidates and not selectable.
5. Class imbalance (amended): every supervised family is run unweighted and with balanced class
   weights (`class_weight="balanced"` natively; XGBoost via balanced sample weights computed inside
   each fit). Within a family, weighting is adopted only if it beats the unweighted variant by the
   rule-2 margin. Resampling (SMOTE, undersampling) is rejected: it needs `imbalanced-learn`, must
   live inside each fold to avoid leakage, and distorts probabilities.
6. Calibrate a model if sigmoid or isotonic cuts its mean CV Brier by ≥ 10 % relative. Isotonic only
   if it also beats sigmoid by 10 % (237 failures is thin for isotonic).
7. Revisit the 70/30 ratio if the default model's learning-curve penalty exceeds 0.01 PR-AUC.
8. Switchable models (amended): the random forest, LightGBM and XGBoost stay in the pipeline as
   selectable models, so the app (ADR 0009) can offer a dropdown once the threshold is concluded.
   Each must pass the vetoes. The default is the rule-2 winner. *(Post-review: this rule first
   listed HistGradientBoosting too; it was removed, see Decision.)*

**Results** (train partition, 7,000 Readings, 25 folds). Fit seconds are per fold. Latency (single
Reading, one thread) and size come from one fit on the whole train partition:

| Candidate | PR-AUC (SD over folds) | Brier | Fit s | Latency ms | Artifact MB |
|---|---|---|---|---|---|
| Random forest, unweighted | **0.894** (0.038) | 0.0075 | 5.7 | 25.7 | 4.6 |
| LightGBM, unweighted | 0.882 (0.042) | 0.0082 | 0.44 | 8.1 | 0.36 |
| Random forest, weighted | 0.880 (0.042) | 0.0103 | 4.4 | 25.6 | 6.1 |
| HistGradientBoosting, unweighted | 0.877 (0.047) | 0.0085 | 1.0 | 11.3 | 0.38 |
| LightGBM, weighted | 0.877 (0.043) | 0.0100 | 0.44 | 7.7 | 0.35 |
| XGBoost, weighted | 0.877 (0.044) | 0.0093 | 0.35 | 7.5 | 0.19 |
| HistGradientBoosting, weighted | 0.875 (0.049) | 0.0094 | 2.3 | 11.3 | 0.38 |
| XGBoost, unweighted | 0.865 (0.042) | 0.0094 | 0.35 | 7.6 | 0.18 |
| Logistic, unweighted | 0.522 (0.074) | 0.0223 | 0.11 | 8.0 | 0.006 |
| Logistic, weighted | 0.477 (0.075) | 0.1052 | 0.13 | 7.8 | 0.006 |
| One-Class SVM (reference) | 0.443 (0.066) | n/a | 0.30 | 7.8 | 0.034 |
| Isolation Forest (reference) | 0.180 (0.041) | n/a | 1.6 | 29.3 | 4.5 |
| Dummy (chance = prevalence) | 0.034 (0.000) | 0.0327 | 0.10 | 7.4 | 0.005 |

All candidates pass both vetoes. Latency depends on machine load: a first run measured the forest at
62.6 ms (63 % of the veto) and this one at 25.7 ms; the boosters were 8–11 ms in both.

Selection walk (challenger minus incumbent; the threshold is 0.020 unless the corrected SE is larger):

| Challenger vs incumbent | Mean delta | Replaces |
|---|---|---|
| Logistic weighted vs unweighted | −0.045 | no |
| **Random forest vs logistic unweighted** (threshold 0.037) | **+0.372** | **yes** |
| Random forest weighted vs unweighted | −0.014 | no |
| HistGradientBoosting unweighted / weighted vs forest | −0.017 / −0.019 | no |
| LightGBM unweighted / weighted vs forest | −0.012 / −0.018 | no |
| XGBoost unweighted / weighted vs forest | −0.030 / −0.018 | no |

**Weighting per family** (weighted minus unweighted): random forest −0.014, HistGradientBoosting
−0.002, LightGBM −0.005, XGBoost **+0.012** (SE 0.011), logistic −0.045. None reaches +0.02, so
unweighted is kept for every family. Weighting also raises the Brier score in every family except
XGBoost (0.0094 → 0.0093), because it inflates probabilities. XGBoost is the only family where
weighting helps on PR-AUC, and it is within noise.

The four tree families are within 0.03 of each other (0.865–0.894), so the forest's lead is inside the
margin for the boosters: the rule keeps it because it is simpler and not worse, not because it
measurably wins.

**Anomaly-detection angle.** Isolation Forest is 0.714 and One-Class SVM 0.451 PR-AUC **below** the
default. Discarding the 237 failure labels costs far more than it gives. They stay reference only.

**Calibration** (relative Brier gain, sigmoid / isotonic; bar 10 %):

| Model (unweighted) | Sigmoid | Isotonic | Decision |
|---|---|---|---|
| Random forest | +6.5 % | +6.8 % | none |
| HistGradientBoosting | −14.7 % | −13.9 % | none |
| LightGBM | +5.3 % | +4.8 % | none |
| XGBoost | +6.1 % | +6.7 % | none |

Out-of-fold reliability is close to the diagonal in the top-probability decile for all four
(predicted vs observed: forest 0.316 vs 0.314, HistGradientBoosting 0.292 vs 0.310, LightGBM
0.293 vs 0.311, XGBoost 0.306 vs 0.309). Below that the models differ: the boosters predict about
0.0002–0.002 in the second decile and observe about 0.014–0.017, so they under-predict there
(XGBoost 0.0016 vs 0.0171); the forest is closer (0.015 vs 0.011). Most Readings receive a predicted
risk of about zero in every model.

**Learning curve** (60 throwaway splits inside the train partition, paired, training on 87.5 % of the
rows, the 80/20 → 70/30 step): PR-AUC lost by training on less data.

| Model | Penalty | vs 0.01 trigger |
|---|---|---|
| Logistic | −0.001 | below |
| Random forest (default) | 0.003 | below |
| LightGBM | 0.003 | below |
| XGBoost | 0.007 | below |
| HistGradientBoosting | **0.026** (SD 0.015) | **above** |

The trigger applies to the default (rule 7), which passes, so **70/30 stands**.

**Interpretability.** The forest has no per-Reading explanation without `shap` (not adopted). Global
permutation importance (forest, fold 0, PR-AUC drop): Torque 0.80, Rotational speed 0.43, Air
temperature 0.36, Tool wear 0.24, Process temperature 0.17, `Type` 0.09. Shuffling a raw input also
breaks the engineered features built from it, so this ranks inputs, not mechanisms. Logistic
regression is readable, but 0.37 PR-AUC lower, which is too weak to prioritise maintenance.

## Decision
Proposed:
- **Default model: random forest, unweighted** (`RandomForestClassifier`, 300 trees, scikit-learn
  defaults otherwise, `random_state` from the config seed), inside the single serialized `Pipeline`
  after the ADR 0004 feature transformer and an array-conversion step (LightGBM and XGBoost reject
  bracketed column names, so conversion to numpy happens at the scikit-learn boundary).
- **Switchable models:** random forest, LightGBM and XGBoost, all unweighted (each family's weighted
  variant stays in the registry and the comparison, but is not offered, per rule 5). Each is its own
  serialized `Pipeline`. A model offered in the app needs its own threshold from ADR 0007, because
  predicted probabilities differ by model.
- **HistGradientBoosting is not offered** *(post-review change)*. Reason: it is redundant with
  LightGBM (the same histogram gradient-boosting method) with a weaker learning curve (penalty 0.026
  against 0.003) and slower fits (1.0 s against 0.44 s per fold), and its PR-AUC is not better
  (0.877 against 0.882). This is a UI scope decision, **not a model-quality verdict**: it stays a
  compared candidate in the registry and notebook 03, and it is reversible at no cost, by adding it
  back to `SWITCHABLE_FAMILIES`. Offering it would need its own threshold (ADR 0007) and its
  calibration result (sigmoid −14.7 %, isotonic −13.9 %: none applied) is unchanged.
- **Class imbalance:** balanced class weighting was tested on every supervised family (rule 5) and
  not adopted. No weighted variant beat its unweighted counterpart by the 0.02 margin (PR-AUC
  delta: random forest −0.014, HistGradientBoosting −0.002, LightGBM −0.005, XGBoost +0.012 with SE
  0.011, logistic −0.045). In every family except XGBoost, where it was equal (0.0094 → 0.0093),
  weighting also raised the Brier score, and in the reliability tables it over-predicts the top
  decile (forest 0.388 predicted vs 0.316 observed), so it distorts the probability the app would
  show. The offered models are therefore the unweighted variants. The weighted variants stay in the
  registry and notebook 03 so the result can be re-run. Resampling is rejected (rule 5).
- **Calibration:** none for any switchable model; all miss the 10 % bar. The probability may be
  shown as the estimated Failure risk, with the low-decile caveat below.
- **Split ratio:** 70/30 is kept (default's penalty 0.003 ≤ 0.01).
- **Not offered, and why:** HistGradientBoosting (redundant with LightGBM, above), weighted variants
  (not better by the margin), logistic regression (0.37 lower), anomaly detectors (far worse, no
  probability), dummy (baseline only).
- **Dependencies:** `xgboost` and `lightgbm` are **runtime** dependencies, because the app can
  select them (approved by Suhas in the review).

## Consequences
- **Each switchable model needs its own threshold, and ideally the same operating-point analysis, in
  ADR 0007.** One threshold cannot be shared across models. ADR 0009's dropdown must load the model
  and its threshold together, and the single holdout score is made once for the model finally
  chosen, not once per dropdown entry.
- One serialized `Pipeline` artifact per switchable model lives in `models/`, which is never
  committed. Training the three offered models costs about 6.5 s per fit (5.7 + 0.44 + 0.35).
- **Install size.** `xgboost` pulls about 84 MB plus 345 MB of NVIDIA libraries (`nvidia-nccl-cu13`),
  and `lightgbm` about 10 MB. The CUDA libraries are dead weight for a CPU app on a free-tier host
  (ADR 0010). `xgboost-cpu` (same 3.4.1, same `import xgboost`, no CUDA) resolves and is a one-line
  swap. It is **not applied**; a human decides.
- The forest is the slowest and largest: 5.7 s per fit, 25.7–62.6 ms per single-Reading score
  (load-dependent), 4.6 MB. Re-measure on the deployment target (ADR 0010).
- The probability is coarse at the low end: many Readings score about zero, and the boosters
  under-predict in the second decile. The app must never show "0 %" or imply certainty (ADR 0009);
  ADR 0007 will threshold probabilities with many near-ties.
- **ADR 0004's engineered features were chosen with HistGradientBoosting as the non-linear model and
  not re-tested with the forest or the offered boosters.** Re-verify them before ADR 0007.
- The 70/30 check passed for the default and for all three offered models (0.003, 0.003, 0.007, each
  below the 0.01 trigger). HistGradientBoosting's 0.026 penalty (under this paired, in-train
  protocol, against about 0.000 in ADR 0005's unpaired full-data re-splits) is the only one above
  the trigger, and it is no longer offered. It also suggests ADR 0005's boosting figure may
  understate the true penalty, since it rests on test-set noise (standard error about 0.007 per
  ratio).
- Selecting the best of 13 on the same CV carries a small winner's-curse bias, and the lead over the
  boosters is within the margin. The unbiased estimate is the single holdout score.
- Untuned defaults are a limitation. A tuning follow-up (nested CV) could reorder the tree
  families; it is a separate decision, warranted because the gaps are within the margin.
- Machine failure predictions say nothing about *why* a Reading is risky; mode prediction is still
  deferred (ADR 0009).

Not decided here: threshold and cost metric (ADR 0007); tuning; refitting on the full data after the
single test evaluation (open since ADR 0005); recomputing ADR 0004's deltas on the 70/30 train
partition; the per-mode predictability check (ADR 0009); `xgboost` vs `xgboost-cpu`; whether to move
the model settings into `configs/default.toml` (a `[model]` table, when this ADR is Accepted).
