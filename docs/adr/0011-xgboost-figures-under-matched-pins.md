# 0011. XGBoost figures under the matched library pins

- Status: Proposed
- Date: 2026-10-13
- Deciders: Suhas (agreed in a grill session 2026-10-13); drafted by an agent
- Refines: [0007](0007-metric-and-threshold.md), **XGBoost figures only**. Nothing else in ADR 0007,
  and nothing in 0004 to 0006 or 0008, is changed. ADR 0007's text and status are left as they are;
  on acceptance the human decides whether to annotate it.
- Numbering: this ADR follows 0010, the deployment ADR that [0009](0009-app-output.md) briefs and that
  ADRs 0006 to 0008 and the decision plan already name. Nothing needs renumbering.

## Context
[ADR 0009](0009-app-output.md) chose marimo WebAssembly on GitHub Pages as the first host. Its
inspection found that artifacts must be fit in an environment matching marimo's Pyodide
(scikit-learn 1.8.0, xgboost 2.1.4, lightgbm 4.6.0, polars 1.33.1, joblib 1.5.3, numpy 2.4.3, scipy
1.17.1); an XGBoost pickle moved across versions changed 47 Flags per 1,000. The owner chose to pin the
whole project to those versions (option A in ADR 0009). The evidence behind ADRs 0005 to 0008 was
produced under newer versions (scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0, polars 2.0.0), so
notebooks 01, 02, 03, 03b and 04 were re-run unchanged, with the same seeds and train partition only.
The holdout was not touched.

## Options considered
1. **Leave ADR 0007's XGBoost figures and let the regenerated artifact differ silently.** Simple, and
   wrong: the app and the accepted text would disagree, and notebook 04's guard stops on the gap.
2. **Supersede ADR 0007 in full.** Heavy for a change in one model's figures; it would also mark the
   forest and LightGBM results as superseded when they reproduced exactly.
3. **Record the changed figures in a short ADR that refines ADR 0007 for XGBoost only** (chosen).

Criterion, declared before the comparison: a figure is restated only if it differs from ADR 0007 at
its published precision. Decisions are re-opened only if a ranking or verdict ADR 0005 to 0008 relied
on flips.

## Results
Train-partition repeated stratified 5×5 CV, as in ADR 0007.

- **Random forest and LightGBM: identical to six decimals** (thresholds, recall, precision, Flags per
  1,000, PR-AUC, Brier). Nothing is restated for them.
- **XGBoost** (10:1 Cost ratio, median over repeats):

  | Quantity | ADR 0007 | Matched pins |
  |---|---|---|
  | Operating threshold, **10:1** | 0.075 | **0.0727** |
  | Threshold at 5:1 | 0.242 | 0.257 |
  | Threshold at 20:1 | 0.021 | 0.0117 |
  | Repeat range at 10:1 | 0.046 to 0.148 | 0.021 to 0.100 |
  | Bootstrap interval at 10:1 | 0.021 to 0.239 | 0.021 to 0.151 |
  | Recall at 10:1 | 83.9 % | 83.4 % |
  | Precision at 10:1 | 70.1 % | 70.6 % |
  | Flags per 1,000 at 10:1 | 40.5 | 40.0 |
  | False alarms per caught failure | 0.43 | 0.42 |
  | Cost per 1,000 (False-alarm units) | 66.7 | 68.0 |
  | PR-AUC / ROC-AUC / Brier | 0.862 / 0.972 / 0.0094 | 0.866 / 0.972 / 0.0090 |
  | Cost against the forest, per 1,000 (corrected SE) | +10.5 (5.95) | +11.8 (5.69) |
  | Engineered features: all three minus raw PR-AUC | +0.086, agrees | +0.079, agrees |

- **No ranking or verdict the earlier ADRs relied on flips.** The forest stays first; XGBoost stays the
  costliest of the three and is not cheaper than the forest; ADR 0004's feature set still agrees for
  all three models; ADR 0008's choice (reject for all models) stands, with no pattern verdict
  flipping for XGBoost. One rank moves below the top (the weighted forest, 0.883, now above
  unweighted LightGBM, 0.882); the weighted variants are not offered.
- The risk-band table for XGBoost moved (Elevated: 56 Readings, 15 % failed). **No bands ship**
  (ADR 0007), so it was not re-judged.
- Notebook 04's frozen-threshold guard, which stopped on 0.0727, now checks 0.073 for XGBoost and
  passes.

## Decision
- **XGBoost's Operating threshold at the 10:1 Cost ratio is 0.0727**, and its Flag describes recall
  83 % and precision 71 %, about 40 Flags per 1,000 Readings, with the ADR 0007 interval caveats. The
  forest's 0.218 and LightGBM's 0.134 stand.
- **ADR 0007's XGBoost figures are replaced by the table above** wherever an app, sidecar or document
  quotes them. The artifact and its `operating-metrics.json` sidecar are regenerated in the matched
  environment, and the manifest (ADR 0009) records the library versions that produced them.
- **A threshold is only valid for the library versions that fit its model.** Any later change to a
  pinned version re-runs the notebooks, and a figure that moves at its published precision gets a new
  ADR of this kind.

## Consequences
- The 10:1 threshold's uncertainty is unchanged in kind: the bootstrap interval still spans a factor
  of about 7 for XGBoost.
- ADR 0007's text keeps the old XGBoost figures. A reader of 0007 alone is misled unless they follow
  this ADR; that is the price of not editing an Accepted ADR.
- The pins freeze the project to marimo's Pyodide until it moves (ADR 0009).

Not decided here: the host (ADR 0010); the `xgboost` against `xgboost-cpu` swap (xgboost 2.1.4 pulls
`nvidia-nccl-cu12`; ADR 0010); model tuning; the real Cost ratio; the single holdout evaluation.
