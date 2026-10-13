# 0005. Split and validation

- Status: Accepted
- Date: 2026-10-12
- Deciders: Suhas (accepted 2026-10-13)

## Context
The spec requires a leakage-safe evaluation (§3, §8) and the decision plan fixes a default: stratified
80/20 holdout touched once at the end, and repeated 5×5 stratified CV on the train partition for every
comparison and threshold choice, with the seed from `configs/default.toml`. The plan also says
**evidence wins over the default**. Evidence is in
[`notebooks/02_split_strategy.py`](../../notebooks/02_split_strategy.py); vocabulary follows
[`GLOSSARY.md`](../../GLOSSARY.md).

Taken as given from [ADR 0004](0004-target-and-features.md): target `Machine failure`, `Type` + 5 sensors
+ all three engineered features, failure modes and identifiers excluded. The probe is logistic
regression on that feature set (plus `HistGradientBoostingClassifier` as a second probe for the
train-size question in comparison 3). This ADR does not choose the model family (ADR 0006).

Constraints from the data ([`decision-plan.md`](../design/decision-plan.md)): 10,000 Readings, 339
Machine failures (3.39 %); every `Product ID` is unique (no repeated machines, so no group split) and
rows have no time order (no temporal split). The only structure a split must respect is the class
imbalance.

Method: each comparison states its decision rule before the numbers; seeds derive from the config
seed. The real holdout is never scored. Comparisons 1 and 3 re-split the full data many times and score
throwaway test sets only to measure split variance; comparison 2 uses the train partition only.
The margins in the rules (10 %, 25 %, 15 %, 0.01) are judgement calls, written down so they can be
challenged.

## Options considered
**1. Plain vs stratified holdout** (80/20, 200 seeds). Rule: keep stratified unless its range of test
positives is > 1 or its test PR-AUC SD is > 10 % above plain.

| | Test positives (mean, range) | Seeds > 10 % off expected 67.8 | Test PR-AUC SD |
|---|---|---|---|
| Plain | 67.9 (48–87), SD 7.5 | 40 % | 0.054 |
| Stratified | 68 every seed | 0 % | 0.052 |

SD ratio stratified/plain 0.97 (bootstrap 95 %: 0.85–1.12). **Stratified wins**: a plain split sometimes
leaves the test set with 48 failures, and the metric spread does not improve measurably (the interval
includes 1), so the benefit is the guaranteed positive count, not a lower metric variance.

**2. Single 80/20 validation split vs repeated stratified 5×5 CV** (train partition, 30 replicates, 8
close-call logistic candidates, placeholder cost-minimising threshold for the plain `C=1` probe).
Rule: adopt repeated CV only if it cuts both ranking disagreement (1 − mean pairwise Spearman ρ) and
the 10:1 threshold SD by ≥ 25 %.

| | Single 80/20 | Repeated 5×5 CV |
|---|---|---|
| Ranking disagreement (1 − ρ) | 0.37 | 0.00 |
| Same winner across replicates | 47 % (5 different winners) | 100 % |
| Per-candidate PR-AUC SD | ≈ 0.05 | ≈ 0.002 |
| Threshold SD at 5:1 / 10:1 / 20:1 | 0.044 / 0.030 / 0.021 | 0.012 / 0.010 / 0.005 |
| Fits per candidate set | 8 | 200 |

**Repeated CV wins** (ranking −100 %, threshold SD −67 %). The candidates differ by 0.007–0.11 PR-AUC
while a single split's noise is ≈ 0.05, so a single split cannot reliably rank them. Cost: 25× the fits
(about 10 s vs 0.4 s per replicate here). The measure is procedure noise on fixed data (what a different
seed alone changes), not sampling error of the data itself.

**3. Split ratio: 80/20 vs 70/30 vs 90/10** (60 seeds each, stratified, 1000-resample bootstrap CI of
test PR-AUC). Rule: 70/30 replaces 80/20 if, on both probes, the CI is ≥ 15 % narrower and the mean test
PR-AUC drops ≤ 0.01, and the logistic probe's between-fold CV PR-AUC SD rises ≤ 15 %. 90/10 replaces
it if its CI is ≤ 15 % wider and its mean PR-AUC is ≥ 0.01 higher.

| Ratio | Test / train positives | CI width, logistic | CI width, boosting | Mean test PR-AUC, logistic / boosting | Fold SD of CV, logistic |
|---|---|---|---|---|---|
| 90/10 | 34 / 305 | 0.321 (+35 %) | 0.169 (+35 %) | 0.497 / 0.892 | 0.056 |
| 80/20 | 68 / 271 | 0.237 | 0.125 | 0.488 / 0.886 | 0.061 |
| 70/30 | 102 / 237 | 0.197 (−17 %) | 0.103 (−18 %) | 0.484 / 0.886 | 0.063 (+3 %) |

**70/30 passes every criterion; 90/10 fails.** Caveat: a CI's width scales as 1/√positives, so 70/30
is expected to be about 18 % narrower by arithmetic alone and the 15 % margin sits close to that. The
CI gain is guaranteed; the finding is that the train-size penalty is small (−0.003 logistic, 0.000
boosting) and selection-side noise barely rises. If a reviewer weighs a smaller train partition more
heavily, 80/20 is the defensible alternative.

**Not considered** (ruled out by the data facts): group-aware and temporal splits; nested CV was not
tested.

## Decision
Accepted (decisions by Suhas, 2026-10-13):
- **Holdout:** stratified on `Machine failure`, **70/30** (30 % test, about 102 test positives), made
  once with a seed from `configs/default.toml` via `predictive_maintenance.evaluation.split`, before
  anything is fitted. The test partition is scored **once**, at the end, never for selection.
- **Model comparison and threshold selection:** repeated stratified **5×5 CV on the train partition**
  (237 train positives), with every preprocessing step fitted inside the fold. A single validation
  split is not used for selection.
- **Reporting:** every test metric carries a bootstrap CI (test CI width of PR-AUC ≈ 0.20 for the
  logistic probe, ≈ 0.10 for the boosting probe at 70/30).
- **Split code:** `stratified_holdout` and `repeated_stratified_cv` live in
  `src/predictive_maintenance/evaluation/split.py` and take their ratios as required arguments. The
  ratio and CV settings move into `configs/default.toml` when this ADR is Accepted.

This **overturns the plan's 80/20 default** and confirms its repeated 5×5 CV and stratification.

## Consequences
- Test-set metrics are more precise (about 102 positives instead of 68) at no measured accuracy cost,
  but 12 % fewer failures are left for training and for CV (237 instead of 271).
- Stratified splitting and repeated CV are decided; both are implemented and tested in the package.
  The seed stays the single source: no other seeds are introduced.
- Repeated CV costs 25× more fits than a single split. Model comparison in notebook 03 should expect
  that; it is cheap for the logistic probe, slower for `HistGradientBoostingClassifier`.
- **The decision plan still says 80/20 and "about 68 test positives"; update it, and the `configs/`
  README, when this ADR is Accepted.** ADR 0007 bootstrap CIs will be computed on about 102 test positives.
- **[ADR 0004](0004-target-and-features.md)'s CV figures were computed on the provisional 80/20 train
  partition.** The feature-set conclusion is not reopened here, but its deltas should be recomputed
  on the 70/30 train partition when convenient to confirm they do not move.
- The threshold probe here (cost-minimising at 5:1, 10:1, 20:1) is a stability test, not the ADR 0007
  decision. The finding that repeated CV makes the selected threshold about 3× steadier carries over to
  whatever rule ADR 0007 picks; the numbers do not.
- Revisit the ratio if ADR 0006 selects a model whose learning curve is much steeper than the two probes
  tested, or if more failures become available (the penalty/CI trade-off shifts).
- Not decided here: whether the final model is refit on the full data after the single test evaluation.
