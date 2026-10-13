# 0007. Metric and threshold

- Status: Accepted
- Date: 2026-10-13
- Deciders: Suhas (rules agreed in a grill session 2026-10-13; accepted 2026-10-13); drafted by an agent

## Context
ADR 0006 chose models by PR-AUC, which ranks Readings but says nothing about *where to act*. The app
(ADR 0009) must turn a Failure risk into a **Flag**, and the spec (§4.4) asks for an evaluation
metric justified by the cost of errors. Fixed by accepted ADRs: target `Machine failure`; `Type` + 5
sensors + 3 engineered features, failure modes and identifiers excluded
([0004](0004-target-and-features.md)); stratified 70/30 holdout (about 102 test failures) made once and
scored once, for the single model finally chosen; threshold work on the train partition only (7,000
Readings, 237 failures) with repeated stratified 5×5 CV, preprocessing fitted in-fold
([0005](0005-split-and-validation.md)); three offered models, random forest (default), LightGBM and
XGBoost, all unweighted and uncalibrated, each its own `Pipeline` ([0006](0006-model-family.md)).
Seed 42 from `configs/default.toml`. Vocabulary follows [`GLOSSARY.md`](../../GLOSSARY.md).

This ADR decides the **operating-point metric and the threshold**, not the model. It does not change
the default model.

Evidence: [`notebooks/03b_threshold.py`](../../notebooks/03b_threshold.py), built on
`predictive_maintenance.evaluation.threshold` and `.compare` (tested) and the 25 identical folds of
ADR 0005. **The holdout was discarded at load time and never scored**; every number below is
train-partition CV.

### What the two errors mean for a maintenance engineer
- A **Missed failure** is a Machine failure the model did not Flag. The machine stops without
  warning: unplanned downtime, possible damage, a rushed repair.
- A **False alarm** is a Flag on a Reading that would not have failed. It costs one unnecessary
  inspection.
- The **Cost ratio** says how many False alarms one Missed failure is worth. It is an
  **assumption, not a measurement**: the dataset has no costs. It is owned by maintenance
  management, lives in `configs/default.toml` (`[threshold]`, documented in `configs/README.md`)
  and is meant to be replaced by the plant's real figure. Expected cost per Reading is
  `(ratio × Missed failures + False alarms) / Readings`, in False-alarm units.

### Disclosures
- **The rules below were written before any threshold number was produced.** The notebook states
  them in its introduction. The margins (0.005, 0.02, 25 %, 10 failures, 10 %, 1 %) are judgement
  calls, so they can be challenged.
- **Thresholds are picked and judged on the same CV scores.** This makes the CV costs slightly
  optimistic. It affects the three models equally, so the comparison between them is fair, but the
  absolute figures are not an unbiased forecast. The unbiased figure is the single holdout score.
- The 5:1, 10:1 and 20:1 ratios and the 10:1 default come from the decision plan, not from data.

## Options considered
**Framing.** (a) A single Cost ratio drives the Flag, and a precision/recall table beside it explains
it; (b) a precision/recall table only, so engineers pick their own point; (c) a fixed recall target.
Option (a) was chosen. A table alone leaves the decision implicit. A fixed recall target ignores
what false alarms cost. The table of (b) is kept as the explanation.

Rules, fixed before the numbers they judge:
1. **Feature re-check first** (ADR 0006 flagged that ADR 0004's engineered features were chosen with
   HistGradientBoosting). All three engineered features minus raw features, mean paired PR-AUC delta
   on identical folds, per offered model, unweighted. *Agrees* at ≥ −0.005, *disagrees* at ≤ −0.02,
   *noise* between. A disagreement stops the work and goes back to a human; ADR 0004 is not edited.
2. **Threshold per CV repeat.** Each repeat pools its five test folds into one out-of-fold score per
   Reading (so the threshold is not fitted on the average of 5 repeats, which is smoother than any
   shipped model). The threshold minimises expected cost per Reading. Candidates are the distinct
   observed scores, so tied Readings are never split. Among exactly-tied minima, the widest interval
   wins and its midpoint is used. The shipped **Operating threshold** is the median over the five
   repeats, reported with the repeat range and a 90 % stratified-bootstrap interval (failures and
   non-failures resampled separately, 2000 resamples).
3. **Viability.** A model's expected cost must beat *flag nothing* by ≥ 25 % at every ratio, or it is
   not offered.
4. **Bands.** Low < t(10:1) ≤ Elevated < t(5:1) ≤ High. They ship only if Elevated and High each hold
   ≥ 10 failures (mean per repeat) and the observed failure rate strictly rises across the bands.
   If the two edges are not strictly ordered, the build fails loudly; if the check fails otherwise,
   no bands ship for that model.
5. **Risk floor.** The 95 % Wilson upper bound on the failure rate of the lowest-scoring half of
   Readings, rounded up to a whole percent, at least 1 %, and the maximum over the three models, so
   the app has one wording. Below it the app says "under F %".
6. **Model cost comparison (informational).** A booster is *cheaper* than the forest if its paired
   per-fold expected-cost saving at 10:1 exceeds `max(10 % of the forest's cost, Nadeau–Bengio
   corrected SE)`. It never changes the default.
7. **Flat optimum (informational).** If the cost at both ends of the 90 % bootstrap interval is within
   10 % of the cost at the median threshold, the optimum is flat.
8. **Metrics.** Primary at the operating point: expected cost at the default ratio, against the
   baselines *flag nothing* and *flag everything*. Headline for engineers: recall, precision, False
   alarms per caught failure, alerts per 1,000 Readings. Supporting: PR-AUC (ranking), ROC-AUC
   (optimistic at 3.4 % prevalence, because the many non-failures flatter it) and Brier (probability
   quality).
9. **Final holdout metrics (later, once).** Thresholds frozen, the chosen model scored once,
   percentile 95 % intervals from a stratified bootstrap (2000 resamples, seed stream 11 of the
   config seed) for recall, precision, expected cost, PR-AUC, ROC-AUC and Brier. The threshold is
   never re-chosen inside a resample.

## Results
All numbers are train-partition CV (25 folds, 5 repeats of 5), per model unless stated.

**Feature re-check** (rule 1; PR-AUC, all three engineered features vs raw features):

| Model | Raw | All three | Mean delta | Corrected SE | Verdict |
|---|---|---|---|---|---|
| Random forest | 0.768 | 0.894 | +0.126 | 0.021 | agrees |
| LightGBM | 0.795 | 0.882 | +0.087 | 0.018 | agrees |
| XGBoost | 0.779 | 0.865 | +0.086 | 0.021 | agrees |

ADR 0004's feature set holds for all three offered models, so the threshold work proceeded.

**Supporting metrics** (pooled out-of-fold, mean over the 5 repeats):

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Random forest | 0.893 | 0.973 | 0.0075 |
| LightGBM | 0.880 | 0.973 | 0.0082 |
| XGBoost | 0.862 | 0.972 | 0.0094 |

ROC-AUC is nearly identical (0.972–0.973) and does not separate the models; PR-AUC and Brier do.

**Operating thresholds** (rule 2; median over repeats, range over repeats, 90 % bootstrap interval):

| Model | Ratio | Threshold | Repeat range | Bootstrap interval | Cost excess at interval ends | Flat |
|---|---|---|---|---|---|---|
| Random forest | 5:1 | 0.242 | 0.218–0.275 | 0.202–0.423 | +3.2 % / +6.5 % | yes |
| Random forest | **10:1** | **0.218** | 0.138–0.242 | 0.138–0.275 | +6.8 % / +6.2 % | yes |
| Random forest | 20:1 | 0.150 | 0.072–0.218 | 0.038–0.238 | +11.5 % / +3.6 % | **no** |
| LightGBM | 5:1 | 0.148 | 0.089–0.336 | 0.045–0.336 | +5.7 % / +7.5 % | yes |
| LightGBM | **10:1** | **0.134** | 0.015–0.148 | 0.004–0.226 | +9.5 % / +5.0 % | yes |
| LightGBM | 20:1 | 0.0048 | 0.001–0.009 | 0.0008–0.045 | +3.2 % / +6.2 % | yes |
| XGBoost | 5:1 | 0.242 | 0.175–0.431 | 0.074–0.452 | +5.8 % / +7.6 % | yes |
| XGBoost | **10:1** | **0.075** | 0.046–0.148 | 0.021–0.239 | +5.2 % / +3.7 % | yes |
| XGBoost | 20:1 | 0.021 | 0.005–0.053 | 0.005–0.066 | +1.7 % / +4.7 % | yes |

The 10:1 thresholds differ a lot between models (0.218, 0.134, 0.075), so one threshold cannot be
shared (ADR 0006's consequence is confirmed). **Threshold uncertainty is large**: with 237 failures
the 10:1 bootstrap interval spans a factor of 2 for the forest and more than 10 for the boosters,
and the repeat range is wide for the boosters. The cost curve is flat around the optimum, though: even at the interval ends
the cost is within about 10 % of the minimum, for eight of the nine cells. The threshold is
therefore imprecise, but the *cost* it implies is stable. The one non-flat cell is the forest at
20:1.

**What each threshold means** (rule 8; CV, mean over repeats; cost per 1,000 Readings in False-alarm
units; saving is against *flag nothing*; *flag everything* costs 966 per 1,000 for every row):

| Model | Ratio | Recall | Precision | Alerts / 1,000 | False alarms per caught | Cost / 1,000 | Flag nothing | Saving |
|---|---|---|---|---|---|---|---|---|
| Random forest | 5:1 | 84.9 % | 82.4 % | 34.9 | 0.21 | 31.7 | 169.3 | 81.3 % |
| Random forest | **10:1** | **85.6 %** | **79.8 %** | **36.3** | **0.25** | **56.2** | 338.6 | **83.4 %** |
| Random forest | 20:1 | 86.8 % | 67.8 % | 43.3 | 0.48 | 103.7 | 677.1 | 84.7 % |
| LightGBM | 5:1 | 83.0 % | 85.9 % | 32.7 | 0.16 | 33.3 | 169.3 | 80.3 % |
| LightGBM | **10:1** | **83.3 %** | **85.1 %** | **33.1** | **0.18** | **61.5** | 338.6 | **81.8 %** |
| LightGBM | 20:1 | 87.3 % | 55.7 % | 53.1 | 0.80 | 109.8 | 677.1 | 83.8 % |
| XGBoost | 5:1 | 81.3 % | 82.9 % | 33.2 | 0.21 | 37.4 | 169.3 | 77.9 % |
| XGBoost | **10:1** | **83.9 %** | **70.1 %** | **40.5** | **0.43** | **66.7** | 338.6 | **80.3 %** |
| XGBoost | 20:1 | 86.8 % | 53.5 % | 55.0 | 0.87 | 114.7 | 677.1 | 83.1 % |

In plain terms, for the default forest at 10:1: of every 100 Machine failures, about 86 are Flagged
and 14 are missed; of every 100 Flags, about 80 are real and 20 are False alarms; and the engineer
sees about 36 Flags per 1,000 Readings. Going from 10:1 to 20:1 buys about one more caught failure
in 100 and costs about 7 more alerts per 1,000 Readings. Recall barely moves between ratios, which
is why the threshold is a stable *operating* choice even though it is imprecise as a number.

All three models are **viable** (rule 3): the smallest saving over *flag nothing* is 77.9 %, far
above 25 %.

**Model cost comparison at 10:1** (rule 6; positive = cheaper than the forest; per 1,000 Readings):

| Model | Saving vs forest | Relative | Corrected SE | Cheaper? |
|---|---|---|---|---|
| LightGBM | −5.3 | −9.5 % | 5.7 | no |
| XGBoost | −10.5 | −18.7 % | 6.0 | no |

Neither booster is cheaper than the forest: both cost more, by 9.5 % and 18.7 % (about one and two
corrected SEs). The default stays the forest.

**Risk floor** (rule 5): 1 % for all three models; shared floor **1 %**. The 95 % Wilson upper bound for the
lowest-scoring half of Readings is at or below 1 % for every model, so the floor sits at the
configured minimum rather than being raised by the data. The app never shows "0 %" and shows "under 1 %" below it.
The 20:1 LightGBM threshold (0.0048) sits **below** this floor: at that ratio the booster flags on
a probability the app is not allowed to show. This is a warning against offering the 20:1 setting for
LightGBM, not a flaw in the floor.

**Bands** (rule 4; CV, mean per repeat). The edges are t(10:1) and t(5:1):

| Model | Band | Readings | Failures | Observed rate | 95 % Wilson |
|---|---|---|---|---|---|
| Random forest | Low | 6745.8 | 34.2 | 0.5 % | 0.4–0.7 % |
| Random forest | Elevated | 10.0 | 1.6 | 16 % | 3.9–46.9 % |
| Random forest | High | 244.2 | 201.2 | 82.4 % | 77.1–86.7 % |
| LightGBM | Low | 6768.0 | 39.6 | 0.6 % | 0.4–0.8 % |
| LightGBM | Elevated | 2.8 | 0.6 | 21 % | 2.6–73.3 % |
| LightGBM | High | 229.2 | 196.8 | 85.9 % | 80.8–89.8 % |
| XGBoost | Low | 6716.4 | 38.2 | 0.6 % | 0.4–0.8 % |
| XGBoost | Elevated | 51.2 | 6.2 | 12.1 % | 5.7–23.8 % |
| XGBoost | High | 232.4 | 192.6 | 82.9 % | 77.5–87.2 % |

**No model passes the band check**: Elevated holds 1.6, 0.6 and 6.2 failures against the required 10.
The Elevated band is a sliver, because t(5:1) and t(10:1) are close (forest 0.242 and 0.218). The
data cannot support a middle band at these ratios. **Bands do not ship** (the edges were strictly
ordered, so the build did not fail; the check failed instead).

## Decision
Accepted:
- **Primary metric at the operating point: expected cost per Reading at the default Cost ratio**, in
  False-alarm units, reported against *flag nothing* and *flag everything*. Engineers see recall,
  precision, False alarms per caught failure and alerts per 1,000 Readings beside it. PR-AUC (model
  choice, ADR 0006), ROC-AUC and Brier are supporting only.
- **Default Cost ratio 10:1, with 5:1 and 20:1 shown as sensitivity.** The ratio is an assumption
  owned by maintenance management and set in `configs/default.toml`. The Flag uses the default only.
- **One Operating threshold per offered model, picked per CV repeat and shipped as the median over
  repeats** (rule 2), with the repeat range and bootstrap interval reported to make the uncertainty
  visible:

  | Model | Operating threshold (10:1) |
  |---|---|
  | Random forest (default) | 0.218 |
  | LightGBM | 0.134 |
  | XGBoost | 0.075 |

  Thresholds at 5:1 and 20:1 are stored beside them. The forest's 20:1 optimum is not flat and
  LightGBM's 20:1 threshold is below the risk floor, so neither should be presented as precise.
- **A Flag is binary (Flagged / Not flagged). No Risk bands ship**, because the band check failed for
  every model (rule 4). The `Risk band` term stays in the glossary as a concept that was considered
  and not adopted; it can return if the 5:1 and 10:1 thresholds are ever far enough apart to hold ≥ 10
  failures in the Elevated band.
- **Risk floor 1 %**: the app shows "under 1 %" below it and never "0 %".
- **Each model ships with a sidecar JSON** (`OperatingPoint`: model, default ratio, thresholds per
  ratio, `bands: null`, floor) beside its serialized `Pipeline`. Writing the sidecars is the training
  command's job and a later ticket.
- **The default model is unchanged (random forest).** Neither booster is cheaper (rule 6). All three
  models are viable (rule 3), so all three stay offered.
- **Final holdout evaluation** follows rule 9, once, for the single chosen model, thresholds frozen.

## Consequences
- ADR 0009's dropdown must load the **model and its operating point together** (the sidecar). A
  Failure risk from one model must never be compared to another model's threshold. The same 10:1
  ratio gives 0.218, 0.134 and 0.075, so the Flag rate differs: about 36, 33 and 41 alerts per 1,000
  Readings.
- The fair comparison across models is the paired per-fold expected cost at the same ratio on identical
  folds, not the threshold value. The app should describe a Flag in terms of recall and precision,
  not a raw probability.
- **Threshold uncertainty is real** (bootstrap interval a factor of 2 to over 10 in probability, 237 failures)
  but the cost is stable around it, so the Flag is robust in what it costs even if the number is not.
  Retraining or a different seed can move it; sidecars must be regenerated whenever a `Pipeline` is
  refitted.
- Two-state output makes the app simpler but gives engineers no "watch closely" middle state. ADR 0009
  must not invent bands from the sensors or probability by hand.
- The Cost ratio is a placeholder. If management supplies a real ratio, change `default_cost_ratio`
  in `configs/default.toml` and rerun notebook 03b; nothing in code changes.
- Holdout intervals will be wide: about 102 failures, so a recall interval of roughly ±7 points.
- `xgboost` vs `xgboost-cpu` is unchanged (ADR 0010).

Not decided here: missing-data policy (ADR 0008); app wording and layout, including how a Flag and the
risk floor are phrased (ADR 0009); deployment (ADR 0010); model tuning; the per-failure-mode check;
refitting on the full data after the single test evaluation (open since ADR 0005); writing the sidecar
files; the real Cost ratio.
