# 0008. Missing-data policy

- Status: Accepted
- Date: 2026-10-13
- Deciders: Suhas (design agreed in a grill session 2026-10-13; accepted 2026-10-13); drafted by an agent

## Context
The raw data has no nulls, so nothing is cleaned at training time. Missing data is an
**inference-time** problem: a **Reading** reaches the app with an input absent (a sensor offline, a
blank `Type`, a value that is not a number or physically impossible). The spec (§8) asks that
missing or invalid input be handled and the choice justified. This ADR decides what the app does with
such an **Incomplete reading**: score it (and how, and how reliably) or refuse it.

Fixed by accepted ADRs: target `Machine failure`; `Type`, the 5 sensors and the 3 engineered
features, failure modes and identifiers excluded ([0004](0004-target-and-features.md)); stratified
70/30 holdout made once and scored once ([0005](0005-split-and-validation.md)); all comparisons on
repeated stratified 5×5 CV of the train partition (7,000 Readings, 237 failures), preprocessing fitted
inside each fold, identical folds; three offered, unweighted, uncalibrated models each as its own
`Pipeline` ([0006](0006-model-family.md)); a binary **Flag**, one **Operating threshold** per model at
the 10:1 default Cost ratio (forest 0.218, LightGBM 0.134, XGBoost 0.075), a 1 % **Risk floor**, and
expected cost at the default ratio as the primary operating-point metric
([0007](0007-metric-and-threshold.md)). Seed 42 from `configs/default.toml`. Vocabulary follows
[`GLOSSARY.md`](../../GLOSSARY.md); this ADR sharpens **Incomplete reading** and adds **Refused
reading**.

Evidence: [`notebooks/04_missing_data.py`](../../notebooks/04_missing_data.py), built on
`predictive_maintenance.evaluation.missingness` and `preprocessing.imputation` (both tested). **The
holdout was discarded at load time and never scored**; every number below is train-partition CV.

### Facts that shape the problem
- **The six inputs can be absent**: `Type` and the five sensors. The engineered features are built from
  raw sensors, so one absent sensor can blank several features (torque absent blanks Torque, Power
  and Wear × torque: 3 of the 8 numeric features).
- **All three offered models accept NaN at predict time** (verified in the installed scikit-learn
  1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, polars 2.0.0). But a model fitted on complete data never saw a
  NaN, so it routes it arbitrarily. On a toy Reading a NaN scored 0.4 (forest), 0.13 (LightGBM) and
  about 0.97 (XGBoost) against true values of about 0.95, 1.0 and 0. "Handles NaN natively" does not
  mean "handles it well".
- **A null `Type` crashes the current `OneHotEncoder`** (`unknown categories [None]`). Under the
  adopted policy (reject) a Reading with an absent input never reaches the `Pipeline`: `inference/`
  refuses it first. The tolerant encoder exists in code only for the measurement.
- The decision plan's "HistGradientBoosting only" for native NaN handling, and its "three bands",
  are **stale** (ADR 0006 dropped HistGradientBoosting from the offer; ADR 0007 shipped no bands). The
  plan is not edited here.

### Disclosures
- **The rules below were fixed, and the margins written into `configs/default.toml`
  (`[missing_data]`), before the final run.** The notebook states them in its introduction. The
  margins (0.90, 0.10, ×0.5 to ×2, 0.05, 0.75) are judgement calls and can be challenged.
- **A 5-fold, 1-repeat smoke run on the random forest was seen during development**, before the full
  run, and suggested that only `Type` (sometimes process temperature) would pass. The margins were
  not changed afterwards. The full run agrees.
- **Complete-trained only.** The models are not retrained on injected nulls. This keeps ADR 0006/0007
  models and thresholds unchanged (complete-Reading scores are bit-identical with and without the
  imputation step; tested). It means "median + missing-indicators" is not offered (an indicator has
  nothing to learn from when training has no nulls), and that a model *trained* on incomplete data
  might cope better. That is untested and not decided here.
- **Absence is injected on 100 % of the validation rows** for each pattern, not at a random rate. The
  app scores one Reading at a time, so the question is "given these inputs are absent, how good is
  the Flag?". The **informative-missingness** question (is a sensor more likely to drop out when the
  machine is about to fail?) cannot be answered from this data, because it has no nulls. One stress
  case and a sizing check stand in for it.
- **Absence rates for the data-loss check (1 %, 5 %, 10 % per input, independent) are assumptions**,
  not measurements.
- **Thresholds are frozen** (ADR 0007) and re-derived in the notebook; the rule rejects any drift
  above 0.0005 from the ADR values. They match.
- **The CV figures are slightly optimistic** for the same reason as in ADR 0007, and identically so for
  all models.
- **Three outcomes are knife-edges** (see Results). The rules are applied as written and not adjusted.
- **The adopted policy is a human override of the rules' result.** Rule 6's walk picks median
  imputation for all three models. The project owner chose `reject` for all three, because the
  visible gain is not reliable enough under the current assumptions: it rests on the knife-edges,
  on complete-trained models, on absence injected on 100 % of validation rows, on assumed
  absence rates, and on a Cost ratio that is a placeholder. The measurement is kept, as a
  reference, so the choice can be revisited when real data says it is worth it (see the health
  metrics in the Decision).

## Options considered
**Framing.** (a) A pre-declared, measured allow-list per model of the absent-input patterns the app
will score, refusing everything else; (b) one blanket rule ("refuse if any sensor is absent"); (c)
always impute and score. (c) hides degradation from the engineer. (b) needs no measurement but throws
away the cases that cost nothing (the results show `Type` and process temperature are such cases).
Option (a) was used as the **measurement**; the policy adopted from it is the owner's override to
`reject` (below). The decision plan's single rule, "adopt a strategy only if PR-AUC drops less
than 10 % relative with one sensor missing", is replaced: PR-AUC ranks Readings but ADR 0007 made
expected cost at the frozen threshold the primary metric, and the Flag rate is what an engineer feels.
PR-AUC is still reported.

Rules, fixed before the numbers they judge:
1. **Scenarios.** All 63 non-empty absent-input patterns, injected on 100 % of the validation rows only
   (training stays complete), on identical folds for every model and strategy. Plus one **informative
   stress**: torque absent only on validation Readings at or above the training fold's 0.75 speed
   quantile, judged on those rows against the same rows complete. It can only veto the `Torque`
   pattern (and, through rule 5, every pattern containing torque); it can never allow one.
2. **Strategies, simplest first:** `reject`; `median` (median for numbers, mode for `Type`);
   `regression` (a ridge per absent set, fitted on complete training rows; a full sensor outage falls
   back to the median); `native` (NaN passed straight to the model). Imputation happens **before** the
   features are derived and is fitted on the **training fold only**. scikit-learn's `IterativeImputer`
   was rejected because it learns nothing when training data is complete.
3. **Frozen Operating threshold** per model, applied unchanged to every pattern. Re-picking a
   threshold on injected data is **not** a decision here; the cost it could recover is reported
   (informational, single-input patterns).
4. **A pattern is scorable** for a strategy only if, paired with the complete Readings on identical
   folds, all three hold at the 10:1 default Cost ratio:
   - **Retained saving.** R = (saving over *flag nothing* with the pattern absent) ÷ (saving with
     complete Readings), per fold. The mean R minus the Nadeau–Bengio corrected SE is ≥ **0.90**.
   - **Recall.** The mean absolute recall drop is ≤ **0.10**.
   - **Flag rate.** Alerts per 1,000 Readings are within **×0.5 to ×2** of the complete rate.

   PR-AUC, Brier and similar are reported, not judged.
5. **Monotonic closure.** A pattern is scorable only if every pattern with one fewer absent input is
   scorable too, and the stress veto of rule 1 passes. Anything not measured, or not scorable, is
   **refused**.
6. **Strategy per model.** A simplest-first walk, as in ADRs 0006 and 0007. `reject` is the incumbent
   (R = 0). A challenger replaces the incumbent only if it makes at least one pattern scorable and its
   paired mean per-fold gain in R over the five single-sensor patterns exceeds
   `max(0.05, corrected SE)`. The walk runs per model on identical folds and injected nulls; each
   model keeps its own result. There is **one global strategy per model**, not one per pattern. The
   walk's winner is a measurement; it is not automatically adopted (see the Decision).
7. **Data loss (informational).** For per-input absence rates of 1 %, 5 % and 10 %, independent: the
   share of Readings that are Incomplete (equal to the share a complete-case retraining set would
   drop, 1 − (1 − p)^6) and the share the app refuses under each policy. Also, for the stress group,
   its share of Readings and of failures. None of it changes rules 1 to 6.

## Results
All numbers are train-partition CV (25 folds, 5 repeats of 5). Complete-Reading baselines at the
frozen thresholds (10:1): random forest recall 85.6 %, 36.3 alerts per 1,000, cost 56.2 per 1,000,
saving 83.4 %; LightGBM 83.3 %, 33.1, 61.5, 81.8 %; XGBoost 83.9 %, 40.5, 66.7, 80.3 % (they match
ADR 0007).

**Strategy walk** (rule 6; gain = paired per-fold gain in R over the five single-sensor patterns):

| Model | Challenger | Mean gain in R | Corrected SE | Needed | Replaced? |
|---|---|---|---|---|---|
| Random forest | median over reject | +0.638 | 0.014 | 0.05 | **yes** |
| Random forest | regression over median | +0.012 | 0.010 | 0.05 | no |
| Random forest | native over median | −0.011 | 0.014 | 0.05 | no |
| LightGBM | median over reject | +0.632 | 0.015 | 0.05 | **yes** |
| LightGBM | regression over median | +0.001 | 0.009 | 0.05 | no |
| LightGBM | native over median | −0.077 | 0.015 | 0.05 | no |
| XGBoost | median over reject | +0.650 | 0.018 | 0.05 | **yes** |
| XGBoost | regression over median | −0.002 | 0.013 | 0.05 | no |
| XGBoost | native over median | −1.707 | 0.079 | 0.079 | no |

**By the rules, median imputation wins for all three models**, and nothing more complex beats it.
(The adopted policy is `reject`; see the Decision.) Regression
imputation adds at most +0.012 of retained saving: indistinguishable from zero. Native NaN handling
is worse, and for XGBoost it is dangerous: with speed, wear or torque absent it Flags almost every
Reading (alerts ×22 to ×25 the complete rate, R below −2).

**One input absent** (rule 4; median imputation; R, mean over folds, ± corrected SE):

| Absent input | RF R | RF recall drop | RF Flag ratio | LightGBM R | LightGBM recall drop | LightGBM Flag ratio | XGBoost R | XGBoost recall drop | XGBoost Flag ratio |
|---|---|---|---|---|---|---|---|---|---|
| `Type` | 0.987 ± 0.007 | 0.009 | 1.01 | 0.985 ± 0.006 | 0.005 | 1.07 | 0.993 ± 0.005 | 0.001 | 1.04 |
| Process temperature | 0.925 ± 0.015 | −0.003 | 1.61 | **0.916 ± 0.018** | 0.009 | 1.60 | 0.933 ± 0.024 | 0.000 | 1.46 |
| Tool wear | 0.721 ± 0.037 | 0.245 | 0.65 | 0.722 ± 0.038 | 0.237 | 0.65 | 0.733 ± 0.042 | 0.239 | 0.59 |
| Air temperature | 0.561 ± 0.033 | 0.284 | 1.50 | 0.535 ± 0.032 | 0.294 | 1.59 | 0.568 ± 0.037 | 0.270 | 1.43 |
| Rotational speed | 0.520 ± 0.026 | 0.220 | 2.48 | 0.516 ± 0.031 | 0.246 | 2.29 | 0.525 ± 0.038 | 0.217 | 2.20 |
| Torque | 0.463 ± 0.033 | 0.431 | 0.75 | 0.473 ± 0.034 | 0.424 | 0.65 | 0.491 ± 0.042 | 0.404 | 0.71 |

(The full table, with every strategy and the stress test, is in the notebook.)

PR-AUC relative drop, reported only: `Type` 5 %, 3.6 %, 4.5 % (forest, LightGBM, XGBoost); process
temperature 27 %, 34 %, 36 %; the other four sensors 22 % to 62 %. The decision plan's 10 % PR-AUC rule
would therefore have allowed `Type` alone.

**Scorable patterns after closure and the stress veto** (median imputation):

| Model | Scorable (everything else refused) | Of 63 |
|---|---|---|
| Random forest | `Type`; process temperature | 2 |
| LightGBM | `Type` | 1 |
| XGBoost | `Type`; process temperature; `Type` + process temperature | 3 |

No pattern that removes tool wear, air temperature, rotational speed or torque passes for any model
or strategy. For patterns with two inputs absent, the best retained saving is 0.913 (forest, `Type` +
process temperature, ± 0.014), 0.906 (LightGBM, same pattern, ± 0.019) and 0.926 (XGBoost, same
pattern, ± 0.022); only XGBoost's clears the rule.

**Knife-edges** (the rules were applied as written, but a reviewer should know):
- LightGBM, process temperature absent: R − SE = 0.916 − 0.018 = **0.898**, against 0.90. It misses
  by 0.002, so LightGBM scores only `Type` while forest and XGBoost also score process temperature.
- Random forest, `Type` + process temperature absent: R − SE = 0.913 − 0.014 = **0.899**. It misses
  by 0.001, while XGBoost's R − SE = 0.926 − 0.022 = 0.904 passes.
- Process temperature passes for forest and XGBoost despite a 27 % to 36 % PR-AUC loss. It passes
  because the cost at the frozen threshold is kept: recall does not fall, but the Flag rate rises by
  about 1.5 to 1.6 times (more alerts, mostly false). The Flag-rate test (×2) is deliberately loose;
  this is the case where it is least comfortable.

The differences between models here are within noise. They are recorded as the rule's outcome, not as
a finding that the models differ in robustness.

**Informative stress** (torque absent only for the top-speed quartile; means over folds, high-speed
rows only; median imputation against the same rows complete):

| Model | Recall (complete → median) | Alerts per 1,000 | Saving over *flag nothing* |
|---|---|---|---|
| Random forest | 73.5 % → 75.8 % | 19.9 → 57.2 | 69.7 % → 50.6 % |
| LightGBM | 69.1 % → 70.9 % | 18.1 → 41.6 | 66.0 % → 54.8 % |
| XGBoost | 71.9 % → 69.2 % | 23.5 → 51.1 | 65.9 % → 46.2 % |

The stress fails for every model (retained saving is far below 0.90), but it changes nothing:
`Torque` already fails on all rows. It would only have mattered if `Torque` had passed.

**Recoverable cost** (informational; median imputation; one input absent; frozen threshold against a
threshold re-picked on the same injected scores, so an optimistic upper bound; cost per 1,000
Readings in False-alarm units; complete-Reading cost 56.2, 61.5 and 66.7):

| Absent input | RF frozen → re-picked | LightGBM frozen → re-picked | XGBoost frozen → re-picked |
|---|---|---|---|
| Torque | 207.8 → 166.2 | 207.7 → 174.5 | 205.5 → 168.5 |
| Rotational speed | 191.9 → 173.9 | 195.9 → 191.5 | 196.1 → 183.9 |
| Air temperature | 180.4 → 160.9 | 190.3 → 184.0 | 184.5 → 181.6 |
| Tool wear | 134.7 → 132.9 | 138.3 → 135.7 | 139.1 → 135.3 |
| Process temperature | 77.4 → 75.6 | 84.9 → 78.7 | 85.4 → 83.3 |
| `Type` | 59.9 → 58.9 | 65.7 → 63.7 | 68.5 → 66.1 |

Re-picking recovers little for `Type`, process temperature and tool wear (about 1 to 6), and a lot for
torque (33 to 42), speed and air temperature (up to 20). Yet even the re-picked cost of any sensor
other than process temperature leaves R at about 0.75 or lower (for example forest torque 166.2 gives
R = 0.61), so **a pattern-specific threshold would not make any refused sensor scorable**. The
threshold stays frozen.

**Data loss: the cost of refusing** (rule 7; per-input absence rate assumed, independent):

| Rate per input | Readings Incomplete (also lost to complete-case retraining) | Refused: `reject` for all | Refused: RF policy | Refused: LightGBM policy | Refused: XGBoost policy |
|---|---|---|---|---|---|
| 1 % | 5.9 % | 5.9 % | 3.9 % | 4.9 % | 3.9 % |
| 5 % | 26.5 % | 26.5 % | 18.8 % | 22.6 % | 18.5 % |
| 10 % | 46.9 % | 46.9 % | 35.0 % | 41.0 % | 34.4 % |

Under independent absence, a 5 % per-input rate makes about a quarter of Readings Incomplete, so a
complete-case retraining set would lose about a quarter of its rows. **Refusal at the sensors is
expensive and unavoidable**: even the best policy still refuses 18 % to 23 % of Readings at 5 %,
because any absent sensor other than process temperature is refused. The median policy recovers 4 to 8
points at 5 % (forest 7.7, LightGBM 3.9, XGBoost 7.9), almost all from `Type` and process temperature
(LightGBM only `Type`).

**Informative-missingness sizing.** The top-speed quartile (where torque would drop out under the
stress) is 25.0 % of train Readings but only 14.3 % of failures (failure rate 1.9 % against 3.4 %
overall). If missingness followed speed, a complete-case retraining set would lose few failures; it
would not be a bias towards removing the failure-rich Readings. This is a sizing check on one
hypothetical mechanism, not evidence about the real one.

**Simplicity against data loss.** `reject` is the simplest policy and costs nothing to build. Median
imputation would cost one pipeline step (fitted on training medians and the modal `Type`, covered by
tests) and one per-model sidecar, and would recover 4 to 8 points of Readings at 5 % per-input
absence, almost all from `Type` and process temperature. The walk's gain in R (about +0.64) is large
against its 0.05 margin, but it comes mostly from sensors that still end up refused, and it rests on
knife-edge outcomes. **`reject` is adopted for all three models**: the data loss it accepts is real
(at 5 % per-input absence about a quarter of Readings are refused), but the part imputation could
recover is small, and the evidence for it is not strong enough to build on yet. What would change
that is measured in production, not argued here (Decision, health metrics).

## Decision
Accepted:
- **Scope.** Six inputs can be absent (`Type` and the 5 sensors). An unusable value (not a number, not
  finite, or physically impossible) is treated as absent; the inference layer converts it. Plausibility
  ranges are not decided here.
- **Adopted strategy: `reject`, for all three models.** The rules' walk picks median imputation
  (Results), and that result is kept as a measured reference; it is not adopted. Native NaN handling
  and regression imputation are not adopted either.

  | Model | Strategy | Scored | Refused |
  |---|---|---|---|
  | Random forest (default) | `reject` | complete Readings only | every Incomplete reading |
  | LightGBM | `reject` | complete Readings only | every Incomplete reading |
  | XGBoost | `reject` | complete Readings only | every Incomplete reading |

  A Reading with no input absent is scored exactly as before (bit-identical).
- **A Refused reading gets no Failure risk and no Flag.** The app tells the engineer which inputs are
  needed. It never scores a Reading on a guess, so there is no "reduced reliability" Failure risk to
  present; ADR 0009 words the refusal only.
- **The policy sidecar is `<model>.missing.json`**, separate from `OperatingPoint` (which is left
  untouched). It is a `MissingDataPolicy` (`evaluation/missingness.py`): model, adopted strategy
  (`reject`, so nothing is scorable), Cost ratio, and a **non-scoring reference**: the strategy that
  was measured (`median`) and, for each pattern it scored acceptably, its retained saving ± SE,
  recall drop and Flag-rate ratio. The reference lets the health monitor tell a *Recoverable
  refusal* from one no imputation would fix. It is regenerated with each refit and never committed
  (`models/`). Writing it is the training command's job and a later ticket.
- **Assessment record for every Reading** (from `inference/`; wording and layout belong to ADR 0009):
  - `status`: `complete` or `refused`;
  - `missing_inputs`: each absent input with the reason (absent, not a number, not finite,
    out of range);
  - `refusal_reason`;
  - `recoverable`: whether the pattern is in the sidecar's reference (a Recoverable refusal).
- **A structured log of every Reading, including refused ones, is a requirement**, written by
  `inference/`: the assessment record, a timestamp, the model, a source identifier (machine or
  line) if the input has one, and, for each *observed* input, only its band against the training
  quartiles (never the value). It feeds a dedicated **data-pipeline health tab** in the dashboard (a
  requirement from the project owner). The tab's layout and the monitoring ticket are ADR 0009/0010's;
  this ADR fixes what must be logged and the metrics below.
- **Imputation-readiness metrics on the health tab** (per model, daily and over a rolling four
  weeks). They exist so the team invests in imputation only when it pays:

  | Metric | Definition | What it answers |
  |---|---|---|
  | **Refusal rate** | refused ÷ all Readings | how much data the policy loses |
  | **Recoverable refusal rate** | Recoverable refusals ÷ all Readings | how much the measured imputation would win back |
  | **Forgone saving** | Σ over Recoverable refusals of the pattern's retained saving R ÷ all Readings, as a share of the model's complete-Reading saving | the value of imputation in the ADR 0007 metric |
  | **Avoidable missed failures** | Recoverable refusals × prevalence × complete recall, per 10,000 Readings (training prevalence 3.4 % and the model's complete recall until labelled outcomes exist) | the same in engineers' terms |
  | **Unrecoverable refusal rate** | (refused − recoverable) ÷ all Readings, by absent input and by pattern | whether imputation is even the lever |
  | **Source concentration** | share of refusals from the five most-refusing sources | one broken sensor, or a spread-out problem |
  | **Observed-input skew** | share of refused Readings whose observed inputs fall in each training-quartile band, against the same share for complete Readings; once outcomes join, the failure rate among refused Readings | whether absence is informative |

  **Review triggers** (defaults set by the project owner, not measured; the plant owns them and can
  change them; they live in `configs/default.toml` under `[pipeline_health]`):
  1. **Build imputation** (a new ADR that reruns notebook 04 with the observed rates in place of
     the assumed 1 %, 5 % and 10 %) when the **Recoverable refusal rate is at least 10 % for two
     consecutive weeks**, with at least 1,000 Readings a week, and **source concentration is below
     50 %**. At 10 %, the forgone saving is about 10 % of the model's saving and about 29 Machine
     failures per 10,000 Readings go unflagged (10 % × 3.4 % prevalence × about 86 % recall). The
     plant decides whether that is worth the effort: build when forgone saving, valued
     in the plant's real Cost ratio, exceeds the cost of building and maintaining the imputer.
  2. **Fix the sensor, not the model** when source concentration is 50 % or more: a few sources
     refusing is a maintenance job.
  3. **Imputation is not the lever** when the **Unrecoverable refusal rate is at least 10 % for
     two consecutive weeks**: the absent inputs are the ones the CV showed no imputation can cover.
     The options are sensor reliability or training on incomplete data (a new ADR); not loosening the
     margins.
  4. **Absence looks informative** when the observed-input skew shows refused Readings concentrated
     in one band (for example the top speed quartile holding more than half of them), or, once
     outcomes are known, a failure rate among refused Readings more than twice that of complete
     Readings. The CV here cannot see this; it reopens this ADR.
- **Code and config boundaries.** The injection harness, strategies and degradation metrics live in
  `src/predictive_maintenance/evaluation/missingness.py` and
  `preprocessing/imputation.py`, with mirrored tests; `notebooks/04_missing_data.py` only imports from
  them (this follows the amended code boundary and overrides the decision plan's allowance for a
  notebook-resident harness). The imputation code stays in the package because the measurement and
  the reference need it, but **no production `Pipeline` includes an imputation step** under `reject`
  (`Candidate.make_pipeline(missing_strategy=None)`); if imputation is adopted later it is a step
  inside the serialized `Pipeline`, never in `app/`. Margins, the stress quantile, the assumed rates
  and the strategy list live in `configs/default.toml` under `[missing_data]`, documented in
  `configs/README.md`.
- **ADRs 0004, 0005, 0006 and 0007 are not edited.** The thresholds and models they accepted do not
  move.

## Consequences
- **ADR 0009** must show a refusal clearly, say which inputs a Refused reading needs, and not offer a
  Failure risk or Flag for it. It must not invent a threshold or a fallback score.
- **A Refused reading is the common failure case.** At a 5 % per-input absence rate about a quarter
  of Readings (26.5 %) would be refused; the measured imputation would have recovered 4 to 8 points of
  that. The health tab's metrics decide whether real rates make that worth building. If they show
  unrecoverable refusals instead, retraining on incomplete data (a new ADR) is the lever, not loosening
  these margins.
- **Dropping Incomplete readings is visible, not silent.** Every refusal is logged and shown, which is
  the owner's requirement for the health tab.
- **The measurement is not wasted.** Its median reference sits in the sidecar. Revisiting the decision
  means rerunning notebook 04 with the observed rates and, if adopted, a new ADR that supersedes this
  one's decision (this ADR, once Accepted, is not edited).
- **The knife-edge outcomes no longer matter for the policy** (LightGBM's process temperature misses by
  0.002, the forest's `Type` + process temperature by 0.001). They still shape the reference, so a
  reviewer should read them as noise.
- **The retained-saving rule depends on the 10:1 Cost ratio** (a placeholder, ADR 0007). A different
  ratio changes R and may change the reference. The notebook is rerun and the sidecar regenerated;
  the code does not change.
- **The informative-missingness case is untested.** If the plant's sensors drop out for reasons
  related to failure (a sensor fails *because* the machine is about to), this ADR's CV cannot show it.
  The health tab's observed-input skew and refused-failure-rate metrics are how the plant would
  find out.
- 63 patterns × 3 strategies × 25 folds × 3 models is a few minutes of CPU, so the harness can rerun
  whenever a model changes.

Not decided here: app wording and layout (ADR 0009); deployment and the `xgboost` vs `xgboost-cpu`
swap (ADR 0010); plausibility ranges for each sensor; retraining on incomplete data; model tuning; the
per-failure-mode check; refitting on the full data after the single test evaluation; the real Cost
ratio; writing the sidecar files and the log; the data-pipeline health tab's layout and the code that
computes the metrics and triggers.
