# 0009. App output

- Status: Accepted
- Date: 2026-10-13
- Deciders: Suhas (design agreed in a grill session 2026-10-13; accepted 2026-10-13); drafted by an agent
- Amended: 2026-10-13, before acceptance. The framework decision changed from "Streamlit" to a gated
  order (marimo WebAssembly on GitHub Pages, then molab, then Streamlit Community Cloud), on the
  owner's instruction after the inspection in Results. The owner then **chose marimo WebAssembly on
  GitHub Pages as the direction**, accepting the compromises listed under "UI framework and host
  order" for now; molab and Streamlit are evaluated only if those compromises prove unacceptable.

## Context
The spec (§4.3, §6, §8) asks the app to "choose and clearly label an output… explain the meaning and
limitations", to use the serialized artifacts without reimplementing preprocessing, and to explain
the output "sufficiently for a maintenance engineer… without overstating certainty". This ADR decides
what the app shows an engineer for **one Reading**, in what words, and what it must never claim. It
also decides the content of the data-pipeline health tab, and the **UI framework and the order in
which hosting options are tried**, because they limit what the screen and the log can do. The host
itself, the comparison of marimo against Streamlit and the monitoring scope for a static site are
ADR 0010 (brief at the end of this ADR).

Fixed by accepted ADRs:
- Target `Machine failure`; `Type`, the 5 sensors and 3 engineered features as inputs; failure modes
  never features ([0004](0004-target-and-features.md)).
- Stratified 70/30 holdout, scored once for the single model finally chosen, never touched here
  ([0005](0005-split-and-validation.md)).
- Three offered models, random forest (default), LightGBM and XGBoost, unweighted and uncalibrated,
  each its own `Pipeline`; the low end of the Failure risk is coarse, never "0 %"
  ([0006](0006-model-family.md)).
- A **binary Flag**, one **Operating threshold** per model at the assumed 10:1 **Cost ratio** (forest
  0.218, LightGBM 0.134, XGBoost 0.075; about 36, 33 and 41 Flags per 1,000 Readings), a 1 % **Risk
  floor**, and "describe a Flag by recall and precision, not a raw probability"
  ([0007](0007-metric-and-threshold.md)). Under the matched library pins (Results) XGBoost's line is
  0.0727 and its Flags about 40 per 1,000 (ADR 0011); the app uses the regenerated figures.
- Policy `reject` for all models: an **Incomplete reading** is a **Refused reading**, with no Failure
  risk and no Flag; the inference layer returns an assessment record and logs every Reading; seven
  pipeline-health metrics and four review triggers ([0008](0008-missing-data-policy.md)).
- The UI calls `inference/` only; no preprocessing or threshold logic in `app/` (AGENTS.md, spec §4.3).

Vocabulary follows [`GLOSSARY.md`](../../GLOSSARY.md). This ADR adds **Example reading** and sharpens
**Refused reading** (shown as "Not assessed", never as "no Flag").

**The decision plan's "App output" row is stale** (three bands, calibration). ADR 0007 shipped no
bands (the band check failed for every model) and ADR 0006 applied no calibration. The plan is not
edited here.

Evidence for one decision only (Failure mode): [`notebooks/05_app_output.py`](../../notebooks/05_app_output.py),
built on `predictive_maintenance.evaluation.modes` (tested). **The holdout was discarded at load time
and never scored**; the numbers are train-partition CV. A second piece of evidence, for the
framework order only, is a one-off inspection of marimo's WebAssembly runtime (Results). Everything
else is design, not measurement.

### Disclosures
- **The framework inspection ran in Node, not in a browser.** It used Pyodide 314.0.0 (the version
  marimo 0.25.1 ships), three models fit in the project's own environment on the train partition,
  and 1,000 train Readings. Load time, memory and browser behaviour were unmeasured at that point;
  the later browser check is in Results. The scripts were
  scratch files outside the repo; the method is in Results so it can be repeated, and a repeatable
  smoke test is a ticket (gate W2).
- **Gates W1 to W6 were written after the inspection**, so W1 to W3 describe what was measured
  rather than predict it. W4 to W6 are unmeasured.
- **Facts about molab and Streamlit Community Cloud come from vendor documentation and memory, not
  tests.** None is relied on here beyond the order the owner chose; ADR 0010 verifies them.
- **The earlier draft chose Streamlit "on the owner's preference".** The owner then asked for marimo
  WebAssembly first. That is a change of preference plus new evidence, not a result of a
  measurement that favours either framework.
- **The mode rule was fixed before the numbers.** The margins (PR-AUC ≥ 0.30, lift ≥ 5, one corrected
  SE) were written into `configs/default.toml` (`[failure_modes]`) and the notebook's introduction
  before the run. They are judgement calls and can be challenged. The notebook was run twice with
  identical results; the second run only widened the printed table.
- **The mode models use ADR 0004's features, chosen for `Machine failure`**, and the default forest.
  No per-mode feature or model search was done.
- **The mode check's standard error understates the uncertainty for rare modes.** It reflects fold
  assignment across 5 repeats, not the sampling noise of 13 RNF positives.
- **The three near-perfect modes say something about this dataset, not about a plant** (Results).
- **The option criteria (C1 to C5) were written while drafting this ADR, after the design choices
  were agreed in the grill session**, so they describe the choices rather than select them. The
  design choices (wording, rounding, content of the health tab) are a human's judgement, not a
  measurement. There is no number that proves them right, so the decision rule is a set of review
  criteria (Decision).
- **The comprehension check is a self-review by the project owner**, who wrote the wording and
  will recognise it too readily. No independent reader has seen the screens. This is weaker than
  the check this ADR first proposed (two or three outside readers) and is the owner's explicit
  choice for now.
- **The Flag's recall and precision are train-partition CV figures**, slightly optimistic for the
  reason given in ADR 0007 (the threshold is picked and judged on the same CV scores). The unbiased
  figure is the single holdout score, which does not exist yet.

## Options considered
**Criteria, as agreed.** An option must: (C1) not overstate certainty; (C2) be usable by
an engineer with no ML knowledge, and say what to do next; (C3) agree with ADRs 0006 to 0008; (C4) be
testable without a UI; (C5) show no number that cannot be traced to a stored artifact or a stated
measurement.

1. **Headline.** (a) the Flag only; (b) the Flag plus a secondary Failure risk; (c) the Failure risk
   only; (d) three Risk bands (the stale plan). (c) fails C1 and C3: a raw risk invites reading
   0.1 on one model against another's line, and ADR 0007 says describe a Flag, not a probability.
   (d) fails C3: no model passed the band check. **(b)** is chosen: the Flag is the decision
   ADR 0007 optimised, and the Failure risk lets an engineer order two unflagged Readings (the
   glossary: "an aid for prioritization").
2. **Failure risk display.** (a) whole percent with the model's own Flag line beside it; (b) a coarse
   5 % grid; (c) no number; (d) words relative to the line ("just below"). (b) shows "about 20 %" on a
   Flagged and an unflagged Reading when the line is 21.8 %, which looks like a bug (C2). (d) is an
   invented band (C3). (c) throws away the ordering. **(a)** is chosen, labelled a rough estimate,
   with rules in the Decision.
3. **Source of "what the Flag means".** (a) extend the `OperatingPoint` sidecar; (b) a new
   `<model>.operating-metrics.json`; (c) hard-code the figures in text. (c) fails C5, since figures
   would not follow a refit. (a) changes a format ADR 0007 fixed. **(b)** follows ADR 0008's
   `.missing.json` precedent and leaves `OperatingPoint` as it is.
4. **Refusal.** (a) a distinct "Not assessed" state; (b) the "no Flag" state with a note. (b) fails
   C1: an unscored machine would look like a machine that passed. **(a)** is chosen.
5. **Per-Reading explanation.** (a) none; (b) feature contributions (needs `shap` or a hand-rolled
   method, so a dependency decision, and the decision plan says "no `shap`"); (c) nearest historical
   Readings; (d) a static global importance chart. (b) and (c) fail C1: a contribution on a forest or
   booster reads as a cause, and ADR 0006 notes the model says nothing about *why*. **(a)** is chosen;
   (d) is a cheap, honest later addition (Not decided here).
6. **Failure mode as a secondary output.** (a) measure per-mode predictability first (the open item
   of ADRs 0004 and 0006); (b) decide without it. **(a)** is chosen. What the result can decide is
   limited (Decision).
7. **Example readings.** (a) manual entry only; (b) entry plus Example readings from the train
   partition, with the recorded outcome shown quietly; (c) the same with the outcome hidden. (c)
   was the first proposal. The owner chose **(b)**: showing the outcome is more candid, so long as the
   screen says the model has seen the Reading (C1).
8. **UI framework and host order.** (a) Streamlit on Streamlit Community Cloud (the earlier draft's
   choice); (b) a marimo app running in the browser (WebAssembly) on GitHub Pages; (c) a marimo app
   on molab, running server-side; (d) Gradio, Dash or Flask. (d) adds code or a dependency for no
   gain. (b) has no backend to run, so nothing to keep alive and no server-side log; its costs are
   the Pyodide version gap and a per-browser log (Results, Consequences). (c) runs the same marimo
   notebook on a real Python server, so the version gap disappears, but it depends on a third-party
   service. (a) is the host the spec names, already has an entry file (`streamlit_app.py`) and a
   runtime dependency (`streamlit>=1.65.0`), and keeps the project's own environment, but needs a
   server-side deployment and its artifacts. **The owner's order is (b), then (c), then (a)**: try
   each only if the one before fails its gate (Decision). The `Presentation` seam keeps all three
   cheap to switch between.

## Results
**Framework inspection: marimo WebAssembly** (Node, Pyodide 314.0.0, wheels from the Pyodide CDN;
the three offered models, unweighted, fit in the project environment on the train partition and
pickled with joblib; then loaded in Pyodide and scored on the first 1,000 train Readings, compared
with the native scores; Flags at the ADR 0007 lines 0.218, 0.134 and 0.075):

| Package | Project lock | marimo's Pyodide |
|---|---|---|
| scikit-learn | 1.9.1 | 1.8.0 |
| xgboost | 3.4.1 | 2.1.4 |
| lightgbm | 4.7.0 | 4.6.0 |
| polars | 2.0.0 | 1.33.1 |
| joblib | 1.6.0 | 1.5.3 |
| numpy | 2.5.4 | 2.4.3 |

| Model | Artifact | Max abs score difference | Flag disagreements (of 1,000) | Native Flags |
|---|---|---|---|---|
| Random forest | 4.6 MB | 0 | 0 | 27 |
| LightGBM | 0.36 MB | 3e-20 | 0 | 27 |
| XGBoost | 0.18 MB | **0.66** | **47** | 28 |

- The package (including `polars` frames and the preprocessing) imports and runs in Pyodide, even
  though `pyproject.toml` asks for `polars>=2.0.0` and Pyodide has 1.33.1. All five runtime
  libraries are built for Pyodide.
- scikit-learn warns on every unpickle ("from version 1.9.1 when using version 1.8.0 … may lead to
  breaking code or invalid results"). The forest and LightGBM scored identically here; that is a
  property of these two formats on this data, not a guarantee. **The XGBoost pickle is unusable across
  these versions** (47 Flags differ out of 1,000).
- Runtime wheels downloaded by the browser on a first visit are about 41 MB (polars 18 MB, scipy
  14 MB, scikit-learn 4.3 MB, numpy 2.8 MB), plus the package. Loading them took 7 s from local disk
  in Node; browser load time was not measured.
- **Reading:** marimo WebAssembly can run this system **only with artifacts fit in a Pyodide-matched
  environment**. Server-side hosts (molab, Streamlit) do not have this gap.
- **Matched-version check** (a throwaway Python 3.12 environment with Pyodide's scikit-learn, xgboost,
  lightgbm, polars, joblib, numpy and scipy; the project's own test suite; repo dependencies
  untouched): **320 of 321 tests pass.** The one failure is
  `test_class_weighting_raises_predicted_risk_for_the_rare_class[random_forest]`, a seed-sensitive
  check on tiny data (the weighted forest's mean in-sample risk 0.0971 against 0.0995 unweighted).
  It is not a break in the code, but it shows that **a forest fit under other library versions draws
  slightly differently**, so thresholds and sidecars will not reproduce exactly.
- **Option A applied (owner's choice, 2026-10-13).** `pyproject.toml` now pins `scikit-learn==1.8.0`,
  `xgboost==2.1.4`, `lightgbm==4.6.0`, `polars==1.33.1` and `joblib==1.5.3`, with `numpy==2.4.3` and
  `scipy==1.17.1` as uv constraints; `uv.lock` is re-locked. Two changes followed, neither altering
  behaviour: `_regress` in `preprocessing/imputation.py` passes `np.intp` index arrays (the numpy 2.4
  stubs rejected lists), and the class-weighting test no longer asserts that a weighted forest's mean
  risk is higher. Under scikit-learn 1.8.0 it is **lower on all 8 seeds tried**, because fully grown
  trees are insensitive to class weights; the test now checks the configuration
  (`class_weight` None or `"balanced"`) for the forest and keeps the risk check for the three
  boosting models. `just check` passes (321 tests, mypy strict, ruff).
- **Evidence re-run under the matched pins** (notebooks 01, 02, 03, 03b and 04, same seed; warnings
  filtered; compared figure by figure with the earlier outputs):
  - **Random forest and LightGBM: unchanged.** The ADR 0007 operating thresholds (0.218 and 0.134 at
    10:1; 0.242 and 0.150 for the forest at 5:1 and 20:1) and their recall, precision and Flags per
    1,000 agree to six decimals.
  - **XGBoost moves.** The 10:1 threshold is **0.0727, not 0.075** (5:1: 0.257 against 0.242; 20:1:
    0.0117 against 0.0206). At the 10:1 threshold: recall 83.4 % (was 83.9 %), precision 70.6 % (70.1 %),
    40.0 Flags per 1,000 (40.5). Its PR-AUC is 0.866 (0.862) and Brier 0.00903 (0.00939). The
    conclusions in ADR 0007 do not change: the forest stays first, XGBoost stays the costliest
    (11.8 per 1,000 dearer than the forest against 10.5), and no model is cheaper.
  - **Hist-gradient-boosting, the weighted variants and the baselines move in the second or third
    decimal** (HistGradientBoosting PR-AUC 0.880 against 0.877 in notebook 03, 0.881 against 0.885 on
    the engineered-feature check in notebook 01). **One rank changes below the top:** the weighted
    forest (0.883) now sits above unweighted LightGBM (0.882), where it was 0.880. The forest stays
    first, the weighted variants still do not beat their unweighted counterparts by the margin the
    earlier ADRs required, and the offered three are the same three, so ADRs 0005 and 0006 are
    not contradicted.
  - **Notebook 04 stopped on its own guard**, which checks the re-derived thresholds against the
    accepted figures to 0.0005; XGBoost's 0.0727 failed it. A throwaway copy with that one constant
    changed to 0.073 completed. Under it the ADR 0008 results are the same for the forest and
    LightGBM, and for XGBoost the figures move in the second or third decimal with **no verdict
    flipping** (reject against median, regression and native; the five single-sensor patterns). ADR
    0008's decision (reject for all models) therefore stands. The guard now checks 0.073 for XGBoost.
  - **Reading:** nothing in ADRs 0004 to 0008 is contradicted. **ADR 0007's quoted XGBoost threshold
    (0.075) and its figures (84 % and 70 %, 40.5 Flags per 1,000) are replaced by the matched-version
    figures in [ADR 0011](0011-xgboost-figures-under-matched-pins.md)** (Proposed with this ADR);
    this ADR quotes them wherever it shows XGBoost.
- **Browser check of the matched build (owner, 2026-10-13).** A throwaway page (outside the
  repository) installed the project wheel in marimo's Pyodide, loaded the three matched-version
  artifacts and scored 1,000 train Readings, comparing with the native run. Chrome 153 on Windows,
  site served from localhost (the 5 MB of models and the wheel were local; Pyodide and its packages
  were not). Reported: **all three models load and score; scores equal the native run within 1e-9
  and every Flag agrees, XGBoost included; page start to first cell 19.0 s; Python heap 325 MB.**
  Native is Python 3.12 and the browser Python 3.14.2, so the pickles also crossed a Python minor
  version. Only the summary was reported, not the per-model table, and it is not stated whether the
  19.0 s was a cold or a warm load. One run, one browser, one connection: it shows the build works,
  not how load time varies.

**Per-mode predictability** (one forest per mode, six inputs only, train partition, repeated
stratified 5×5 CV, mean over 5 repeats of out-of-fold PR-AUC ± corrected SE; 7,000 Readings):

| Mode | Positives | Prevalence | PR-AUC ± SE | Lift | ≥ 0.30 | Lift ≥ 5 | − 1 SE > prevalence | Predictable |
|---|---|---|---|---|---|---|---|---|
| TWF | 33 | 0.47 % | 0.083 ± 0.010 | 17.7 | no | yes | yes | **no** |
| HDF | 73 | 1.04 % | 0.992 ± 0.003 | 95.1 | yes | yes | yes | **yes** |
| PWF | 72 | 1.03 % | 0.998 ± 0.0004 | 97.0 | yes | yes | yes | **yes** |
| OSF | 71 | 1.01 % | 0.995 ± 0.002 | 98.1 | yes | yes | yes | **yes** |
| RNF | 13 | 0.19 % | 0.0025 ± 0.0004 | 1.3 | no | no | yes | **no** |

HDF, PWF and OSF are predicted almost perfectly from the six inputs. TWF is too weak (PR-AUC 0.083,
below the 0.30 floor even though its lift is 17.7). RNF, a failure the dataset names "random", is at the
level of chance (lift 1.3). The pass on three modes is **a property of this synthetic dataset, where
those modes look close to deterministic functions of the inputs**; it is not evidence that a real
plant's causes could be read from its sensors. That was not investigated here. TWF and RNF failures
(33 + 13 of the 237 train failures, with overlaps) would have no cause shown even if the three were.

## Decision

### What the engineer sees for one Reading
- **Headline: the Flag**, from the selected model's `OperatingPoint` at the 10:1 default. Three states:
  - **Inspect** (Flagged);
  - **No inspection flagged** (not Flagged);
  - **Not assessed** (Refused reading).

  The Flag is decided by `inference/` on the **unrounded** score, never from the displayed number.
- **Secondary line: the Failure risk**, labelled "rough estimate, not calibrated".
  - Below the Risk floor: "under 1 %".
  - From 1 % to 95 %: "about N %", the nearest whole percent, halves up.
  - Above 95 %: "over 95 %". Never "0 %" or "100 %".
  - Beside it: "this model flags from N %" (the model's own line, rounded the same way), and a fixed
    note that a risk within a percent of the line can go either way and that a higher number on
    another model does not mean more risk.
- **What the Flag means**, in plain words from the model's operating-metrics sidecar (CV on the train
  partition, labelled as such), integers with "about": of every 100 machine failures, about N were
  Flagged; of every 100 Flags, about M were real; about K Readings in 1,000 are Flagged. A Flagged
  result adds "about 100 − M of every 100 Flags are false alarms"; a not-flagged result adds "about
  100 − N of every 100 failures were not flagged", so the screen never reads as a clearance. Example
  for the default forest: 86 of 100 failures Flagged, 80 of 100 Flags real, 36 per 1,000.
- **Sensitivity**, read-only, in an expander: recall, precision and Flags per 1,000 at 5:1, 10:1 and
  20:1 for the selected model, **metrics only, no thresholds** (so LightGBM's 20:1 line, below the
  floor, never appears). The forest's 20:1 row is marked "not precise" (ADR 0007: its optimum is not
  flat). The Flag always uses 10:1. The engineer cannot change the ratio.
- **Always on screen with any result:**
  - "Estimated from historical readings."
  - "Synthetic benchmark data, not your plant."
  - "Assumes one missed failure costs as much as 10 false alarms. This is an assumption, not a
    measurement." (the number is read from the config, not typed).
  - "An aid for prioritising inspections, not a guarantee."
  - Every result block names its model.
- **Banned in any rendered text:** "will fail", "safe", "healthy", "OK", "no risk", "0 %", "100 %",
  "probability of failure" as a statement of fact, "predicts failure", "guaranteed", and any accuracy
  claim that does not say which test it comes from. A test scans every rendered state for these.
  (Wording above is illustrative; the final copy is a ticket and must pass the test.)

### Refused reading
- Shown as **Not assessed**, in its own style, never the "No inspection flagged" style.
- It lists each absent input with its reason from `missing_inputs` (absent, not a number, not finite,
  out of range) and says: "No risk or Flag is given. This does not mean the machine is fine. Enter
  the missing values and assess again; if a sensor is offline, treat this machine as unassessed."
- It never shows a Failure risk, a Flag, a threshold or a fallback score.
- `recoverable` is **not** shown to the engineer; it belongs to the health tab. No promise of future
  imputation is made on screen.

### Example readings
- `[app] example_count` (default 10) Readings drawn from the **train partition only**, with the seed
  from the config. **No holdout Reading is ever offered.** How many are recorded failures is the
  selection ticket's job (a stratified draw at 3.4 % prevalence would give none), and needs its own
  key, asked for then.
- After the assessment, a muted line: "Recorded in the data: Machine failure / no failure. This
  Reading was in the training data, so agreement proves nothing." Only the `Machine failure` outcome
  is shown; the five mode flags are not.
- Editing any input clears the outcome line (an input to the `Presentation`, not app logic).

### Model dropdown and artifacts
- Default: random forest. Switching re-assesses the same entered Reading with the new model and shows
  "Model changed: the Flag line and the figures below now belong to <model>". Only the selected
  model's result is shown, never two side by side; `inference/` scores with one model per call, so a
  Failure risk never meets another model's threshold.
- Each model loads from `models/` as one unit: its serialized `Pipeline`, its `OperatingPoint`, its
  `.missing.json`, its **`<model>.operating-metrics.json`** (model, source label `cv_train` or
  `holdout`, and for each cost ratio recall, precision, Flags per 1,000 and false alarms per caught
  failure, with intervals once the holdout is scored) and a **`<model>.manifest.json`** (model name,
  **the versions of scikit-learn, xgboost, lightgbm, polars, joblib and numpy the `Pipeline` was fit
  with**, and the SHA-256 of the other four files). `inference/artifacts.py` checks that all five
  exist, that the names agree, that every hash matches and **that the recorded library versions equal
  the running ones**. A model that fails is **disabled in the dropdown with
  a one-line reason and is never scored with a fallback threshold**. If the default model fails, the
  app shows an error state, not another model. The manifest makes a sidecar left over from before a
  refit, or an artifact fit under other library versions (the XGBoost case in Results), detectable
  without changing the formats of ADR 0007 or 0008. All of these files are written
  by the training command, are never committed (`models/`), and are a ticket.
- **Log sink.** `inference/` writes each assessed Reading (the ADR 0008 fields, no raw values) to a
  sink it is given. Where the sink writes (session memory, a download, a server file) is ADR 0010's.

### Failure mode and explanation
- **No Failure mode output and no per-Reading explanation ship.** The measurement (Results) says
  HDF, PWF and OSF are predictable here and TWF and RNF are not, so a mode output is **worth a
  follow-up ADR**, not shipped in this one. It needs one model per mode, its own Operating
  threshold (ADR 0007's scope), sidecars and a missing-data check, and would cover three of five
  modes, so most Flags would read "cause unknown" for TWF and RNF. The rule that decides "worth a
  follow-up ADR" is in `configs/default.toml` and unchanged by the result.
- Failure-mode flags are never features for `Machine failure`.

### Pipeline-health tab (content, not layout)
Computed in `monitoring/` from the ADR 0008 log, the sidecars and the config; the tab only renders.
Per model, over the windows the log supports: daily and a rolling four weeks for a log that persists,
**the session** for one that does not (ADR 0010 decides which):
- **Shown by default:** Refusal rate; Recoverable and Unrecoverable refusal rates; Source
  concentration; **Flag rate** against the model's expected Flags per 1,000 (from the
  operating-metrics sidecar), over assessed Readings only.
- **Behind a "details" expander:** Forgone saving; Avoidable missed failures; Observed-input skew.
- **Triggers** are shown as status lines ("not triggered", "triggered for N weeks", "not
  evaluable") beside the config value they use (`[pipeline_health]`), for the four ADR 0008
  triggers. A fifth, **Flag rate off expected** (more than 2 times or less than half the expected
  rate, mirroring ADR 0008's band), is a new trigger; its keys are added with the monitoring ticket,
  after asking, and are not decided here. A trigger is "not evaluable" while a week has fewer than
  `min_weekly_readings`; the tab then says "not enough data: N of 1,000 Readings this week" and
  invents nothing.
- **Simulated data, clearly labelled.** A demo has almost no traffic, so a toggle replays a simulated
  log built from train Readings with absent inputs injected at one of `[missing_data]
  assumed_missing_rates`. While it is on, a permanent banner reads "Simulated data, not real usage"
  and shows the assumed rate. It is built and tested in `monitoring/`. Persisting the real log, or
  not, is ADR 0010's; where it is not persisted the simulated log is the only way to see the
  triggers work, and the tab says so.
- **Spec §4.4 metrics** (README, later): refusal rate (data quality) and Flag rate against
  expected (prediction-distribution shift) need no outcomes. Post-outcome performance (Flag
  precision in use) is defined but waits for labelled outcomes.

### UI framework and host order
**Framework: marimo.** The app is a marimo notebook in app mode that renders a `Presentation` and
holds no logic. The same file runs in the browser (WebAssembly) and on a server (molab), so the first
two options share all code. Streamlit stays as the last option behind the existing
`streamlit_app.py`; no dependency is added, changed or removed by this ADR.

**Chosen: marimo WebAssembly on GitHub Pages (option 1).** The owner accepts these compromises for
now: (i) artifacts fit in a Pyodide-matched environment, with thresholds and sidecars regenerated
there; (ii) a per-session log, with no cross-user view; (iii) published model files; (iv) a heavy
first load; (v) a two-model build (forest and LightGBM) if XGBoost cannot pass W2. **Options 2 and 3
are evaluated only if** a gate below fails and no ticket within this ADR fixes it, **or** the owner
records one of (i) to (v) as unacceptable after seeing it in practice. The owner records the outcome
in ADR 0010.

| Option | When it is evaluated |
|---|---|
| 1. marimo WebAssembly on GitHub Pages | Now. Adopted if all gates pass. |
| 2. marimo on molab | If option 1 fails W2, W3 or W6 without a fix, or the owner finds a compromise unacceptable. |
| 3. Streamlit on Community Cloud | If option 2 also fails or is unsuitable. |

1. **marimo WebAssembly on GitHub Pages** is adopted if all of these pass:

   | Gate | Passes when | Status now |
   |---|---|---|
   | W1 Runtime | The package, `polars` frames and all offered pipelines import and score in marimo's Pyodide | **Passed in a browser** (Chrome 153, Windows; Results) |
   | W2 Score equality | For each offered model, artifacts fit in a Pyodide-matched environment give scores equal to the native run (within 1e-9) and the same Flag on a fixed set of train Readings | **Passed in a browser** for all three models, XGBoost included, on 1,000 train Readings (Results) |
   | W3 Size and load | The published site is within the host's limits and the first load is acceptable to the owner | Models 5.1 MB in total, about 41 MB of runtime wheels; 19.0 s to the first cell and 325 MB heap in one run; acceptability is the owner's call | **Passed**: the owner accepted a first load of about 19 s (2026-10-13) |
   | W4 Screen parity | Every state in this Decision (three headline states, model-change notice, expander, banner, tabs) renders with marimo elements, all wording coming from `Presentation` | Expected, not built |
   | W5 Log | `inference/` logs every Reading without raw values to a sink, and `monitoring/` computes the health metrics from it (scope in ADR 0010) | Not built |
   | W6 Repo rules | Artifacts are built and published without committing `models/` to the main branch and without editing an Accepted ADR; any CI change is asked first | Needs a CI change (ADR 0010) |

   A model that fails W2 is disabled in the WebAssembly build (the dropdown rule above); the app then
   offers two models, and ADR 0006 is not edited. If the owner wants all three, W2 is the gate.
2. **marimo on molab** is tried if option 1 fails W2, W3 or W6 and no ticket fixes it. It runs the
   project's own environment, so there is no version gap. Facts to verify in ADR 0010 (unverified
   here): the notebook runs as an app that a viewer can open without an account; the artifacts reach
   it although `models/` is not committed; and the service is stable enough for a graded demo.
3. **Streamlit on Streamlit Community Cloud** is used if options 1 and 2 both fail. It is the host
   the spec names. Facts to verify in ADR 0010 (unverified here): memory and size limits against
   `xgboost`, `lightgbm` and `scikit-learn` (the `xgboost` vs `xgboost-cpu` swap is there); how the
   artifacts and a log reach it; the sleep behaviour of a free app.

**A Pyodide-matched environment** (for option 1) means the artifacts are fit and scored with the
library versions in the Pyodide table in Results, and the manifest records them. Changing the
project's dependency pins for this is a dependency change and is asked before it happens. The
thresholds and sidecars are regenerated by the training command in that environment; ADR 0007's
thresholds (0.218, 0.134, 0.075) were found under the project's own versions. **Re-run under the
matched pins (Results), the forest and LightGBM reproduce exactly and XGBoost's becomes 0.0727**; no
decision in ADRs 0004 to 0008 moved; XGBoost's figures are recorded in ADR 0011.

### Code and config boundaries
Each new module gets a mirrored test file. `app/` renders strings; it contains no threshold,
rounding, wording or preprocessing logic. **`inference/`, `monitoring/` and the `Presentation` code
import no UI framework** (no `marimo`, no `streamlit`), so they run under Pyodide and under
Streamlit; only the entry notebook and `streamlit_app.py` import one. The entry notebook's location
and name are a ticket.
- `inference/assess.py`: the assessment record (ADR 0008) and the Flag decision on the raw score.
- `inference/presentation.py`: maps an assessment, an `OperatingPoint` and the operating metrics to a
  `Presentation` (headline state, risk text, line text, meaning text, caveats, refusal lines, example
  outcome line) and holds the banned-wording list.
- `inference/artifacts.py`: loads and checks the five-file unit.
- `monitoring/health.py`: the metrics and triggers; `monitoring/simulate.py`: the labelled demo log.
- `evaluation/modes.py` (done, tested): the per-mode check; the notebook only imports.
- Config: `[failure_modes]` (`min_pr_auc`, `min_lift`) and `[app]` (`example_count`) are added and
  documented in `configs/README.md`.

### Decision rule: what would make this design wrong
No number measures it. Review criteria, checked by the project owner before moving this ADR to
Accepted (against the spec checklist: "explains its output sufficiently… and does not overstate
certainty"):
1. **Wording.** Every state (Inspect, No inspection flagged, Not assessed, model unavailable, Example
   reading) passes the banned-wording test and carries the always-on caveats. A test enforces it.
2. **No false result.** No path gives a Failure risk or Flag for a Refused reading. "0 %" and
   "100 %" cannot be produced. The Flag is always decided with the selected model's own line.
3. **Comprehension, by self-review** (see Disclosures): shown one Flagged, one unflagged and one
   Not assessed screen, can the owner answer "what does a Flag mean?", "does no Flag mean the machine
   is fine?" and "what do I do with Not assessed?" without "will fail", "safe" or "ignore it".
   Any such answer means the wording is revised first. The outcome goes into this ADR before it is
   Accepted.
4. **Framework gates.** Before ADR 0010 picks a host, the owner records W1 to W6 as pass or fail
   with evidence. A screen state that cannot be rendered, or a Flag that differs between the native
   and the WebAssembly run, fails its gate; it is not worked around in the UI.

**After release**, this ADR is reopened (a new ADR supersedes it) if: Flag precision in use, once
outcomes are known, is under half the CV figure; the Flag rate sits outside 0.5 to 2 times the
expected rate; or an ADR 0008 trigger fires and its remedy changes what the screen can say.

## Consequences
- **A Refused reading is common** (about a quarter of Readings at 5 % per-input absence, ADR 0008),
  so the Not assessed state is one the engineer will see often. Its wording matters as much as the
  Flag's.
- **The risk line can look inconsistent near the Flag line** after rounding; the fixed note and the
  unrounded decision are the mitigation. If engineers misread it, the next step is dropping the number
  (option 2c), not adding precision.
- **The Flag's recall and precision are per model and differ** (forest 86 % and 80 %, LightGBM 83 % and
  85 %, XGBoost 83 % and 71 % under the matched pins; ADR 0007 gave 84 % and 70 %), so switching
  models changes what a Flag promises. The notice and the
  per-model text are how the screen says so.
- **The figures change when the holdout is scored.** The chosen model then shows holdout figures with
  intervals (about ±7 points on recall), and the other two stay on CV, each labelled.
- **Showing the recorded outcome of Example readings is candid but easy to over-read.** The "agreement
  proves nothing" clause is required; without it a demo of correct-looking examples overstates
  accuracy. After the refit on the full data (open since ADR 0005), train-partition Readings are still
  the only legitimate examples.
- **Five sidecar files per model** is more to write and keep in step. The manifest is what makes
  that safe; writing them is the training command's job.
- **The mode result is strong for three modes and means little outside this dataset.** A follow-up ADR
  would need to decide whether it is worth five more models for a synthetic benchmark.
- **No explanation per Reading** means the engineer gets a decision without a reason. This is the
  honest position for these models; it limits trust and is a known cost.
- **The framework choice is conditional.** marimo WebAssembly on GitHub Pages is the chosen
  direction and is adopted only if W1 to W6 pass; W1, W2 and W3 passed (W1 and W2 in a browser, W3
  on the owner's acceptance of a 19 s first load), W4 and W5 are unbuilt and W6 needs a CI decision.
  molab and Streamlit are the planned fallbacks, evaluated only on the conditions in the Decision.
- **A matched build ties the app to Pyodide's versions** (scikit-learn 1.8.0, xgboost 2.1.4, lightgbm
  4.6.0, polars 1.33.1) and freezes the models to them until marimo's Pyodide moves. The pins are
  applied; the XGBoost threshold is 0.0727 there, not ADR 0007's 0.075 (forest and LightGBM
  unchanged), and sidecars are always regenerated in that environment.
- **A static site publishes the model files.** Anyone can download the `Pipeline`s. That is
  acceptable for a synthetic benchmark and would not be for a real plant's model.
- **A static site has no shared log.** Each browser has its own session, so there is no cross-user
  health picture. The health tab then shows session figures and the labelled simulated log; real
  monitoring remains a README design (spec §4.4 asks to define the metrics, not to run them).
- **First load is heavy** (about 41 MB of runtime wheels before any assessment; 19.0 s to the first
  cell in one Chrome run), then cached by the
  browser.
- **`streamlit` stays a runtime dependency and `streamlit_app.py` stays** until ADR 0010 fixes the
  host. If marimo WebAssembly is adopted, removing them is a dependency change and is asked first.
  The `Presentation` seam keeps the logic testable and the framework replaceable.
- **Review is a self-review.** Wording errors a first-time reader would catch may survive. Revisit
  with an outside reader before any real use.

Not decided here: the host actually used, the marimo vs Streamlit comparison, the `xgboost` vs
`xgboost-cpu` swap, where the log is persisted and CI (ADR 0010, brief below); page and tab layout,
the input form (units, `Type` as a dropdown) and sensor plausibility ranges; the final copy of every
message; how many Example readings are recorded failures; the keys for the Flag-rate trigger;
writing the five sidecar files, the log and the monitoring code; a Failure mode output (a follow-up
ADR); a global importance chart; a models-disagree view (Flags only); model tuning; the real Cost
ratio; the single holdout evaluation and refit on the full data; the README's monitoring metrics and
deployment risks; the system-design diagram.

## Brief for ADR 0010
ADR 0010 must contain:
1. **A comparison table of marimo (WebAssembly on GitHub Pages, and on molab) against Streamlit
   (Community Cloud)**, filled from verified facts, with these rows at least: where Python runs;
   control of library versions (the gap in Results); how artifacts reach the app without committing
   `models/`; first-load and cold-start time; log persistence; what monitoring is possible; free-tier
   limits and cost; model and data confidentiality (a static site publishes the files); screen
   features needed by this ADR (tabs, dropdown with notice, expander, banner, upload and download);
   reactivity and the cost of a rerun; spec fit (§4.3 names Streamlit; "or an equivalent service");
   effort and rollback to the next option. It ends with the gate results W1 to W6.
2. **A simplified monitoring scope for a static site**, **confirmed by the owner 2026-10-13**:
   - the **log is per session**, held in memory, with a download button (ADR 0008 fields, no raw
     values) and an upload that `monitoring/` parses in the browser, so nothing leaves it;
   - the **health tab** computes the ADR 0008 metrics and triggers from the session or an uploaded
     log, and shows "not evaluable" below `min_weekly_readings`; the labelled simulated log shows the
     triggers working;
   - **no alerting, scheduler, central store or cross-user view is built.** What a hosted deployment
     would add is described in the README (spec §4.4), not implemented;
   - the §4.4 metrics that need no outcomes (refusal rate, Flag rate against expected) are the
     ones the tab can show; Flag precision in use is defined and waits for labelled outcomes.
3. **Decisions that need the owner first:**
   - **The matched pins: option A is applied** (single environment pinned to Pyodide's versions; see
     Results for the pins, the two code and test changes, and the notebook re-run). Settled with the
     owner: XGBoost's figures are recorded in ADR 0011 and notebook 04's guard checks them; the
     `<model>.manifest.json` is the shared artifact id; `xgboost` against `xgboost-cpu` (2.1.4 pulls
     `nvidia-nccl-cu12`) stays with this ADR 0010.
   - A CI job that builds and publishes artifacts and the wheel (a CI change).
   - Whether the repository must be public for Pages.
   - A repeat of the browser check across browsers and connections, cold and warm, for W3 (one Chrome
     run is done; see Results). The check page is a throwaway, so the repeatable smoke test stays a
     ticket.
