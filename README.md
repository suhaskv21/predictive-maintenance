# Predictive Maintenance

[![CI](https://github.com/suhaskv21/predictive-maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/suhaskv21/predictive-maintenance/actions/workflows/ci.yml)
[![Pages deploy](https://github.com/suhaskv21/predictive-maintenance/actions/workflows/pages.yml/badge.svg)](https://github.com/suhaskv21/predictive-maintenance/actions/workflows/pages.yml)
[![Live app](https://img.shields.io/badge/live%20app-GitHub%20Pages-blue)](https://suhaskv21.github.io/predictive-maintenance/)

**Live app: <https://suhaskv21.github.io/predictive-maintenance/>**

The first load is heavy: the page downloads a Python runtime into your browser, which took about 19 s to
93 s where it was measured (see [Browser evidence](#browser-evidence)). The footer shows the commit the
site was built from. The site is rebuilt only on request (see [Deployment](#deployment-provider-and-limitations)),
so it can lag the `master` branch.

## Problem, users and outcome

A maintenance engineer has one machine's sensor values (product Type, air and process temperature,
rotational speed, torque, tool wear) and must decide whether to inspect it now. This system estimates a
**Failure risk** for that **Reading**, from the AI4I 2020 benchmark data, and raises a **Flag** ("Inspect")
when the risk reaches the model's own **Operating threshold**. The target is the recorded `Machine failure`.
It is an aid for prioritising inspections, not a guarantee.
on any plant.

**How predictions reach engineers.** The engineer opens the URL, enters the six inputs on one screen, and
reads a headline: *Inspect*, *No inspection flagged*, or *Not assessed* (an input is missing or impossible,
so no risk and no Flag are given, and the screen says that this does not mean the machine is fine). Beside
the headline are the Failure risk ("rough estimate, not calibrated"), the model's Flag line, plain words
for what a Flag means (how many real failures are flagged, how many Flags are real, how many Readings per
1,000 are flagged) and always-on caveats. Pushing assessments into a work-order system is a production
addition and is not built ([system design](docs/design/system-design.md)).

**A useful outcome** is that engineers inspect the machines most likely to fail first, accept that some
inspections will find nothing, and never read "No inspection flagged" or "Not assessed" as a clearance.

## Decisions and rationale

| Decision | Choice | Why (ADR) |
|---|---|---|
| Data source | The supplied file `data/raw/ai4i2020.csv` (10,000 Readings, 14 columns, no nulls), committed. Nothing is downloaded. | Small, public and immutable, so the work reproduces offline. It is validated on load by `data/clean`. |
| Target | Binary `Machine failure` (339 of 10,000, 3.39 %). | The spec's default target. Predicting the *mode* of failure is deferred, not rejected ([0004](docs/adr/0004-target-and-features.md)). |
| Features | `Type` (one-hot) and five sensors, plus three engineered features: Power (torque × angular speed), Temperature delta, Wear × torque. All are computed inside the serialized Pipeline, never in the app. | Each engineered feature raised a non-linear model's CV PR-AUC (+0.034, +0.030, +0.019) ([0004](docs/adr/0004-target-and-features.md)). |
| Excluded columns | `TWF, HDF, PWF, OSF, RNF` (the five Failure modes) are outcome fields, so using them to predict `Machine failure` would be leakage. `UDI` and `Product ID` are identifiers; the signal in `UDI` is a row-position artifact (`UDI` 4001 to 5000 fail at 13.1 % against 2.3 % elsewhere) that a new Reading does not have. | Enforced in code (`data/schema.py`) and tested ([0004](docs/adr/0004-target-and-features.md)). |
| Missing data | Strategy `reject`. Any absent or unusable input (absent, not a number, not finite, or physically impossible) means the Reading is **not assessed**: no risk, no Flag, no fallback score. Plausible ranges are in `configs/default.toml` (`[plausibility]`). The training data has no nulls, so no imputer is fitted. | Median imputation was measured as a reference and not adopted; nothing is scored on a guess ([0008](docs/adr/0008-missing-data-policy.md), [0013](docs/adr/0013-input-plausibility-ranges.md)). |
| Split and validation | Stratified 70/30 split made once, before anything is fitted (the 30 % is the **Holdout**, about 102 failures). Model comparison and threshold choice use repeated stratified 5×5 CV on the train partition, with preprocessing fitted inside each fold. The Holdout is scored once. | Stratifying guarantees the failure count in the Holdout. Repeated CV ranks close candidates reliably where a single split cannot ([0005](docs/adr/0005-split-and-validation.md)). |
| Primary metric and threshold | Expected cost per 1,000 Readings at an assumed **Cost ratio** of 10 (one Missed failure counts as 10 False alarms), reported against *flag nothing* and *flag everything*. Supporting: recall, precision, False alarms per caught failure, Flags per 1,000, PR-AUC, ROC-AUC, Brier. The Operating threshold is **0.218**, the median over CV repeats of the cost-minimising threshold, frozen before the Holdout. | PR-AUC ranks models but cannot say where to act. The ratio is an assumption owned by maintenance management and set in `configs/default.toml` ([0007](docs/adr/0007-metric-and-threshold.md)). |
| Model | Random forest (300 trees, unweighted, scikit-learn defaults), serialized with its preprocessing as one `Pipeline`. | See the trade-off below ([0006](docs/adr/0006-model-family.md)). |
| Output shown | A binary Flag (Inspect / No inspection flagged) on the **unrounded** score, a secondary Failure risk ("under 1 %", "about N %", "over 95 %", never "0 %" or "100 %"), the model's own Flag line, and a Not assessed state. No Risk bands (the data could not support a middle band). | A Flag is easy to act on; the risk lets an engineer order two unflagged Readings without mistaking it for a guarantee ([0007](docs/adr/0007-metric-and-threshold.md), [0009](docs/adr/0009-app-output.md)). |
| How to act | *Inspect*: inspect the machine, and expect some Flags to be false alarms. *No inspection flagged*: carry on, but some real failures are not flagged. *Not assessed*: fix the input, or treat the machine as unassessed if a sensor is offline. | Wording is tested against banned phrases ("will fail", "safe", "healthy", "OK", "no risk", "0 %", "100 %", "guaranteed", and others) on every state. |
| Deployment | marimo exported as a WebAssembly site on GitHub Pages (Python runs in the visitor's browser). Limits are listed below. | Free, always on, and needs no server ([0010](docs/adr/0010-deployment.md), scope in [0012](docs/adr/0012-five-hour-submission-scope.md)). |

**Model choice and the interpretability trade-off.** Candidates were compared on repeated CV PR-AUC,
simplest first. Logistic regression is readable but 0.37 PR-AUC lower (about 0.52 against 0.894), too weak to
prioritise maintenance. The random forest, LightGBM and XGBoost are within 0.03 of each other, a gap inside
the selection noise, so the forest was kept as the default. Its cost: no per-Reading explanation (SHAP was
not adopted). Global permutation importance ranks Torque, then Rotational speed, Air temperature, Tool wear,
Process temperature and `Type`, but it ranks inputs, not mechanisms. Balanced class weighting and
resampling were tried and not adopted: weighting did not beat the unweighted model by the margin and
distorted the probability shown. The score is not calibrated, which is why the app calls it a rough estimate.

## Results on the Holdout

Scored once, for one model, on the CI-built artifact (Pipeline SHA-256
`c55639bc96772158806b8201e61d845323508063feef1bb8954157940aef32b3`), on 2026-10-14. The Holdout has 3,000
Readings and 102 Machine failures. The Operating threshold (0.218) and the Cost ratio (10, an assumption)
were frozen before scoring. Intervals are percentile 95 % intervals from 2,000 stratified bootstrap
resamples, in which the threshold is never re-chosen. The committed source is
[`reports/holdout-metrics.json`](reports/holdout-metrics.json).

| Metric | Estimate | 95 % interval |
|---|---|---|
| Recall | 0.863 | 0.794 to 0.922 |
| Precision | 0.830 | 0.766 to 0.899 |
| Flags per 1,000 Readings | 35.3 | 31.7 to 39.0 |
| False alarms per caught failure | 0.205 | 0.112 to 0.306 |
| Expected cost per 1,000 Readings (False-alarm units, ratio 10, assumed) | 52.7 | 31.3 to 76.3 |
| PR-AUC | 0.914 | 0.867 to 0.955 |
| ROC-AUC | 0.986 | 0.971 to 0.996 |
| Brier | 0.00696 | 0.00504 to 0.00909 |

In counts (worked out from the table, not separately reported): of 102 failures, 88 were flagged and 14 were
missed; 106 Readings were flagged, 18 of them false alarms. For comparison at the same ratio, flagging
nothing costs 340 per 1,000 Readings (every failure missed) and flagging everything costs 966.

Cross-validation on the train partition gave recall 0.856 and precision 0.798 at the same threshold, so the
CV figures were slightly more pessimistic on precision here. They are not the quoted figures; the Holdout is.

**Metric rationale and class imbalance.** Only 3.4 % of Readings fail, so a model that flags nothing is
96.6 % "right" and useless. Accuracy is therefore not reported, and the app is banned from quoting it.
PR-AUC (whose no-skill value is the 3.4 % prevalence, not 0.5) and the cost-based operating point carry the
evaluation. **Costs.** A *Missed failure* (false negative) is unplanned downtime, possible damage and a
rushed repair. A *False alarm* (false positive) is one unnecessary inspection. A higher Cost ratio lowers the
threshold, so more Readings are flagged and fewer failures are missed, at the price of more false alarms.
**The 10:1 ratio is an assumption, not a measurement.** The dataset has no costs, and maintenance management
should replace it with the plant's real figure (`default_cost_ratio` in `configs/default.toml`, then
retrain). The Flag meaning shown in the app quotes these Holdout figures, labelled as held-out test data, from the
first site build made after the report was committed; the footer's build commit tells you which build
you are looking at.

## Setup and reproduction

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). The commands run from a clean clone.

```bash
uv sync                  # install runtime and dev dependencies (versions pinned to marimo's Pyodide)
just check               # ruff, ruff format check, mypy (strict), pytest
just train               # fit the default model on the train partition and write models/ (about 20 s)
just holdout-check       # the committed Holdout report belongs to the Pipeline just trained
just marimo-app          # run the app locally (needs `just train` first)
```

| Command | What it does |
|---|---|
| `just train` | Trains the random forest on the train partition only, re-derives the Operating threshold by repeated CV and **fails** if it differs from the accepted 0.218 by more than 0.0005, then writes the five-file unit to `models/` (Pipeline, OperatingPoint, missing-data policy, operating-metrics, manifest with SHA-256 and library versions). `models/` is git-ignored and never committed. |
| `just holdout` | One-off: scores the Holdout once with the frozen threshold and writes `reports/holdout-metrics.json`. The report is already committed, so a second run **refuses**. Rescoring needs `--rescore` and a new ADR. |
| `just holdout-check` | Fails with "holdout figures belong to a different model" if the report's hash is not the Pipeline in `models/`. Never rescores. |
| `just holdout-apply` | Checks the report, then labels the unit's operating-metrics `holdout`, with intervals. The build does this after training. |
| `just site` | Rehearses the Pages build into `build/site`. |
| `just test`, `just lint`, `just typecheck` | The three parts of `just check` on their own. |
| `just notebook` | Opens the marimo notebooks (evidence behind the ADRs; the pipeline never depends on them). |

- **Seeds.** One source: `seed = 42` in [`configs/default.toml`](configs/default.toml); every random
  component derives from it (the bootstrap uses seed stream 11 of it). Other settings are documented in
  [`configs/README.md`](configs/README.md).
- **Data.** `data/raw/ai4i2020.csv` must be present, with the 14 columns above and no nulls; loading fails
  with a validation error otherwise.
- **Reproducibility.** A fresh local `just train` produced a Pipeline with the same SHA-256 as the CI build,
  so the Holdout report matches both.
- **Layout.** Source in `src/predictive_maintenance/<stage>/` (`data`, `preprocessing`, `training`,
  `evaluation`, `inference`, `deploy`, plus `pipeline` and an empty `monitoring` placeholder), tests
  mirroring it under `tests/`. The app imports `inference/` only
  and holds no preprocessing, threshold or wording logic. Working agreement: [`AGENTS.md`](AGENTS.md).

## System design

[System-design diagram](docs/design/system-design.md) (Mermaid): ingestion, training, serving and
monitoring. Built parts are solid and described-only production additions are dashed.

## Production-readiness critique

### Monitoring (described, not built)

The deployed site is static and keeps no log, so nothing below runs today. In production the system would
log one record per assessed or refused Reading (timestamp, model, status, missing inputs, and for each input
only a band against the training quartiles, never raw values) and compute:

1. **Refusal rate** = Refused readings (shown as "Not assessed") ÷ all Readings, daily and over a rolling four weeks. *Purpose:*
   under policy `reject` every missing input is a lost assessment, so a rise means a failing sensor or feed.
   Broken down by absent input and source, it separates one broken sensor from a spread-out problem, and it
   tells the team whether imputation would be worth building (triggers in ADR 0008).
2. **Flag rate against expected** = Flags ÷ assessed Readings, compared with the Holdout's 35.3 per 1,000
   (interval 31.7 to 39.0) at the 3.4 % training failure rate. *Purpose:* an early, label-free signal of
   input drift or a changed machine population. A rate far outside the band means "investigate", not
   "the model is wrong"; the real failure rate may differ from the benchmark's. Review bands are the plant's
   to set.
3. **Post-outcome precision and recall.** Precision = Flagged Readings that ended in a Machine failure ÷
   Flags. Recall = failures that were flagged ÷ all failures. *Purpose:* the only direct check that the model
   still works. **Label delay:** a failure may appear days or weeks after the Reading, so these are computed
   on a cohort only once its follow-up window has closed, and they trail reality. Inspections change
   outcomes (a flagged machine that is serviced may never fail), which biases precision down and needs
   care; the figures are compared with the Holdout intervals, not with a point value.
4. **Input drift** = for each of the six inputs, the distance between its distribution in a recent window
   (for example a week of assessed Readings) and in the training partition, as the Jensen-Shannon
   divergence between the share of Readings in each training-quartile band and the 25 % each band held in
   training (`Type` uses its L, M and H shares). *Purpose:* a label-free warning that the machines or
   sensors the model sees no longer look like its training data, which is when its Flag line and its
   figures can no longer be trusted. It works on the band shares the log already holds, so no raw sensor
   value is stored. If raw values may be kept, the two-sample Kolmogorov-Smirnov statistic against a
   training sample is the finer alternative for the five numeric inputs, and the same rules apply.
   - **Review on the size of the shift, not on a p-value.** With thousands of Readings a KS or
     chi-square test rejects trivial shifts. The review line is set from the metric's own noise: the
     divergence between random training subsets of the same window size gives the "no drift" spread, and
     the line sits above it. The values are the plant's to set; none was measured here.
   - **Per input, not pooled.** Five numeric inputs and `Type` are checked separately, so a shift names its
     sensor, and a drift on several inputs at once is read as a population change. Correlated inputs
     (air and process temperature) will move together; this is a screen, not a proof.
   - **Drift is not degradation.** A shift tells you the model is operating off its training data, not that
     it is wrong. It triggers a look at the Flag rate and, once labels arrive, at post-outcome precision
     and recall (metrics 2 and 3). Coarse quartile bands also miss shape changes inside a band.

Dashboards, alerting and the health tab are designed in ADRs 0008 to 0010 and deferred by ADR 0012.

### Retraining and validation strategy

- **When:** on a schedule (for example quarterly, a judgement call), or when monitoring flags sustained drift
  or a falling post-outcome precision or recall, or when the Cost ratio changes (that moves the threshold).
- **Candidate:** refit with the same code and configuration on all labelled Readings up to a cutoff, and
  re-derive the Operating threshold by the ADR 0007 method. Never reuse this submission's Holdout: it has
  been scored, so it can no longer be used for selection.
- **Validation:** score the candidate and the live model on a **fresh holdout** (Readings after the cutoff,
  never used for fitting or selection) with the same cost metric and intervals. The candidate must not be
  worse on expected cost, and its Flag meaning figures must be restated from that holdout.
- **Shadow period:** the candidate scores live Readings silently beside the live model. Flag rate,
  agreement and, as outcomes arrive, precision and recall are compared before it replaces the live model.
- **Promotion:** a human approves it; the artifact unit's manifest hashes make rollback a matter of
  restoring the previous unit. Not built: automation, a registry, and a refit on the full data (a refit would
  have no unbiased evaluation, so it is treated as retraining and validated this way).

### Deployment risks and mitigations

| Risk | Mitigation |
|---|---|
| **Heavy first load.** About 41 MB of runtime wheels plus the model. 19 s to the first cell in an earlier Chrome run, and about 93 s to first render in a headless run behind a TLS-inspecting proxy. | The owner's rule is to reconsider the host if a second tester sees over 60 s. The next host is molab (ADR 0010). |
| **Dependence on Pyodide's host.** The runtime and packages load from their own host at run time. If it is down, the app does not load. | Check the URL before submission and periodically. A server-hosted fallback is described in ADR 0010 but not built. |
| **Pyodide moves.** A marimo upgrade changes the bundled library versions. | `marimo` and the libraries are pinned exactly to marimo 0.25.1's Pyodide; the build checks the wheel names only Pyodide-bundled packages; the app disables a unit whose manifest library versions differ from the running ones and shows "model unavailable" instead of scoring. Reading the versions from the exported site is deferred. |
| **Public model files.** The Pipeline is published with the site and anyone can download it. | Acceptable for a model trained on a public synthetic benchmark. Real plant data would need a hosted inference service with access control (dashed in the diagram). |
| **Per-browser state.** Nothing persists and nothing is shared, so there is no log and no cross-user monitoring. | The monitoring above needs a server-side log; it is described, not built. |
| **Stale thresholds when pins move.** The Operating threshold depends on library versions, so a changed pin can move scores against a fixed line. | Training re-derives the threshold and fails if it differs from the accepted 0.218 by more than 0.0005. The manifest records versions. Any figure that moves needs a new ADR (the ADR 0011 precedent). |
| **Single browser checked.** | See [Browser evidence](#browser-evidence). Treat other browsers as untested. |
| **Stale site.** The site is rebuilt only for a head commit containing `TRIGGER_CI`, so it can lag `master`. | The footer shows the build commit; the last commit before submission carries the token and the checklist compares the two. |
| **Assumed cost, synthetic data.** The 10:1 ratio is an assumption and the data is a benchmark. | Both are stated on every screen. Replace the ratio with the plant's and retrain before real use. |

### Deployment provider and limitations

GitHub Pages, serving a marimo WebAssembly export, built by `.github/workflows/pages.yml` (locked sync, full
check, train, Holdout hash check, wheel, export, contract checks, deploy). It runs only when the head commit
message contains `TRIGGER_CI` or on manual dispatch; `ci.yml` runs the checks on every push. Limits: the
heavy first load and runtime dependency above, public model files, no server-side state or logs, and the
spec's "equivalent service" is read here as covering a static site that runs Python in the browser, which a
grader expecting a server-side app (the spec names Streamlit Community Cloud) could read differently.

## Browser evidence

Stated plainly, with its limits:

- **Deployed site, headless Chromium (build `c314f96`).** The page loaded, installed the wheel and
  scikit-learn in Pyodide, downloaded the model, and assessed the default Reading ("under 1 %", flag line
  from about 22 %). The footer showed the build commit. First render took about **93 s**, from one machine
  behind a TLS-inspecting proxy. This is one automated run, not a person on a phone.
- **Earlier check before deployment (ADR 0009, 0010).** Chrome 153 on Windows, localhost, one run: scores
  within 1e-9 of the native run with the same Flags on 1,000 train Readings, 19.0 s to the first cell, 325 MB
  heap. It is not known whether that was a cold load.
- **Phone and Firefox cold test:** no result is recorded yet (ticket 11, `ready-for-human`). Until it is, the
  app has been seen working in one automated Chromium run only; Firefox, Safari and phones are untested.

## Not implemented

Deferred by [ADR 0012](docs/adr/0012-five-hour-submission-scope.md) and described above only:

1. Health tab, `monitoring/` package, the seven ADR 0008 metrics and four triggers, and the Flag-rate trigger.
2. Sample log and its regeneration check.
3. Session log with download and upload.
4. Node and Pyodide smoke test.
5. LightGBM, XGBoost, the model dropdown and the model-change notice.
6. Reading Pyodide package versions from the exported site, and the cross-browser repeat of the check.

Also not built:

- **A refit on the full data** after the Holdout evaluation. The deployed model is fitted on the 70 % train
  partition only.
- The cuttable app items of the five-hour plan (tickets 08 to 10): the Example-reading selector, the
  "outside the range of the training data" note and the sensitivity expander (recall and precision at the
  5:1, 10:1 and 20:1 Cost ratios), and the in-app golden check (a score comparison at load). Plausibility
  limits do apply: physically impossible inputs are refused. The sensitivity figures at 5:1 and 20:1 exist
  in the unit's operating-metrics file but are not shown in the app.
- Imputation, Failure-mode output, per-Reading explanations, Risk bands, calibration, tuning, a real Cost
  ratio, alerting, and retraining automation.

ADRs 0008 to 0010 describe a larger system than ships; where they differ, ADR 0012 decides.

## Documents

- [`TECHNICAL_SPEC.md`](TECHNICAL_SPEC.md): requirements. [`GLOSSARY.md`](GLOSSARY.md): terms (Reading, Flag,
  Failure risk, Operating threshold, Cost ratio, Holdout).
- [`docs/adr/`](docs/adr/): every non-trivial decision with its evidence.
- [`reports/`](reports/README.md): the committed Holdout result. [`configs/`](configs/README.md): settings.
