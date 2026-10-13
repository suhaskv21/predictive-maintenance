# Five-hour submission: live failure-risk app with leakage-safe evidence

Status: ready-for-agent

Source of truth: `TECHNICAL_SPEC.md` (§4–§6), ADRs 0004–0012 (0012 sets the scope and order), `GLOSSARY.md`.
Vocabulary below follows the glossary (Reading, Flag, Failure risk, Operating threshold, Cost ratio, Risk floor,
Refused reading, Example reading, Holdout).

## Problem Statement

A maintenance engineer has a machine's sensor values and needs to decide whether to inspect it now. Today the
repository holds the evidence (notebooks, ADRs 0004–0011, tested evaluation and candidate code) but nothing an
engineer can open: no trained artifact, no assessment code, no app, no live URL, no Holdout result, and no README
that explains how to read the output. The submission has a hard budget of about five hours, and the spec's required
items (live URL, Holdout evaluation, README, diagram, passing `just check`) are at risk if the larger design in ADRs
0008–0010 is built first.

## Solution

Build the smallest path that satisfies spec §6, as recorded in ADR 0012:

- one command trains the default model (random forest, ADR 0006) on the train partition only and writes a verified
  five-file artifact unit;
- an `inference` layer loads the unit, checks it, assesses one Reading and returns an assessment record plus a
  Presentation (the words); it holds no UI code and writes no log;
- a one-screen marimo app, published as a static WebAssembly site on GitHub Pages, shows one of three headline
  states (Inspect, No inspection flagged, Not assessed) with the Failure risk, what the Flag means, and always-on
  caveats, and imports no logic;
- the Holdout is scored exactly once, from the CI-built artifact, and the figures are committed;
- a build workflow that runs only on request deploys the site and enforces the cheap build checks;
- the README and a system-design diagram describe everything not built (health tab, logs, extra models) as
  production additions.

Engineers see a Flag they can act on, a rough Failure risk that never claims certainty, and an explicit "Not
assessed" when inputs are missing.

## User Stories

1. As a maintenance engineer, I want to enter the six inputs of a Reading (Type and five sensor values) into one form, so that I can assess a machine without any ML knowledge.
2. As a maintenance engineer, I want a clear headline of "Inspect" or "No inspection flagged", so that I know what to do next.
3. As a maintenance engineer, I want the Flag to be decided by the model's own Operating threshold on the unrounded score, so that the headline never contradicts the model because of display rounding.
4. As a maintenance engineer, I want a secondary Failure risk line labelled "rough estimate, not calibrated", so that I can order two unflagged Readings without mistaking it for a guarantee.
5. As a maintenance engineer, I want the Failure risk shown as "under 1 %" below the Risk floor, "about N %" between, and "over 95 %" above, so that I never see "0 %" or "100 %".
6. As a maintenance engineer, I want to see the model's own Flag line beside the Failure risk, so that I can see how close a Reading is to being flagged.
7. As a maintenance engineer, I want a fixed note that a risk within a percent of the line can go either way, so that near-threshold results are not over-read.
8. As a maintenance engineer, I want plain words for what a Flag means (of every 100 failures about N were flagged, of every 100 Flags about M were real, about K per 1,000 Readings are flagged), so that I can judge how much to trust it.
9. As a maintenance engineer, I want a flagged result to say how many Flags are false alarms, so that I expect some wasted inspections.
10. As a maintenance engineer, I want an unflagged result to say how many real failures were not flagged, so that it never reads as a clearance.
11. As a maintenance engineer, I want a "Not assessed" state, styled differently from "No inspection flagged", when an input is absent or unusable, so that an unscored machine never looks like a passed machine.
12. As a maintenance engineer, I want Not assessed to list each missing input and why (absent, not a number, not finite, out of range), so that I know what to fix.
13. As a maintenance engineer, I want Not assessed to say "this does not mean the machine is fine; treat it as unassessed if a sensor is offline", so that I do not read silence as safety.
14. As a maintenance engineer, I want a Refused reading to show no Failure risk, no Flag, no threshold and no fallback score, so that nothing can be misread.
15. As a maintenance engineer, I want physically impossible values (negative speed, non-finite numbers, text) treated as absent, so that garbage input is refused rather than scored.
16. As a maintenance engineer, I want to pick an Example reading from the training data, so that I can try the app without typing values.
17. As a maintenance engineer, I want an Example reading to show its recorded Machine failure outcome quietly, with "this Reading was in the training data, so agreement proves nothing", so that a demo does not overstate accuracy.
18. As a maintenance engineer, I want the recorded outcome line to disappear when I edit any input, so that it never describes a Reading I changed.
19. As a maintenance engineer, I want an expander with recall, precision and Flags per 1,000 at the 5:1, 10:1 and 20:1 Cost ratios (metrics only, no thresholds), so that I can see how sensitive the Flag is to the cost assumption.
20. As a maintenance engineer, I want the forest's 20:1 row marked "not precise", so that I do not treat a non-flat optimum as exact.
21. As a maintenance engineer, I want every screen to say "estimated from historical readings", "synthetic benchmark data, not your plant", the 10:1 cost assumption (read from config) and "an aid for prioritising inspections, not a guarantee", so that limits are always visible.
22. As a maintenance engineer, I want every result to name its model, so that I know which Flag line applies.
23. As a maintenance manager, I want the Cost ratio shown as an assumption and read from configuration, so that I can replace it with the plant's real figure.
24. As a maintenance manager, I want the app's quoted Flag figures to be the Holdout figures (with intervals) once scored and labelled as such, so that headline numbers are the unbiased ones.
25. As a reviewer, I want a banned-wording test over every rendered state ("will fail", "safe", "healthy", "OK", "no risk", "0 %", "100 %", "guaranteed", "predicts failure", unqualified "probability of failure", unsourced accuracy claims), so that overstated certainty cannot ship.
26. As a reviewer, I want the app to import no logic (no threshold, rounding, wording or preprocessing), so that the UI-calls-inference-only rule holds.
27. As a reviewer, I want `inference` to import no UI framework, so that it runs in the browser and in tests.
28. As a data scientist, I want one command that trains the forest on the train partition only, with the seed from `configs/`, so that training is repeatable without notebook state.
29. As a data scientist, I want training to re-derive the Operating threshold by the ADR 0007 method (per CV repeat, median across repeats, 10:1) and compare it to the accepted 0.218 within 0.0005, so that a silent drift in the model or library versions fails loudly.
30. As a data scientist, I want a test proving that training never reads the Holdout, so that the leakage rule is enforced, not promised.
31. As a data scientist, I want preprocessing and model serialized together as one scikit-learn Pipeline, so that inference applies identical transformations.
32. As a data scientist, I want failure-mode columns and identifiers excluded from features in the trained Pipeline, so that leakage is impossible by construction.
33. As a data scientist, I want training to write a five-file unit (Pipeline, OperatingPoint, missing-data policy, operating-metrics, manifest), so that the app has everything it needs and nothing else.
34. As a data scientist, I want the missing-data policy file to state strategy `reject` with an empty reference marked "not measured in this build", so that downstream readers are not misled.
35. As a data scientist, I want the manifest to record library versions and SHA-256 of the other four files, so that a stale or mismatched file is detectable.
36. As a data scientist, I want the operating-metrics file to carry CV figures labelled `cv_train` until the Holdout is scored, then `holdout`, so that every quoted figure states its source.
37. As a data scientist, I want artifact loading to fail if any file is missing, a name disagrees, a hash mismatches, or library versions differ from the running ones, so that a bad model is never scored with a fallback threshold.
38. As a data scientist, I want a failed default model to put the app in an error state rather than substituting another model, so that a Flag never meets a foreign threshold.
39. As a data scientist, I want complete Readings scored bit-identically to the training-time Pipeline, so that the app and the evidence agree.
40. As the project owner, I want a one-off command that scores the Holdout once, for the forest, with frozen thresholds, so that the submission has one unbiased result.
41. As the project owner, I want that command to write a committed report with the Pipeline's SHA-256, the date, and recall, precision, expected cost, PR-AUC, ROC-AUC and Brier each with a percentile 95 % bootstrap interval (2000 resamples, the seed stream from ADR 0007 rule 9), so that results are reviewable and the threshold is never re-chosen inside a resample.
42. As the project owner, I want the Holdout command run from the CI-built artifact, so that the reported numbers belong to the deployed file.
43. As the project owner, I want builds to read the committed Holdout report and never rescore, failing with "holdout figures belong to a different model" if the rebuilt Pipeline's hash differs, so that the Holdout cannot be silently reused for selection.
44. As the project owner, I want rescoring to need an explicit flag and a new ADR, so that a second look at the Holdout is a deliberate, recorded act.
45. As the project owner, I want a fallback to a fingerprint check (pinned versions plus scores on fixed train Readings within 1e-9) if CI and local builds are not bit-identical, never a rescore, so that reproducibility problems do not become leakage.
46. As the project owner, I want the build workflow to run only when the head commit message contains the trigger token (any case) or on manual dispatch, so that the live site changes only when I ask.
47. As the project owner, I want the build to fail if a unit file or manifest entry is missing, a re-derived threshold drifts beyond tolerance, the exported site exceeds the size budget, or training read the Holdout, so that a bad build never deploys.
48. As the project owner, I want the build to check that the wheel's declared dependencies name only packages Pyodide bundles, so that the browser install cannot fail on a native dependency.
49. As the project owner, I want the Streamlit dependency moved to an optional extra, so that the wheel stays installable in the browser while `streamlit_app.py` remains as a fallback entry.
50. As the project owner, I want the site footer to show the build commit, so that I can confirm the live site matches the last commit.
51. As a visitor, I want an in-app golden check at load that scores a few embedded train Readings and compares them with the manifest's stored scores, with a visible failure banner on mismatch, so that a browser scoring difference is visible instead of silent.
52. As a visitor, I want the app to work on a phone and in Firefox from a cold load, so that the public URL is genuinely usable.
53. As a grader, I want a public URL in the README that loads and assesses one Reading, so that spec §4.3 and §6 are met.
54. As a grader, I want the README to state the problem, users, target, features, excluded columns, missing-data handling, split, metric, model trade-off, output and deployment limits, so that every spec §8 decision is explicit.
55. As a grader, I want the README to define at least two monitoring metrics (refusal rate and Flag rate against expected) with purpose, plus post-outcome precision and recall and their label delay, so that spec §4.4 is met.
56. As a grader, I want the README to describe a retraining strategy: a candidate validated on a fresh holdout and a shadow period before replacing the live model, so that model replacement is controlled.
57. As a grader, I want the README to list deployment risks with mitigations (heavy first load, dependency on Pyodide's host, Pyodide moves, public model files, per-browser state, stale thresholds when pins move, one browser checked), so that the critique is honest.
58. As a grader, I want a system-design diagram covering ingestion, training, serving and monitoring, with built parts solid and described-only parts dashed, so that the gap between challenge and production is visible.
59. As a grader, I want the README to list what is not implemented (health tab, logs, extra models, smoke test, refit on full data), so that nothing is implied beyond what ships.
60. As a grader, I want the README to document setup, the training command, the Holdout command, the seed and data expectations, so that the work reproduces from a clean install.
61. As a future maintainer, I want the deferred items (ADR 0012 list) recorded, so that I can build them in the stated order after submission.
62. As a future maintainer, I want `just check` (lint, mypy strict, tests) to pass before any claim of done, so that the repository stays shippable.

## Implementation Decisions

**Scope and order (ADR 0012, Accepted).** Forest only, at the accepted 10:1 Operating threshold of 0.218. The
design in ADRs 0008–0010 is larger than what ships; where they differ, ADR 0012 wins. Order and stop rule: ADR,
tickets and approved edits; training command; `inference`; entry notebook, build workflow and first triggered push;
phone and Firefox cold test; Holdout; README and diagram; final triggered commit and checklist. If the live URL is
not up by the 3:15 mark, cut in this order: the sensitivity expander, Example readings, the in-app golden check,
diagram detail. Never cut: the live URL, the Holdout, the banned-wording test, a passing `just check` shown before
each claim of done.

**Training module (existing `training` and `evaluation` packages).**
- A single entry command (`just train`) fits the default candidate from the existing candidate registry on the
  train partition produced by the existing stratified holdout split. The Holdout is discarded at load time and never
  touched; a test asserts this.
- Seeds and settings come only from the config (single seed source). Missing-data and threshold margins stay as
  configured.
- It re-derives the Operating threshold with the existing repeated-CV threshold method, checks it against the
  accepted figure within the configured tolerance, and fails otherwise.
- Output is a five-file unit per model: serialized Pipeline; OperatingPoint (model, default ratio, thresholds per
  ratio, no bands, Risk floor); missing-data policy (strategy `reject`, empty reference, "not measured in this
  build"); operating-metrics (CV figures labelled `cv_train`, per Cost ratio: recall, precision, Flags per 1,000,
  false alarms per caught failure); manifest (model name, library versions, SHA-256 of the other four files).
- Artifacts are never committed.
- The model is registered so that LightGBM and XGBoost can be added later without changing the contract; they are
  not offered now.

**Inference module (new, no UI imports, no logging).**
- Artifact loader: checks file presence, names, SHA-256 and library versions; a failing unit is disabled and is
  never scored with a fallback. A failed default model yields an error state.
- Assessment: takes one Reading's six inputs, classifies each input as present or unusable (absent, not a number,
  not finite, out of range), refuses any Incomplete reading under policy `reject`, otherwise scores with the
  Pipeline and decides the Flag on the unrounded score against the model's own Operating threshold. Returns an
  assessment record: status (`complete` or `refused`), missing inputs with reasons, refusal reason, and a
  recoverable flag (always false in this build, because the reference is empty).
- Presentation: maps the assessment, OperatingPoint and operating-metrics to a Presentation: headline state, Failure
  risk text, Flag-line text, Flag-meaning text, caveats, refusal lines, Example-reading outcome line, sensitivity
  rows. It holds all rounding (halves up; Risk floor below; "over 95 %" above; never "0 %" or "100 %") and all
  wording, and owns the banned-wording list. Cost ratio text is read from config, not typed.
- Plausibility (approved by Suhas, 2026-10-13): only physically impossible values are treated as absent, with
  reason "out of range". Valid ranges, held in a `[plausibility]` config table and documented in the config
  README, not in code: air temperature 200–400 K; process temperature 200–450 K; rotational speed above 0 to
  10,000 rpm; torque 0–500 Nm; tool wear 0–1,000 min; Type one of L, M, H. Values inside these ranges but outside
  the training data's min/max are still scored, with a non-refusing note on the result: "outside the range of
  the training data, treat the result as unreliable" (the first item cut if time runs short). That note, like
  all wording, comes from the Presentation and passes the banned-wording test.

**Entry app (marimo, repository root, beside the existing Streamlit stub).**
- One screen: input form (Type as a dropdown, five numeric inputs), Example reading selector, three headline
  states, Failure risk line, Flag meaning, always-on caveats, sensitivity expander, build-commit footer, and the
  in-app golden check with a visible failure banner.
- Renders strings from the Presentation; imports no logic. Excluded from the linter by prior approval.
- Example readings (approved 2026-10-13): the configured count (10), drawn only from the train partition with the
  config seed, per outcome class: `[app] example_failures = 3` recorded failures and the rest non-failures (a plain
  draw at 3.4 % prevalence would give none). The draw fails loudly if the train partition has fewer failures than
  requested. The key is documented in the config README.

**Holdout scoring (one-off command).**
- Scores the Holdout once, for the forest, with the frozen threshold, from the CI-built artifact. Writes a
  committed report holding the Pipeline SHA-256, date and each ADR 0007 rule 9 metric with its bootstrap
  interval. The forest's operating-metrics then show these figures, labelled `holdout`.
- Normal builds read the report and never rescore; a hash mismatch fails the build. Rescoring requires an explicit
  flag plus a new ADR. If builds are not bit-identical, switch the check to a fingerprint, never to a rescore.
- No refit on the full data for this deliverable; the README describes it as retraining.

**Build and deploy.**
- A new workflow, separate from the existing test workflow, runs only on a head-commit message containing the
  trigger token (case-insensitive) or manual dispatch; no path filter, no pull-request trigger. The existing check
  workflow stays automatic.
- Steps: locked sync; full check; train; build wheel; export the marimo site in WebAssembly mode as a directory;
  contract checks; upload and deploy to Pages. Permissions are least-privilege, with Pages write only on the
  deploy job.
- Contract (cheap items only): five-file unit present and manifest hashes agree; re-derived threshold within
  tolerance; site size within budget; training does not read the Holdout; wheel dependencies name only
  Pyodide-bundled packages; Holdout report hash matches the built Pipeline.
- Config: a new deploy table with accepted thresholds (forest 0.218), threshold tolerance (0.0005) and site budget
  (100 MB), documented alongside the other config keys.
- The Streamlit dependency moves to an optional extra (name decided at its ticket); `marimo` is pinned exactly in
  the dev group. The first step of the app ticket is to verify how marimo installs the local wheel and its
  declared dependencies.
- XGBoost is not swapped for its CPU variant; revisit only if the cold dependency install exceeds 3 minutes.
- Enabling Pages (source: GitHub Actions) is the owner's manual step, needed early in the build.

**Documentation.**
- README: live URL, setup, commands, seeds and data expectations, all spec §8 decisions, the Holdout results with
  intervals, a monitoring section (refusal rate; Flag rate against expected; post-outcome precision and recall),
  retraining and validation strategy, deployment risks and mitigations, and a "not implemented" list (spec §4.1.5).
- A Mermaid system-design diagram, drawn last: built parts solid, described-only parts dashed, covering ingestion,
  training, serving, monitoring and the production additions.
- Glossary gains no new terms beyond Holdout (already added); any term the build invents goes through the
  domain-modelling process first.

**ADR conflicts.** None. ADR 0012 refines 0008–0010 in scope and order only, and this spec follows 0012. This
spec asks for the five-file unit and the Refused-reading screen in full, which ADR 0012 keeps. Any new
non-trivial decision made during the build (for example plausibility ranges, the extra's name, the fingerprint
fallback) is a Proposed ADR, not an edit to an Accepted one.

**Approvals (AGENTS.md "ask first").** Recorded in ADR 0012: the build workflow, exact `marimo` pin, the
`streamlit` extra, linter exclusion of the app, and the deploy config table. Approved by Suhas on 2026-10-13, in
this spec:
- The extra is named `streamlit`: the dependency moves from the main list to an optional extra, and the app
  command runs with that extra (`uv sync --extra streamlit` to install it).
- New config keys: `[app] example_failures = 3` and a `[plausibility]` table (ranges above), both documented in the
  config README.
- The `marimo` pin is exactly `0.25.1` (already locked; matches the Pyodide pins). No other dependency is added;
  the npm `pyodide` package is not needed while the smoke test is deferred. Any further dependency is asked first.
- The plausibility approach is not a decision that needs its own ADR as approved; changing it (for example to
  refuse values outside the training range) would be a Proposed ADR.

## Testing Decisions

**What a good test is here.** Assert externally visible behaviour: the files a command produces, the assessment
and Presentation returned for given inputs, and the text a user would see. Do not assert internals (private
helpers, intermediate arrays, call order, specific model hyperparameters beyond the configured contract).

**Seam 1 (main): the `inference` public interface.** Load a unit, assess a Reading, read the Presentation. Tests:
- complete Reading above and below the threshold gives Inspect and No inspection flagged; the Flag follows the
  unrounded score when the displayed percent equals the threshold's percent;
- every missing or unusable input kind (absent, not a number, not finite, out of range, each of the six inputs)
  gives Not assessed with the right reason, and never a Failure risk, Flag, threshold or fallback score;
- risk text rules: under the Risk floor, rounding halves up, over 95 %, never "0 %" or "100 %";
- plausibility: values on each side of every configured range limit are refused or scored accordingly; a value
  inside the range but outside the training min/max is scored with the "unreliable" note;
- the banned-wording scan over every rendered state (Inspect, No inspection flagged, Not assessed, model
  unavailable, Example reading) and the always-on caveats present on each;
- Example-reading outcome line shown only for an unedited Example reading;
- artifact checks: missing file, wrong name, bad hash, version mismatch each disable the unit with a reason and
  never fall back; a failed default gives the error state;
- a complete Reading's score from the loaded unit equals the Pipeline scored directly (bit-identical).

**Seam 2: the `just train` output.** Run on a small fixture dataset and assert on the produced unit: five files,
manifest hashes and versions correct, OperatingPoint threshold within tolerance of the re-derived value, the
missing-data policy says `reject` with an empty reference, operating-metrics labelled `cv_train`, serialized
Pipeline contains no failure-mode or identifier columns, and a spy on the Holdout partition shows it is never read.
The Holdout command is tested at its output too: report fields present, hash recorded, and a second run without the
explicit flag refuses to rescore.

**Not tested through new seams.** The marimo app and the build workflow are thin shells: covered by the in-app
golden check, the build contract steps, and the manual phone and Firefox cold test. The README and diagram are
reviewed against the spec §6 checklist.

**Prior art.** Existing mirrored tests under `tests/predictive_maintenance/` for the candidate registry, threshold
and split code (fixtures in the shared conftest), the config loading test, and the layout test that enforces the
mirrored test-file convention. New modules get a mirrored test file each.

**Done means** `just check` (lint, strict typecheck, tests) passes and its result is shown.

## Out of Scope

Deferred by ADR 0012 and described in the README only:
- the health tab, the `monitoring` package, the seven ADR 0008 metrics, four triggers and Flag-rate trigger;
- the Sample log and its regeneration check;
- the Session log sink with download and upload;
- the Node and Pyodide smoke test (the in-app golden check stands in);
- LightGBM, XGBoost, the model dropdown and the model-change notice, and their deploy thresholds;
- reading Pyodide package versions from the exported site and the cross-browser repeat;
- an optional Streamlit or molab host.

Also out of scope: imputation (policy is `reject`), Failure mode output, per-Reading explanations, Risk bands,
calibration, model tuning, a refit on the full data, a real Cost ratio, alerting or retraining automation, access
control on model files, and any change to an Accepted ADR.

## Further Notes

- The five hours is the owner's constraint, not an estimate; nothing here was timed. CI training time with the
  threshold re-derived by repeated CV is unmeasured (about 3 minutes derived for three models, one is less, and a
  2-core runner may be 2 to 3 times slower).
- Unverified at spec time: how marimo's WebAssembly export installs a local wheel and its dependencies; whether
  the in-browser result matches native within 1e-9 for the forest (it did in an earlier browser run, under matched
  pins); how long the site's first load takes on other connections.
- Risk: a source change pushed without the trigger token leaves the live site and README numbers behind the main
  branch. The last commit before submission must contain the token, and the checklist must confirm the live site
  was built from the head commit.
- Risk: the Holdout hash can fail a build if training is not bit-identical across machines. That is the intended
  failure; the fingerprint fallback applies.
- Pins are tied to marimo's Pyodide (scikit-learn 1.8.0, xgboost 2.1.4, lightgbm 4.6.0, polars 1.33.1). Any pin
  change re-runs the notebooks and may need an ADR of the ADR 0011 kind.
- Suggested ticket split (for a later breakdown): training command and unit; artifact loader; assessment and
  Presentation with banned-wording test; Holdout command; entry notebook and golden check; build workflow, config
  and contract; README and diagram.
