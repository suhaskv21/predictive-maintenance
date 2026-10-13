# 0012. Scope of the five-hour submission

- Status: Accepted
- Date: 2026-10-13
- Deciders: Suhas (agreed in a grill session and accepted the document 2026-10-13); drafted by an agent
- Refines: [0008](0008-missing-data-policy.md), [0009](0009-app-output.md) and [0010](0010-deployment.md),
  in scope and build order only. Their text and status are left as they are (ADR 0011 is the precedent).
  Nothing in 0004 to 0007 is changed. ADR 0011's XGBoost figures stay valid but are not used by this build.

## Context
Accepted ADRs 0008 to 0010 describe a larger system than can be built in the time the owner has: about
**five hours in total** to the submission. They specify three models behind a dropdown, a health tab with
seven metrics and four triggers, a Session log with download and upload, a Sample log, a Node and Pyodide
smoke test and a seven-item build contract. ADR 0008 calls the health tab "a requirement from the project
owner".

Spec §6 does not require most of that. It requires a live app that uses the serialized artifacts, an output
that does not overstate certainty, evaluation on data not used to fit, a README that **describes** monitoring
(two metrics), retraining and risks (§4.4), and a system-design diagram. The spec says production monitoring
is to be "addressed in the design/critique", not built (§7).

This ADR records what is built in the five hours, what is deferred, and four decisions made on the way, so a
reader of 0008 to 0010 is not misled about what shipped. Vocabulary follows [`GLOSSARY.md`](../../GLOSSARY.md),
which gains **Holdout** with this ADR.

### Disclosures
- **The five hours is the owner's constraint**, not an estimate of the work. The schedule below is a plan,
  with a stop rule. Nothing was timed.
- **Three facts were checked and one was not.** Verified: the throwaway browser check installed the project
  wheel with `micropip.install(..., deps=False)`, and the wheel declares `streamlit>=1.65.0`. Verified on PyPI:
  `xgboost` 2.1.4 resolves on x86_64 Linux to a 223.6 MB wheel plus `nvidia-nccl-cu12`; `xgboost-cpu` 2.1.4 is
  4.5 MB, with Linux and Windows wheels and **no macOS wheel**. **Not verified:** how marimo's `html-wasm`
  export installs a local wheel and its declared dependencies, whether `xgboost-cpu` scores identically to
  `xgboost` for this model, and the CI training time (about 3 minutes of CV on this machine for three models,
  derived from ADR 0006's per-fold fit times; one model is less; a 2-core runner may be 2 to 3 times slower).
  A background agent was asked to verify these and did not report.
- **GitHub's `contains()` in an `if:` expression is case-insensitive** (from GitHub's documented behaviour,
  not tested here), so the commit token below matches in any case.

## Options considered
1. **Build everything the ADRs describe** — rejected: it does not fit, and it would put the spec's required
   items (live URL, holdout, README) at risk.
2. **Cut the spec's required items instead** — rejected: §6 lists them.
3. **Ship the required path and defer the rest, recorded here** (chosen).

## Decision

### Built
- **Forest only.** The default model of ADR 0006 with its Operating threshold of 0.218 (ADR 0007).
- **Training command (`just train`)**: train partition only, seed from `configs/`, re-derives the threshold by
  the ADR 0007 method, writes ADR 0009's five-file unit. `<model>.missing.json` is written with strategy
  `reject` and an **empty reference**, marked "not measured in this build".
- **`inference/`**: `artifacts.py` (loads and checks the unit: files, names, SHA-256, library versions),
  `assess.py` (the assessment record and the Flag on the unrounded score) and `presentation.py` (the
  wording, with the banned-wording test). `inference/` returns the assessment record. It writes **no log**.
- **One-screen app** at the repository root, `marimo_app.py`, beside `streamlit_app.py`: input form,
  the three headline states (Inspect, No inspection flagged, Not assessed), the Failure risk line, the Flag
  meaning, the always-on caveats, Example readings from the train partition, and the 5:1, 10:1 and 20:1
  sensitivity expander (metrics only). The build commit is shown in the footer. It imports no logic.
- **`pages.yml`**, exporting with `marimo export html-wasm` to GitHub Pages, with the contract items that are
  cheap: unit files and manifest present, a re-derived threshold within tolerance of the accepted figure, site
  size within budget, and a test that training does not read the holdout.
- **In-app golden check**: at load, the notebook scores a few embedded train Readings and compares them with
  the scores stored in the manifest, with a visible failure banner on mismatch. It replaces the Node smoke test.
- **README and a Mermaid diagram** at `docs/design/system-design.md`, drawn last. Built parts are solid boxes
  and described-only parts are dashed.

### Deferred (described in the README, not built)
1. Health tab, `monitoring/`, the seven ADR 0008 metrics and four triggers, the Flag-rate trigger.
2. Sample log and its CI regeneration check (contract item 5).
3. Session log sink and its download and upload.
4. Node and Pyodide smoke test (contract item 4); the in-app golden check stands in for it.
5. LightGBM, XGBoost, the dropdown and the model-change notice. Their `[deploy]` thresholds are added when
   they ship.
6. Reading the Pyodide package versions from the exported site (the site half of contract item 2), and the
   cross-browser repeat of the browser check.

ADR 0009's wording rules, the Refused-reading screen and the five-file unit are **not** deferred.

### Four decisions made on the way
- **The one holdout result.** A one-off command scores the Holdout once, for the forest, with frozen
  thresholds, and writes `reports/holdout-metrics.json` (committed; `reports/` is not `models/`): the
  Pipeline's SHA-256, the date, and each ADR 0007 rule 9 metric with its bootstrap interval. **Run it
  from the CI-built artifact**, so the numbers belong to the deployed file. Builds read the file and **never
  rescore**; if the rebuilt Pipeline's hash differs, the build fails ("holdout figures belong to a different
  model"). Rescoring needs an explicit flag and a new ADR. The forest's `operating-metrics` sidecar then
  shows these figures, labelled `holdout`. **No refit on the full data** for this deliverable (open since ADR
  0005); the README treats it as retraining. If CI and local builds are not bit-identical, switch the check from
  file bytes to a fingerprint (pinned versions plus scores on fixed train Readings within 1e-9), never to a
  rescore.
- **`pages.yml` runs only on request.** A push whose head commit message contains `TRIGGER_CI`, or
  `workflow_dispatch`. No path filter, no pull-request trigger. `ci.yml` (`just check`) stays automatic on
  every push and pull request. The match is case-insensitive.
- **The `streamlit` dependency moves to an optional extra** (its name is a ticket), so the wheel declares only
  libraries Pyodide bundles. `streamlit_app.py` stays. A check that the built wheel's `Requires-Dist` names
  only Pyodide-bundled packages joins the build contract. The first step of the app ticket is to verify how
  marimo installs the wheel. `marimo` is pinned exactly in the dev group.
- **`xgboost` is not swapped for `xgboost-cpu`.** The swap would break a macOS install (spec §6 asks for a
  clean install) and would need `xgboost.__version__` in the manifest, since the distribution name changes.
  Trigger to revisit: the cold `uv sync` step in `pages.yml` takes more than **3 minutes** (the owner's number,
  not a measurement). The form then is a Linux-only environment marker, not an unconditional swap.

### Approvals recorded (AGENTS.md "ask first"), 2026-10-13
Approved by Suhas: `pages.yml`; the exact `marimo` pin; an npm `pyodide` package in the CI job only (not
needed while the smoke test is deferred); the `streamlit` extra; excluding `marimo_app.py` from ruff; a
`[deploy]` table in `configs/default.toml` and `configs/README.md` with `accepted_thresholds` (forest 0.218),
`threshold_tolerance` (0.0005) and `site_budget_mb` (100), replacing notebook 04's hard-coded 0.073 for XGBoost
once XGBoost ships. Not approved: `[sample_log]` keys (not needed), and the key that sets how many Example
readings are recorded failures (asked at its ticket). Enabling Pages (Settings, Pages, Source: GitHub Actions)
is the owner's step, needed by about hour 2:30.

### Order and stop rule
0:00 this ADR, tickets, approved dependency and config edits; 0:30 training command; 1:30 `inference/`;
2:15 entry notebook, `pages.yml`, first `TRIGGER_CI` push; 3:00 phone and Firefox cold test, fixes; 3:30
holdout; 4:00 README and diagram; 4:45 final `TRIGGER_CI` commit and checklist.
If the live URL is not up by 3:15, cut in this order: the sensitivity expander, Example readings, the in-app
golden check, diagram detail. **Never cut:** the live URL, the holdout, the banned-wording test, and a
passing `just check`, shown before each claim of done.

## Consequences
- **ADRs 0008 to 0010 describe more than ships.** The health tab, the logs, the two extra models and the
  smoke test exist only as design. The README says so plainly and lists them under "not implemented"
  (spec §4.1.5). A reader of those ADRs should read this one.
- **A stale site is possible.** `pages.yml` runs on request, so a source change pushed without `TRIGGER_CI`
  leaves the live site and the README's numbers behind `master`. The footer shows the build commit, and the
  last commit before submission must contain `TRIGGER_CI`, with "live site built from `master`'s HEAD" on the
  checklist. Only the head commit of a push is checked, so the token goes in the last commit pushed.
- **A browser scoring difference would not fail the build**, because there is no Node smoke test. The in-app
  golden check shows it to a visitor, and the phone and Firefox run is the only browser evidence, as before.
- **The holdout hash can fail a build** if training is not bit-identical across machines. That is the intended
  failure (it stops a silent rescore), with the fingerprint fallback above.
- **The Flag's figures for the forest are the holdout's**, with intervals of roughly ±7 points on recall (about
  102 failures), against CV figures that are slightly optimistic (ADR 0007).
- **Whether marimo installs the wheel's dependencies is still unverified.** The `streamlit` extra removes the
  risk if it does; the first app ticket confirms it.
- **The `.missing.json` reference is empty**, so nothing downstream can tell a Recoverable refusal from another.
  Nothing downstream exists yet.
- **Revisit after submission:** build deferred items 1 to 3 first (they carry ADR 0008's owner requirement),
  then the other two models with their ADR 0011 figures, then the smoke test.

Not decided here: the `streamlit` extra's name; the entry notebook's layout and final copy; how many Example
readings are recorded failures; sensor plausibility ranges; the public URL beyond Pages' default form; the real
Cost ratio; model tuning; a refit on the full data after submission.
