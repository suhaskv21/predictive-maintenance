# 0010. Deployment

- Status: Accepted
- Date: 2026-10-13
- Deciders: Suhas (design agreed in a grill session 2026-10-13; accepted 2026-10-13); drafted by an agent.

## Context
The spec (§4.3, §5, §6) asks for a small web app on a free-tier platform ("Streamlit Community Cloud
or an equivalent service"), a live public URL in the README, serialized artifacts used by the app,
and a way to reproduce them ("documented reproducible instructions to generate them"). §4.4 asks the
README to describe monitoring, two metrics, retraining and deployment risks. This ADR decides **where
the app is published, how the model artifacts and the package reach it without committing `models/`,
what the log and the monitoring do on that host, and what CI builds and checks.** It does not decide
what the app shows ([ADR 0009](0009-app-output.md), Accepted).

Fixed by accepted ADRs and the owner:
- ADR 0009 chose a framework and a host order: **marimo WebAssembly on GitHub Pages**, then marimo on
  molab, then Streamlit on Community Cloud. Options 2 and 3 are evaluated only if a gate fails with no
  fix, or the owner finds an accepted compromise unacceptable in practice. The accepted compromises
  are: (i) artifacts fit in a Pyodide-matched environment; (ii) a per-session log; (iii) published
  model files; (iv) a heavy first load; (v) a two-model build only if XGBoost fails W2 (it passed).
- The five-file artifact unit per model (the `Pipeline`, the `OperatingPoint`, `<model>.missing.json`,
  `operating-metrics.json`, `manifest.json`), the manifest check, and "a model whose files do not
  match is disabled" are ADR 0009's. `inference/`, `monitoring/` and `Presentation` import no UI
  framework. `models/` is never committed (AGENTS.md).
- The pins equal marimo's Pyodide (scikit-learn 1.8.0, xgboost 2.1.4, lightgbm 4.6.0, polars 1.33.1,
  joblib 1.5.3; numpy 2.4.3 and scipy 1.17.1 as uv constraints). XGBoost's figures under them are
  [ADR 0011](0011-xgboost-figures-under-matched-pins.md) (Proposed): threshold 0.0727.
- The monitoring scope for a static site was confirmed by the owner on 2026-10-13 (Decision).

**Stale in the decision plan.** [`docs/design/decision-plan.md`](../design/decision-plan.md) predates
the marimo and pins decisions: its Dependencies row names no pins and no marimo, and it treats
deployment as a ticket without a host. This ADR and ADR 0009 supersede it for those points. The plan
is not edited here.

### Disclosures
- **The vendor facts were gathered by a research agent from vendor documentation on 2026-10-13** and
  are cited per row in the table. Some are from source code (marimo's Pyodide version) or from one
  `curl` against a live Pages site (response headers). The facts marked "not verified" are exactly
  those the agent could not find in vendor documentation; none was tested by this project.
  Streamlit's resource figures are dated "as of February 2024" by the vendor.
- **Nothing was measured for this ADR.** W1 to W3 are the owner's browser check and the earlier Node
  inspection reported in ADR 0009: Chrome 153 on Windows, one run, from localhost, 19.0 s to the first
  cell, 325 MB heap. It is not known whether the 19.0 s was a cold or a warm load. A repeat across
  browsers and connections is a ticket.
- **The Pyodide runtime wheels (about 41 MB) were not served by the owner's localhost; Pyodide and its
  packages came from their own host.** Whether that host is a CDN, which one, and what it guarantees
  was not verified. The published site will depend on it at run time.
- **The criteria (C1 to C5) and the numbers below were written after the design was agreed in the
  grill session**, so they describe the choices rather than select them. The numbers (60 s first load,
  100 MB site budget, 20 % limit margin, 0.0005 threshold tolerance, the sample-log sizes) are the
  owner's judgements, not measurements, and can be challenged.
- **The smoke test proposed here is not a browser.** A Node and Pyodide run shares the Python and
  library versions with a browser and nothing else. Browser behaviour (memory, load time, rendering)
  stays with the manual check and tickets.
- **The ADR proposes a CI workflow and dependency changes; it makes none.** Implementing them is a
  ticket, and each is asked first (AGENTS.md).
- **The spec's "or an equivalent service" is read by the owner as covering a static site that runs
  Python in the browser.** A grader who expects a server-side Streamlit app could read it differently.
  That is a judgement, recorded as a risk (Consequences).

## Options considered
**Criteria, as agreed.** An option must: (C1) satisfy spec §4.3 (free tier, live public URL) and §5
(artifacts or reproducible instructions); (C2) run the five-file unit under the matched library
versions, with scores equal to the native run; (C3) need no committed `models/` and no edit to an
Accepted ADR; (C4) keep the log free of raw values and the model files' publication a conscious
choice; (C5) be cheap to leave if a gate fails.

1. **Host.** (a) marimo WebAssembly on GitHub Pages; (b) marimo on molab; (c) Streamlit on Community
   Cloud. ADR 0009 fixed the order; the table below fills the facts. (b) cannot receive uncommitted
   artifacts without a build step at start (a GitHub-synced notebook sees repository files), and its
   status, storage quota and stability are undocumented. (c) is the spec's named host but needs a
   server-side artifact route, the `xgboost-cpu` swap, and a server-side log. **(a)** is confirmed.
2. **Artifact route.** (a) CI trains in the pinned environment, writes the units, builds the wheel,
   exports the site and publishes everything as the Pages artifact; (b) the same, with the artifacts
   also attached to a GitHub Release and the site fetching from there; (c) force-add `models/` to the
   default branch (`models/README.md` says this must be recorded in an ADR). (c) breaks W6 and AGENTS.md
   and puts a 5 MB binary into history at every refit, with a stale-artifact risk the manifest only
   detects after the fact. (b) adds a cross-origin fetch and a second place to go stale.
   **(a)** is chosen; a release can be added later without changing the contract.
3. **Log on a static site.** (a) memory only, with download and upload; (b) browser storage
   (localStorage or IndexedDB); (c) a hosted store. (b) survives a reload but leaves data on a shared
   machine; (c) needs a backend. **(a)** is chosen.
4. **Sample log.** (a) generated by CI and published, not committed; (b) committed, regenerated by a
   command, checked in CI; (c) generated in the browser at run time. The owner chose **(b)**: it can
   be reviewed in the repository. Its cost is staleness, which the CI check removes (Decision).
5. **Smoke test.** (a) Node and Pyodide, scores on 1,000 train Readings; (b) headless browser
   (playwright, a new dependency); (c) the manual browser check only. **(a)** is the build gate; (b) is
   a ticket.
6. **Pyodide drift.** (a) pin `marimo` exactly and have the build compare the exported site's Pyodide
   package versions with the pins; (b) the score test only. **(a) plus the score test.** (b) would
   miss a version change that happens to score the same.
7. **`xgboost` against `xgboost-cpu`.** (a) swap now; (b) defer. **(b)** (Decision).

### Comparison: marimo (WebAssembly on Pages, and on molab) against Streamlit (Community Cloud)
Cell marks: **V** verified in vendor documentation on 2026-10-13 (source in brackets); **M**
measured by the owner or the earlier inspection (ADR 0009); **O** observed in one `curl`; **U** not
verified (no vendor statement found, or not tested); **J** judgement.

| Row | marimo WebAssembly on GitHub Pages | marimo on molab | Streamlit on Community Cloud |
|---|---|---|---|
| Where Python runs | In the visitor's browser (Pyodide 314.0.0, Python 3.14.2, marimo 0.25.1). **V** [marimo source, tag 0.25.1: `pyodide_constraints.py`; not in prose docs] | On molab's servers; "4 CPUs and 32 GB of RAM" per notebook. **V** [docs.marimo.io/guides/molab] | On Streamlit's servers; CPU 0.078 to 2 cores, memory 690 MB to 2.7 GB. **V** [docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app, "as of February 2024"] |
| Library-version control | None: Pyodide's fixed versions, so artifacts are fit under matched pins (ADR 0009). XGBoost across versions failed (47 of 1,000 Flags); matched, it passed. **M** | The project's own environment: packages installed on import. **V** [molab guide]. Version control inside molab: **U** | The project's lock: reads `uv.lock` first, then Pipfile, environment.yml, `requirements.txt`, `pyproject.toml`; one file only. **V** [app-dependencies] |
| How artifacts arrive without a committed `models/` | Built in CI and published in the Pages artifact (no commit to a branch). **V** [using-custom-workflows-with-github-pages]; marimo serves a `public/` folder and local wheels in a directory export, not a single-file export. **V** [marimo guides/wasm, exporting/webassembly_html] | A GitHub-synced notebook sees the repository's files, so uncommitted files need a build at start or a commit. **V** (files) / **U** (build at start) | Secrets, Git LFS, or a build at start in app code; no documented "download from a release" feature. **V** [secrets-management, file-organization] / **U** |
| First load and cold start | About 41 MB of runtime wheels plus 5 MB of models; 19.0 s to the first cell, one Chrome run; 325 MB heap. **M** Warm load: **U** | Notebooks shut down after 90 minutes idle; start time of a viewed app: **U**. **V** [molab guide] | Apps without traffic for 12 hours sleep; any visitor can wake one; wake time: **U**. **V** [manage-your-app] Install time of `xgboost` 2.1.4's 224 MB wheel plus `nvidia-nccl-cu12`: **U** |
| Log persistence | Session memory; a download and an upload (this ADR). No server storage. **J** | "Limited amount of persistent storage", quota and persistence "may change at molab's discretion". **V** (caveat) / **U** (quota) | Filesystem not documented as persistent; strongly implied ephemeral. **U** |
| Monitoring possible | The ADR 0008 metrics and triggers from one session or an uploaded log; no alerting, scheduler or cross-user view. **J** | A server could keep a shared log, but persistence is undocumented. **U** | A server-side log is possible; persistence is the problem. **U** |
| Free-tier limits and cost | Site 1 GB; soft bandwidth 100 GB a month; deploy time out after 10 minutes; the 10-builds-an-hour soft limit does not apply to Actions deploys; free on public repositories only. **V** [github-pages-limits; plans]. Per-file Pages limit: **U** | "Is molab free? Yes." **V** Status (beta or released), storage quota and any stability guarantee: **U** | Free; private repositories supported; storage 50 GB; repository size cap and install timeouts: **U**. **V** for the rest |
| Confidentiality of model files | Published and downloadable by anyone. Acceptable for a synthetic benchmark (accepted compromise iii). **J** | Notebooks are "public but undiscoverable by default", so reachable by anyone with the link. **V** | Server-side and not directly downloadable by visitors, though the repository holding them or the app code that loads them may be public. **J** |
| Screen features (tabs, model dropdown with notice, expander, banner, upload, download) | marimo has the elements (tabs, dropdown, accordion, callout, file upload, download); **not verified here**, and gate W4 is unbuilt. **U** | Same notebook as option 1. **U** | Streamlit has them (tabs, selectbox, expander, banner via `st.warning`, `file_uploader`, `download_button`). **U** (not tested here) |
| Reactivity and cost of a rerun | Cell dependency graph, run in the visitor's own CPU; marimo's WebAssembly runs without OS threads or parallelism. **V** [guides/wasm] Cost of a rerun here: **U** | Server-side, same model. **U** | The script reruns on each interaction on a server shared by visitors with at most 2 cores. **U** |
| Spec fit (§4.3 names Streamlit) | "Or an equivalent service" read by the owner as covering a free static host. Public URL needs no sleeping server. **J** | As option 1, with a third-party service whose status is unknown. **J** | Exactly the named host. **V** |
| Effort and rollback | Needs the Pyodide-matched build and the CI job (tickets). Rolling back to option 2 or 3: see Decision. **J** | Artifact build at start or commit, and a molab publish step. **J** | The `xgboost-cpu` swap, a Streamlit entry over `inference/`, an artifact route, and a server log. **J** |

**PyPI facts for the swap.** The `xgboost` 2.1.4 Linux x86_64 wheel is 223,642,416 bytes and requires
`nvidia-nccl-cu12` on Linux outside aarch64. `xgboost-cpu` 2.1.4 exists, is 4,533,116 bytes and has no
`nvidia-nccl-cu12`. **V** [pypi.org/pypi/xgboost/2.1.4/json; pypi.org/pypi/xgboost-cpu/2.1.4/json].
The browser uses Pyodide's own XGBoost, so the swap does nothing there.

## Results
No measurement was repeated or made for this ADR. The gate results below are ADR 0009's evidence.

| Gate | Status | Evidence |
|---|---|---|
| W1 Runtime | **Passed** | Chrome 153, Windows: the package, `polars` frames and all three pipelines load and score (ADR 0009, Results) |
| W2 Score equality | **Passed** | All three models, XGBoost included, within 1e-9 of the native run with the same Flags on 1,000 train Readings (ADR 0009, Results) |
| W3 Size and load | **Passed** | Site about 32 MB in the earlier check, against a 1 GB Pages limit **V**; 19.0 s to the first cell, accepted by the owner on 2026-10-13; one run |
| W4 Screen parity | **Open** | Unbuilt (ADR 0009's screens; a ticket) |
| W5 Log | **Open** | Unbuilt. The scope is fixed here; the code is a ticket |
| W6 Repo rules | **Open** | The route is fixed here (Pages artifact, no commit of `models/`). W6 passes when the CI job has run once and the check in the Decision rule holds; that needs a CI change, asked first |

## Decision

### Host: marimo WebAssembly on GitHub Pages
**Confirmed as the host**, deployed with a custom GitHub Actions workflow to the `github-pages`
environment. The repository is public (it is today), which Pages needs on a free account. The public
URL goes in the README once the first deploy has been checked; its exact form is not decided here. The
marimo app is a notebook in `run` mode exported with `marimo export html-wasm` to a directory (not a
single file, which would drop the local wheel and `public/`). Where the entry notebook lives is a ticket
(ADR 0009).

**What would make the owner switch** (in addition to ADR 0009's rule: a gate fails with no fix, or a
compromise (i) to (v) is judged unacceptable in practice). Each reopens this ADR, and the next option
is molab, then Streamlit:
1. Pyodide moves and XGBoost fails W2 under the new pins, and the owner will not accept a two-model
   build.
2. A second tester, on another connection and browser, sees a first load over **60 s**, or a failure.
3. Any verified Pages limit is within **20 %** of being exceeded.
4. A published model file turns out to expose something it should not: real data or a secret.
5. The page cannot be reached when the grader opens it.

The 60 s and the 20 % are the owner's judgements, not measurements.

### Artifact route: built in CI, published with the site
- One build produces, in the pinned environment: the five-file unit for each offered model, the
  `predictive_maintenance` wheel, the Sample log, and the exported site. All of it goes into the Pages
  artifact; **nothing is committed and `models/` stays ignored.** `models/README.md`'s force-add
  option is **not used**, so no ADR exception is needed.
- The training command (a ticket; `just train` is referenced in `.gitignore` and does not exist yet)
  trains on the train partition only, reads the seed and every setting from `configs/`, re-derives the
  Operating thresholds by the ADR 0007 method, and writes the units.
- **If the free tier or the route later fails, the fall-back is a force-added or branch-published
  artifact (option 2c).** Its technical costs: a binary in history on every refit; a stale file the
  manifest detects only at load; an exception to AGENTS.md that needs an ADR; and a deploy that
  depends on a human refit and commit. Nothing found in the vendor facts makes this necessary now.

### Build contract
The build fails if any of these holds (numbers are the owner's judgements):
1. A unit lacks one of its five files, or a name or a SHA-256 in `manifest.json` disagrees.
2. A manifest version differs from the `pyproject.toml` pins, or from the Pyodide package versions read
   from the exported site.
3. A re-derived Operating threshold differs from the accepted figure by more than **0.0005**: forest
   0.218, LightGBM 0.134, XGBoost 0.0727 (ADR 0011, once Accepted; the same guard as notebook 04).
4. The smoke test finds a score difference of **1e-9** or more, or any Flag that differs, on 1,000 train
   Readings, for any offered model.
5. The Sample log regenerated from the config differs from the committed one.
6. The exported site exceeds **100 MB** (about 10 times headroom against Pages' 1 GB; the wheels and models
   alone are about 5 MB).
7. The holdout is read: training reads the train partition only, and a test asserts it.

A model that fails the contract is not published as available (ADR 0009's disabled-model rule); a build
in which the default model fails does not deploy.

### CI: a separate `pages.yml` (proposed; asked first)
`ci.yml` stays as it is. A new workflow `pages.yml`:
- **Triggers:** a push to the default branch builds and deploys; pull requests build and run the contract
  and the smoke test without deploying; `workflow_dispatch` runs it by hand.
- **Steps:** `uv sync --locked`; `just check`; train in the pinned environment and write the units;
  build the wheel; check the Sample log; export with `marimo export html-wasm --mode run`; run the
  Node/Pyodide smoke test; `upload-pages-artifact`; `deploy-pages` (default branch only).
- **Permissions:** `contents: read` for the build job; `pages: write` and `id-token: write` for the deploy
  job only, with the `github-pages` environment.
- A separate file keeps a Pages failure from hiding a failing test and leaves `ci.yml` with one job.
- **Not measured:** the training time in CI with the threshold re-derived by repeated CV, and the
  `uv sync` time with the 224 MB `xgboost` wheel. Both are for the ticket to record.

### Smoke test
A Node script loads the Pyodide version the exported site uses, installs the built wheel, loads the five-file
units and scores 1,000 train Readings against the native scores. It fails the build under contract item 4.
The npm `pyodide` package is a dependency of the CI job only, not of `pyproject.toml`, and adding it is
asked first. A headless-browser test, and the repeat of the browser check across browsers and connections,
cold and warm, are tickets. **The smoke test is not a browser**, and passing it does not show load time or
memory.

### Pyodide drift
- `marimo` is pinned exactly in the dev group (a dependency change, asked first; today it is
  `>=0.25.1`, so a re-lock could move Pyodide silently).
- The build reads the Pyodide package versions from the exported site and fails on any difference from the
  pins or the manifest (contract item 2). The smoke test is the backstop for value drift.
- When marimo moves to a new Pyodide: the pins and constraints are updated, the notebooks re-run, the artifacts
  and thresholds regenerated, and any figure that moves at its published precision gets a new ADR like
  ADR 0011 (ADR 0011's rule). W2 is re-run first; if XGBoost fails, compromise (v) applies.

### Log and monitoring on a static site (owner-confirmed 2026-10-13)
- **Session log.** Held in memory only, so a reload clears it. **Download** is JSON Lines, one record per
  assessed or refused Reading, with ADR 0008's fields (timestamp, model, assessment record,
  `missing_inputs`, `refusal_reason`, `recoverable`, the source identifier if any, and a training-quartile
  band for each observed input) and a **`schema_version`** on every line. **No raw sensor values**, and a
  test scans both logs for them. **Upload** is parsed in the browser by `monitoring/`; a malformed line is
  rejected with its line number; nothing leaves the browser. An uploaded log replaces the analysed
  window, and the tab labels the source: "this session", "uploaded file" or "sample log".
- **Sample log.** A committed file, `data/sample/sample-log.jsonl` (outside `models/`), always labelled "Sample log".
  Proposed contents: **4 weeks of about 1,050 Readings each** (about 4,200 lines, about 1 MB as JSONL and about
  100 KB gzipped), built from train Readings only, with absent inputs injected at an assumed rate from `[missing_data]`,
  and **one degraded week** so a trigger visibly fires. A command regenerates it from the config seed; **CI
  regenerates it and fails on any difference**, so it cannot go stale after a refit or a pin change. Its size is
  kept small so parsing in Pyodide does not slow the page; the parse time is not measured and is a ticket.
  When the session has no Readings the tab shows the Sample log under the permanent banner of ADR 0009 and
  says so; the engineer can switch the source. This is how ADR 0009's "simulated log" is delivered; the layout
  is not decided here.
- **Config keys proposed, not added** (adding them is a config change, asked first, with the sample-log
  ticket): a `[sample_log]` table (Readings per week, weeks, the degraded week and its assumed missing rate)
  and a `[deploy]` table (the site-size budget, the threshold tolerance). `configs/README.md` is updated then.
- **Health tab.** Computes the ADR 0008 metrics and triggers from the session or an uploaded log; shows "not
  evaluable" below `min_weekly_readings`; no data and no log gives "No Readings yet". **No alerting,
  scheduler, central store or cross-user view is built.** Refusal rate and Flag rate against expected need no
  outcomes; Flag precision in use waits for labelled outcomes.
- **What the README says a real deployment would add** (spec §4.4), described, not built: a central log
  store with outcomes joined later; alerting and a scheduler for the ADR 0008 triggers; input drift against
  the training quartiles; the two no-outcome metrics (refusal rate, Flag rate against expected) and
  post-outcome precision and recall once labels arrive, with the label delay stated; retraining with a
  candidate validated on a fresh holdout and a shadow period before it replaces the live model; a model
  registry with the manifest as the version id; access control on the model files, since a real plant's model
  would not be public.

### `xgboost` against `xgboost-cpu`
**Not swapped.** The browser uses Pyodide's own XGBoost, so Pages and molab do not need the swap. The swap
matters for CI install size (a 224 MB wheel plus `nvidia-nccl-cu12`) and for the Streamlit fallback, which
reads `uv.lock`. It is a dependency change and would need ADR 0011's figures re-checked under it. It is the
first step if Streamlit is used, and a ticket if CI install time proves painful; either is asked first.

### Rollback and fallbacks
- **`streamlit` and `streamlit_app.py` stay** while the site goes live, as the documented fallback entry.
  Once the live URL has been verified in the README, removing them is a separate dependency-change commit that
  the owner is asked about first.
- **To molab:** the same notebook and the same `inference/`, `monitoring/` and `Presentation`; the artifacts need
  a build at start or a commit, and the log's persistence is undocumented, so it stays per session.
- **To Streamlit:** the same three packages unchanged; a Streamlit entry over `inference/` (the existing stub
  grows), the `xgboost-cpu` swap, an artifact route (secrets, LFS, or a build at start), and a server-side log
  file that an ephemeral filesystem may lose. The artifacts would not need the Pyodide match, so the pins could
  be relaxed, which is a new ADR and a dependency change.

### Decision rule: what would make this deployment wrong
No single number measures it. Review criteria, checked by the project owner before moving this ADR to
Accepted, against the spec checklist:
1. **Live URL (§4.3, §6).** After the first deploy the public URL loads and assesses one Reading, on a device and a
   connection that are not the owner's; first load under 60 s.
2. **Score equality.** The smoke test passes on the deployed build and the deployed manifest matches the pins.
3. **No committed artifacts.** `git ls-files models/` lists only `README.md`; no Accepted ADR was edited.
4. **No raw values.** A downloaded Session log and the Sample log contain no sensor values.
5. **No leak.** The published model files hold nothing beyond the fitted `Pipeline`, the `OperatingPoint` and the
   sidecars; the published file list and the pickles are inspected once.
6. **README (§4.4, §6).** It lists two no-outcome metrics, the retraining approach, and the risks below with
   mitigations.
7. **Wrong if** any trigger under "What would make the owner switch" occurs.

The risks the README lists, at least: the heavy first load; a runtime dependency on the host of Pyodide and its
packages; a Pyodide move that breaks the artifacts; public model files; a per-browser log with no shared view;
the staleness of every threshold when a pin moves; a single browser and connection checked.

## Consequences
- **No server to keep alive and no sleeping app.** GitHub Pages and the Pyodide package host (next point) are
  the services in the chain; the owner's account and a public repository are the dependencies.
- **The site depends on the Pyodide package host at run time.** If it is unreachable or slow, the app does not
  start. Not verified, not mitigated here; the README lists it.
- **Artifacts and site cannot drift apart**, because one build makes them, and a stale Sample log fails the build.
- **The project is frozen to marimo's Pyodide** (ADR 0009, ADR 0011) and every marimo upgrade is a drift check; the
  exact marimo pin turns a silent move into a failing build.
- **The log is not a record.** A reload loses the Session log; real monitoring is a README design.
- **The spec's "equivalent service" is the owner's reading** and may not be a grader's. Streamlit stays as the
  fallback for that reason.
- **molab's and Streamlit's unverified rows stay unverified** until a gate failure makes one of them the next option.
- **Model files are public**, acceptable for synthetic data and not for a real plant (ADR 0009).
- **The smoke test proves versions and scores, not the browser.** One manual run is the only browser evidence.

### Tickets this implies (not written here)
The training command and the five-file unit; the manifest check in `inference/artifacts.py`; the log sink and
`monitoring/` (health, simulate, parse); the Sample log command and its keys; the entry notebook (W4);
`pages.yml` and the contract checks; the Node/Pyodide smoke test; the exact `marimo` pin; the cross-browser,
cross-connection repeat of the browser check; the README's monitoring and risks section; removing `streamlit`
after the URL is verified.

Not decided here: what the app shows and says (ADR 0009); the XGBoost figures (ADR 0011); the layout of the
health tab and the source switch; the entry notebook's name and place; the public URL's form; the `xgboost-cpu`
swap, applied only when needed; removal of `streamlit`; model tuning; the real Cost ratio; the single holdout
evaluation and the refit on the full data; the code for any of the above.
