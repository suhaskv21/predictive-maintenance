# 05: Build workflow and first live deploy

**What to build:** A separate build workflow that, on request, trains in CI, builds the wheel, exports the app as a static WebAssembly site, checks the build contract and deploys it to GitHub Pages. A pushed commit with the trigger token produces a public URL that loads and assesses a Reading.

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 04 (One-screen marimo app (local))

**Status:** resolved

- [x] Owner prerequisite, needed before the first deploy: enable Pages with source GitHub Actions (Settings, Pages). The agent states this clearly and stops if the deploy step reports Pages is disabled.
- [x] The workflow runs only when the head commit message contains `TRIGGER_CI` (any case) or on manual dispatch; no path filter and no pull-request trigger. The existing check workflow is untouched and stays automatic.
- [x] Steps: locked sync; full check; train; build the wheel; export the site as a directory in WebAssembly mode; contract checks; upload and deploy. Pages write permission only on the deploy job; read-only elsewhere.
- [x] Contract checks fail the build on: a missing unit file or manifest mismatch; a re-derived threshold outside tolerance of the accepted figure; a site over the configured budget; training reading the Holdout; wheel dependencies not bundled in Pyodide.
- [x] The footer shows the deployed build commit.
- [x] After the first triggered push the public URL loads and assesses one Reading; the URL is recorded in the ticket's comments.
- [x] The time of the cold sync and the training step is recorded; the 3-minute trigger for revisiting the XGBoost swap is noted if exceeded.
- [x] `just check` passes and its result is shown.

## Comments

Implemented (not committed): `.github/workflows/pages.yml` (push + dispatch; `build` job gated on `TRIGGER_CI` in the head commit
or dispatch, `contents: read`; `deploy` job alone has `pages: write` and `id-token: write`), `deploy/contract.py` (unit files and
manifest via `load_unit`, unit threshold within tolerance of `[deploy]` accepted figure, site size against `site_budget_mb`,
wheel `Requires-Dist` names only Pyodide-bundled packages), `deploy/stage.py` (stamps `BUILD_COMMIT` and the PEP 723 wheel header
into a copy of `marimo_app.py`, stages `public/models` and `public/configs`), `inference/site.py` (the notebook downloads the
unit and config from the site's `public/` folder in the browser; local runs read the folders beside the notebook), `just site`.
"Training reading the Holdout" is the two `-k holdout` tests in `test_train.py`, run as an explicit workflow step; the threshold
re-derivation itself stays in training, which fails outside tolerance. `just check`: ruff, mypy strict, 647 tests pass.

**Owner prerequisite (before the first deploy):** Settings, Pages, Source: GitHub Actions. The deploy job fails with "Pages is
not enabled" until then. A trigger commit on a non-default branch builds but may be refused by the `github-pages` environment's
branch rule.

**Local rehearsal** (`just site` steps, then headless Chromium on `http.server`): wheel built, site 31 MB (budget 100 MB), contract
passed, the page loaded in the browser, installed the wheel and scikit-learn in Pyodide, downloaded the unit, assessed the default
Reading ("Failure risk ... under 1 %", flag line from about 22 %) and showed "Build commit: local123". About 60 s to first render
here (behind a TLS-inspecting proxy).

**Timings.** Local, 2-core-or-more laptop: `just train` 17 s wall. The CI cold sync and training times are written to each run's
summary (a warning is added if the cold sync exceeds 3 minutes, the trigger to revisit the XGBoost swap); copy them here after
the first run.

**First triggered deploy (2026-10-14):** public URL https://suhaskv21.github.io/predictive-maintenance/ (commit `c314f96`).
Checked in headless Chromium: the page loaded, installed the wheel, downloaded the unit and assessed the default Reading
("Failure risk (rough estimate, not calibrated): under 1 %", "This model flags from about 22 %") with footer
"Build commit: c314f96", matching the head commit. About 93 s to first render from this machine (behind a TLS-inspecting proxy).

**CI timings:** cold `uv sync --locked` 4 s; training step 23 s. Both far under the 3-minute trigger, so the XGBoost swap is
not revisited. Not in this ticket: the Holdout report hash check (ticket 07).
