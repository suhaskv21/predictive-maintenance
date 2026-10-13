# 05: Build workflow and first live deploy

**What to build:** A separate build workflow that, on request, trains in CI, builds the wheel, exports the app as a static WebAssembly site, checks the build contract and deploys it to GitHub Pages. A pushed commit with the trigger token produces a public URL that loads and assesses a Reading.

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 04 (One-screen marimo app (local))

**Status:** ready-for-agent

- [ ] Owner prerequisite, needed before the first deploy: enable Pages with source GitHub Actions (Settings, Pages). The agent states this clearly and stops if the deploy step reports Pages is disabled.
- [ ] The workflow runs only when the head commit message contains `TRIGGER_CI` (any case) or on manual dispatch; no path filter and no pull-request trigger. The existing check workflow is untouched and stays automatic.
- [ ] Steps: locked sync; full check; train; build the wheel; export the site as a directory in WebAssembly mode; contract checks; upload and deploy. Pages write permission only on the deploy job; read-only elsewhere.
- [ ] Contract checks fail the build on: a missing unit file or manifest mismatch; a re-derived threshold outside tolerance of the accepted figure; a site over the configured budget; training reading the Holdout; wheel dependencies not bundled in Pyodide.
- [ ] The footer shows the deployed build commit.
- [ ] After the first triggered push the public URL loads and assesses one Reading; the URL is recorded in the ticket's comments.
- [ ] The time of the cold sync and the training step is recorded; the 3-minute trigger for revisiting the XGBoost swap is noted if exceeded.
- [ ] `just check` passes and its result is shown.
