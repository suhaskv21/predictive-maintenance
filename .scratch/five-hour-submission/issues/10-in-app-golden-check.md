# 10: In-app golden check

**What to build:** When the app loads in the browser it scores a few embedded train Readings and compares them with the scores stored in the manifest, and shows a visible banner if they differ. A visitor can thereby see a browser scoring difference that would otherwise be silent. Cuttable under the ADR 0012 stop rule (third cut).

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 04 (One-screen marimo app (local))

**Status:** ready-for-agent

- [ ] The check, in the inference layer, loads the golden Readings and scores from the manifest and compares them (tolerance 1e-9) with fresh scores from the loaded Pipeline; it returns pass or a list of differences.
- [ ] The app shows a visible failure banner on mismatch (and the failure wording passes the banned-wording test); on pass it shows nothing intrusive.
- [ ] A test forces a mismatch (altered stored score) and asserts the failure result and banner text; another asserts a pass on a fresh unit.
- [ ] `just check` passes and its result is shown.
