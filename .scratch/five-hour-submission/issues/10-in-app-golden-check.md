# 10: In-app golden check

**What to build:** When the app loads in the browser it scores a few embedded train Readings and compares them with the scores stored in the manifest, and shows a visible banner if they differ. A visitor can thereby see a browser scoring difference that would otherwise be silent. Cuttable under the ADR 0012 stop rule (third cut).

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 04 (One-screen marimo app (local))

**Status:** resolved

- [x] The check, in the inference layer, loads the golden Readings and scores from the manifest and compares them (tolerance 1e-9) with fresh scores from the loaded Pipeline; it returns pass or a list of differences.
- [x] The app shows a visible failure banner on mismatch (and the failure wording passes the banned-wording test); on pass it shows nothing intrusive.
- [x] A test forces a mismatch (altered stored score) and asserts the failure result and banner text; another asserts a pass on a fresh unit.
- [x] `just check` passes and its result is shown.

## Comments

Implemented (not committed): `inference/golden.py` (`check_golden`, `GoldenResult`; tolerance 1e-9; an absent, empty or
malformed golden block is a failure, never a pass), `golden_banner` in `presentation.py` (scanned by
`banned_wording_in`; quotes up to three differences), one cell and one banner in `marimo_app.py`, tests in
`test_golden.py`, `test_presentation.py` and `test_marimo_app.py`. `just check`: ruff, mypy strict, 670 tests pass.
The shipped `models/` unit passes its own check. A Disabled unit skips the check (no Pipeline to score).

Review notes: the banner says the Failure risk "may be wrong" but the app still shows the Presentation below it; hiding
the verdict on a failed check was not asked for. Not checked: the banner in a real browser build.
