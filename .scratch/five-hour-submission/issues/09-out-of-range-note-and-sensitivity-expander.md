# 09: Out-of-range note and sensitivity expander

**What to build:** Two honesty aids on the result. A Reading inside the plausible ranges but outside what the model was trained on is still scored, with a note that the result is unreliable. An expander shows how the Flag would behave at 5:1, 10:1 and 20:1 Cost ratios, metrics only. Cuttable under the ADR 0012 stop rule (the expander is the first cut).

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 04 (One-screen marimo app (local))

**Status:** resolved

- [x] A value outside the training min/max recorded in the operating-metrics, but inside the plausibility range, is scored normally and the result carries the note "outside the range of the training data, treat the result as unreliable"; the note passes the banned-wording test and does not appear for in-range Readings.
- [x] The expander (read-only) shows recall, precision and Flags per 1,000 at 5:1, 10:1 and 20:1 for the model, with no thresholds; the forest's 20:1 row is marked "not precise".
- [x] The Flag still uses 10:1 only and the engineer cannot change the Cost ratio.
- [x] Wording comes from the Presentation; the notebook gains no logic.
- [x] Mirrored tests cover the note on both sides of the training range and the expander's rows and marking.
- [x] `just check` passes and its result is shown.

## Comments

Implemented: `Assessment.outside_training` (inputs beyond the training min/max recorded in operating-metrics; still scored),
`range_note`, `SensitivityRow` and the sensitivity fields in `presentation.py` (rows read from the `sensitivity` block once
the Holdout is applied; the forest's 20:1 row carries "not precise"), and rendering in `marimo_app.py`. Tests in
`test_assessment.py`, `test_presentation.py` and `test_marimo_app.py`. `just check`: ruff, mypy strict, 726 tests pass.

Review notes: `assess()` raises if operating-metrics has no `training_ranges` rather than scoring without the note. After
the Holdout is applied the 10:1 expander row is the CV figure while "What this Flag means" quotes the Holdout figure; both
are labelled with their source. Follow-up UI changes beyond this ticket: single-column layout, "What this Flag means" and
"Model card" expanders, and removal of the "Synthetic benchmark data" caveat (ADR 0009 and spec story 21 still list it;
a superseding ADR is not yet written). Not checked: the layout in a real browser.
