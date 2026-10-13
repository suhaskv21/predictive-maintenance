# 09: Out-of-range note and sensitivity expander

**What to build:** Two honesty aids on the result. A Reading inside the plausible ranges but outside what the model was trained on is still scored, with a note that the result is unreliable. An expander shows how the Flag would behave at 5:1, 10:1 and 20:1 Cost ratios, metrics only. Cuttable under the ADR 0012 stop rule (the expander is the first cut).

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 04 (One-screen marimo app (local))

**Status:** ready-for-agent

- [ ] A value outside the training min/max recorded in the operating-metrics, but inside the plausibility range, is scored normally and the result carries the note "outside the range of the training data, treat the result as unreliable"; the note passes the banned-wording test and does not appear for in-range Readings.
- [ ] The expander (read-only) shows recall, precision and Flags per 1,000 at 5:1, 10:1 and 20:1 for the model, with no thresholds; the forest's 20:1 row is marked "not precise".
- [ ] The Flag still uses 10:1 only and the engineer cannot change the Cost ratio.
- [ ] Wording comes from the Presentation; the notebook gains no logic.
- [ ] Mirrored tests cover the note on both sides of the training range and the expander's rows and marking.
- [ ] `just check` passes and its result is shown.
