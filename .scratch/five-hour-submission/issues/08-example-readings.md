# 08: Example readings

**What to build:** An engineer can try the app without typing: pick one of 10 Example readings from the training data (3 recorded failures, 7 non-failures), assess it, and see its recorded outcome quietly, with the reminder that agreement proves nothing. Cuttable under the ADR 0012 stop rule (second cut).

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 04 (One-screen marimo app (local))

**Status:** ready-for-agent

- [ ] The selection draws from the train partition only, never the Holdout, with the config seed, per outcome class using `[app] example_failures` and the remaining count from `example_count`; it fails loudly if the train partition has fewer failures than requested.
- [ ] Choosing an Example reading fills the form; assessing it shows the outcome line from the Presentation naming only `Machine failure` (never the five mode flags).
- [ ] Editing any input clears the outcome line.
- [ ] A test asserts no Holdout Reading can be selected, and another asserts the outcome line text passes the banned-wording test.
- [ ] `just check` passes and its result is shown.
