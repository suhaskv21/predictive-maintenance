# 03: Presentation and banned-wording test

**What to build:** Every word an engineer will see comes from one place in the inference layer. Given an assessment, the OperatingPoint and the operating-metrics, it returns a Presentation for each state: Inspect, No inspection flagged and Not assessed (plus the model-unavailable error state), and a test proves none of them overstates certainty.

Parent: `.scratch/five-hour-submission/spec.md` (Seam 1; ADR 0009 wording rules).

**Blocked by:** 02 (Assess one Reading from the unit)

**Status:** resolved

- [x] The headline is the Flag. Failure risk is a secondary line labelled "rough estimate, not calibrated": "under 1 %" below the Risk floor, "about N %" (nearest whole percent, halves up) from the floor to 95 %, "over 95 %" above; "0 %" and "100 %" cannot be produced. The model's own Flag line sits beside it, with the fixed "within a percent can go either way" note.
- [x] The Flag meaning is plain words from the operating-metrics (of 100 failures about N flagged; of 100 Flags about M real; about K per 1,000). A Flagged result states how many Flags are false alarms; a not-flagged result states how many real failures were not flagged.
- [x] Always-on caveats on every state: estimated from historical readings; synthetic benchmark data, not your plant; the Cost ratio assumption read from config, not typed; an aid for prioritising inspections, not a guarantee; the model is named.
- [x] Not assessed is its own style, lists each missing input with its reason, says "No risk or Flag is given. This does not mean the machine is fine. Enter the missing values and assess again; if a sensor is offline, treat this machine as unassessed.", and never shows a Failure risk, Flag, threshold or fallback score. `recoverable` is not shown.
- [x] Presentation also holds the Example-reading outcome line ("Recorded in the data: … This Reading was in the training data, so agreement proves nothing.") for use by ticket 08, and holds the banned-wording list.
- [x] A test scans every rendered state and the always-on caveats for the banned wording ("will fail", "safe", "healthy", "OK", "no risk", "0 %", "100 %", "guaranteed", "predicts failure", unqualified "probability of failure", unsourced accuracy claims) and checks the caveats are present on each state.
- [x] No UI-framework import in this layer; mirrored tests per module.
- [x] `just check` passes and its result is shown.

## Comments

Implemented (not yet committed): `inference/presentation.py` (`present`, `present_unavailable`, `risk_text`,
`flag_line_text`, `example_outcome_line`, `BANNED_WORDING`, `banned_wording_in`) with mirrored
`tests/.../inference/test_presentation.py`. Code review (standards + spec) done; fixed: "0 %"/"100 %" scan
now also catches "100.0 %", non-breaking space and "no-risk"; a unit-failure reason containing banned wording
is replaced by a generic one; test typing tidied. `just check`: ruff, mypy strict, 594 tests pass.

Interpretations to confirm:
- `present` takes the config's `cost_ratio` and raises `ValueError` if it differs from the unit's
  OperatingPoint (a stale unit), rather than quoting the wrong assumption.
- Unknown metrics `source` raises; the only known labels are `cv_train` and `holdout`.
- Figures round to "fewer than 1" / "more than 99" (of 100) so a Flag meaning never reads as none or all.
- Undefined (None) figures show "What a Flag means is not available for this model."
- The "higher number on another model does not mean more risk" half of the ADR note is omitted: one model only.
- Any accuracy claim and "probability of failure" are banned outright, not only unsourced/unqualified.
- The example outcome line is shown only for a complete assessment.
