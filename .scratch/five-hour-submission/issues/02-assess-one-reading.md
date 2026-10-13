# 02: Assess one Reading from the unit

**What to build:** The inference layer loads and checks the unit from ticket 01 and assesses a single Reading: a Refused reading when any input is absent or unusable, otherwise a Flag decided by the model's own Operating threshold on the unrounded score. It imports no UI framework and writes no log. A test or a REPL call with six inputs returns an assessment record.

Parent: `.scratch/five-hour-submission/spec.md` (Seam 1).

**Blocked by:** 01 (Train a verified artifact unit)

**Status:** ready-for-agent

- [ ] The loader checks file presence, names, SHA-256 and library versions; any failure disables the unit with a one-line reason and never scores with a fallback threshold; a failed default model yields an error state, not another model.
- [ ] The six inputs (Type and five sensors) are classified as present or unusable: absent, not a number, not finite, or out of range per the new `[plausibility]` config table (air 200–400 K, process 200–450 K, speed above 0 to 10,000 rpm, torque 0–500 Nm, wear 0–1,000 min, Type L, M or H). The table is documented in the config README.
- [ ] Any Incomplete reading gives status `refused`, each missing input with its reason, a refusal reason, and `recoverable` false; no Failure risk, Flag, threshold or score is returned.
- [ ] A complete Reading gives status `complete`, the unrounded Failure risk and the Flag from the model's own threshold; a Reading whose rounded percent equals the threshold's rounded percent is still decided on the unrounded score.
- [ ] A complete Reading's score from the loaded unit equals the Pipeline scored directly (bit-identical).
- [ ] Mirrored tests cover each of the six inputs in each unusable form, both sides of every range limit, and each artifact failure mode.
- [ ] `just check` passes and its result is shown.
