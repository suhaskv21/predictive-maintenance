# 0013. Plausibility ranges for the six inputs

- Status: Accepted
- Date: 2026-10-13
- Deciders: Suhas (agreed in a grill session and accepted the document 2026-10-13); drafted by an agent
- Refines: [0008](0008-missing-data-policy.md) and [0012](0012-five-hour-submission-scope.md), which list plausibility ranges as not decided

## Context
Assessment treats an unusable input (not a number, not finite, or physically impossible) as absent, so the
Reading is refused rather than scored (GLOSSARY: Incomplete reading). ADRs 0008, 0009 and 0012 leave the
physical limits undecided. Scoring a value no sensor could produce, such as a negative speed, would give a
confident-looking Failure risk from garbage.

## Options considered
1. Refuse values outside the training data's min/max — strict, but refuses valid machines that run slightly
   beyond the 10,000-row sample, and the sample is small for the tails.
2. Refuse only physically impossible values; score the rest and note when a value is outside the training
   range — refuses garbage, keeps real extremes assessable, and tells the engineer when the model is
   extrapolating.
3. No range check — nothing is refused for being absurd.

## Decision
Option 2. Valid ranges, held in the `[plausibility]` table of `configs/default.toml` and documented in
`configs/README.md`, not in code:

| Input | Valid range |
|---|---|
| Air temperature [K] | 200 to 400 |
| Process temperature [K] | 200 to 450 |
| Rotational speed [rpm] | above 0 to 10,000 |
| Torque [Nm] | 0 to 500 |
| Tool wear [min] | 0 to 1,000 |
| Type | L, M or H |

A value outside its range is reported with the reason "out of range". Values inside the ranges but outside
the training data's min/max are still scored, with a non-refusing note on the result.

## Consequences
The ranges are generous physical limits, not measured: they are judgement calls and can be tightened per
site by editing the config. Changing the approach, for example to refuse values outside the training range
(option 1), needs a new ADR that supersedes this one.
