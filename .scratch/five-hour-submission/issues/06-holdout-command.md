# 06: Holdout command

**What to build:** A one-off command scores the Holdout once, for the forest, with the frozen Operating threshold, and writes a report of the unbiased result with bootstrap intervals. A second run without the explicit flag refuses, so the Holdout cannot be used to tune anything.

Parent: `.scratch/five-hour-submission/spec.md` (ADR 0012, "The one holdout result"; ADR 0007 rule 9).

**Blocked by:** 01 (Train a verified artifact unit)

**Status:** resolved

- [x] The command loads a unit, scores the Holdout once and writes a report with the Pipeline's SHA-256, the date, and recall, precision, expected cost at the default Cost ratio, PR-AUC, ROC-AUC and Brier, each with a percentile 95 % interval from a stratified bootstrap (2000 resamples, the ADR 0007 seed stream) in which the threshold is never re-chosen.
- [x] The report lives under the reports directory, which is committed (not the models directory).
- [x] If a report already exists, the command refuses to rescore unless an explicit flag is passed; the message says a rescore needs a new ADR.
- [x] A check, usable by the build, that compares the report's hash with a given Pipeline and fails with "holdout figures belong to a different model" on mismatch; documented fingerprint fallback (pinned versions plus scores on fixed train Readings within 1e-9) is noted for use only if builds are not bit-identical, never a rescore.
- [x] Mirrored tests run the command on the fixture dataset and assert on the report (fields, hash, intervals), the refusal on a second run and the hash-mismatch failure.
- [x] `just check` passes and its result is shown.
