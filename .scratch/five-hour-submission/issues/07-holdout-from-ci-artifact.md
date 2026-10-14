# 07: Holdout from the CI-built artifact

**What to build:** The submission has one committed Holdout result that belongs to the deployed file, and the deployed app quotes it. Builds read the report and fail if the rebuilt Pipeline is a different model.

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 05 (Build workflow and first live deploy), 06 (Holdout command)

**Status:** resolved

- [x] The Holdout command is run once against the artifact built by CI (not a local build) and the resulting report is committed.
- [x] The build workflow gains the hash check from ticket 06: it fails with "holdout figures belong to a different model" if the rebuilt Pipeline's hash differs, and never rescores.
- [x] The operating-metrics for the forest show the Holdout figures labelled `holdout` (with intervals), and the app's Flag meaning text quotes them with that label; the CV-labelled figures remain only in the sensitivity data.
- [x] If CI and a local build are not bit-identical, record the observation and switch the check to the fingerprint fallback described in ticket 06 (and write a Proposed ADR for it), never to a rescore.
- [x] The Holdout figures with intervals are recorded in the ticket's comments for the README.
- [x] `just check` passes and its result is shown.

## Comments

Implemented (not committed): `evaluation/holdout.py` gains `apply_report` / `--apply` (`just holdout-apply`); `bootstrap_metric_cis`
gains `flags_per_1000` and `false_alarms_per_caught` (the Flag meaning quotes them; the RNG stream is unchanged);
`.github/workflows/pages.yml` runs `holdout --check` then `holdout --apply` right after training, before the wheel and staging.
`just check`: ruff, format, mypy strict clean; 652 tests pass.

**Holdout run (2026-10-14), once, against the CI-built artifact.** The unit was downloaded from the deployed site
(`.../public/models/`, build `c314f96`): Pipeline SHA-256 `c55639bc96772158806b8201e61d845323508063feef1bb8954157940aef32b3`.
Report: `reports/holdout-metrics.json` (untracked until committed). A second `just holdout` refused; `--check` passes.

**Bit-identical observation.** A fresh local `just train` produced the same SHA-256 for the Pipeline and byte-identical copies
of all four other files as the CI build. So the hash check stands; the fingerprint fallback and its ADR were not needed.

**Holdout figures for the README** (3,000 Readings, 102 failures, threshold 0.218, Cost ratio 10:1, 2,000 stratified bootstrap
resamples, percentile 95 % intervals):

| Metric | Estimate | 95 % interval |
|---|---|---|
| Recall | 0.863 | 0.794 to 0.922 |
| Precision | 0.830 | 0.766 to 0.899 |
| Flags per 1,000 | 35.3 | 31.7 to 39.0 |
| False alarms per caught failure | 0.205 | 0.112 to 0.306 |
| Expected cost per 1,000 (false-alarm units) | 52.7 | 31.3 to 76.3 |
| PR-AUC | 0.914 | 0.867 to 0.955 |
| ROC-AUC | 0.986 | 0.971 to 0.996 |
| Brier | 0.00696 | 0.00504 to 0.00909 |

**Shape of operating-metrics after apply:** `source: holdout`; `ratios["10.0"]` holds the four quoted figures plus `intervals`;
the CV figures for 5, 10 and 20 sit only under `sensitivity` (`source: cv_train`) for the ticket 09 expander. The Flag meaning
reads "Figures from held-out test data." (the glossary label for `holdout`); intervals are in the file and README, not the screen.

**Not done here:** the live site still shows CV figures until the next `TRIGGER_CI` push (this work is uncommitted). The report
must be committed in the same push as the workflow change, or the new check step fails with "No such file".
