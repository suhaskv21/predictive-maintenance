# 07: Holdout from the CI-built artifact

**What to build:** The submission has one committed Holdout result that belongs to the deployed file, and the deployed app quotes it. Builds read the report and fail if the rebuilt Pipeline is a different model.

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 05 (Build workflow and first live deploy), 06 (Holdout command)

**Status:** ready-for-agent

- [ ] The Holdout command is run once against the artifact built by CI (not a local build) and the resulting report is committed.
- [ ] The build workflow gains the hash check from ticket 06: it fails with "holdout figures belong to a different model" if the rebuilt Pipeline's hash differs, and never rescores.
- [ ] The operating-metrics for the forest show the Holdout figures labelled `holdout` (with intervals), and the app's Flag meaning text quotes them with that label; the CV-labelled figures remain only in the sensitivity data.
- [ ] If CI and a local build are not bit-identical, record the observation and switch the check to the fingerprint fallback described in ticket 06 (and write a Proposed ADR for it), never to a rescore.
- [ ] The Holdout figures with intervals are recorded in the ticket's comments for the README.
- [ ] `just check` passes and its result is shown.
