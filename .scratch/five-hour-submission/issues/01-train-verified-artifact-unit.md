# 01: Train a verified artifact unit

**What to build:** One command (`just train`) trains the default model (random forest, ADR 0006) on the train partition only, with the seed from the config, and writes the five-file artifact unit that every later ticket consumes. After it runs, a person can see a Pipeline, its Operating threshold and its figures on disk, all consistent with each other and with the accepted 0.218.

Parent: `.scratch/five-hour-submission/spec.md` (Seam 2: the train output).

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] The command reads only the train partition from the existing stratified 70/30 split; a test proves the Holdout is never read.
- [ ] Preprocessing and model are serialized together as one scikit-learn Pipeline; failure-mode columns and identifiers are absent from its inputs.
- [ ] The Operating threshold is re-derived by the ADR 0007 method (per CV repeat, median across repeats, 10:1) and the command fails if it differs from the accepted figure by more than the configured tolerance.
- [ ] The unit has five files: Pipeline; OperatingPoint (model, default ratio, thresholds per ratio, no bands, Risk floor); missing-data policy (strategy `reject`, empty reference, marked "not measured in this build"); operating-metrics (CV figures per Cost ratio labelled `cv_train`, plus the training min/max of each input); manifest (model name, library versions, SHA-256 of the other four files, and scores of a few fixed train Readings for the golden check).
- [ ] New config keys `[deploy]` (accepted thresholds, tolerance, site budget) and `[app] example_failures = 3` are added and documented in the config README; the single seed source is unchanged.
- [ ] Artifacts are never committed; the models directory stays ignored.
- [ ] Mirrored tests run the command on a small fixture dataset and assert on the produced unit only (files, hashes, versions, labels, threshold within tolerance, no Holdout read).
- [ ] `just check` passes and its result is shown.
