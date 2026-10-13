# Reports

Committed results (unlike `models/`, which is git-ignored). `figures/` is generated and ignored.

- `holdout-metrics.json`: the one Holdout result (ADR 0007 rule 9, ADR 0012). Written by
  `just holdout` from the CI-built artifact: Pipeline SHA-256, date, and recall, precision, expected
  cost per 1,000 at the default Cost ratio, PR-AUC, ROC-AUC and Brier, each with a percentile 95 %
  stratified-bootstrap interval (resamples from `configs/`, seed stream 11). The threshold is the
  frozen Operating threshold, never re-chosen.
  - A second `just holdout` refuses. Rescoring needs a new ADR, then `--rescore`.
  - `just holdout-check` (for builds) fails with "holdout figures belong to a different model" if
    the report's hash is not the Pipeline's in `models/`. If builds prove not bit-identical, switch
    that check to a fingerprint (pinned versions, scores on fixed train Readings within 1e-9),
    never to a rescore.
