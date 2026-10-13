# Configs

Run configuration, loaded by `predictive_maintenance.config.load_config`. `load_config` requires every
setting; `default.toml` is the only place the values live. Further settings (hyperparameters) are
added when the ADRs that decide them are accepted.

- `seed`: the single source of randomness for the pipeline.
- `[split]` (ADR 0005): `test_size` = test share of the stratified holdout; `cv_folds` × `cv_repeats`
  = repeated stratified CV on the train partition.
- `[threshold]` (ADR 0007, Proposed):
  - `cost_ratios`: **assumed** cost ratios, shown side by side (ascending).
  - `default_cost_ratio`: the one ratio that sets each model's operating threshold. It must be one of
    `cost_ratios` and larger than the smallest.
- `[evaluation]` (ADR 0007, Proposed): `bootstrap_resamples` = stratified bootstrap resamples for
  confidence intervals on the final test metrics.

## What a cost ratio means

A **cost ratio** says how many **false alarms** one **missed failure** is worth.

- A missed failure is a Machine failure the model did not flag: unplanned downtime.
- A false alarm is a flagged Reading that did not fail: one unnecessary inspection.

`default_cost_ratio = 10` means "missing one real failure is as costly as 10 false alarms". A higher
ratio makes the operating threshold lower, so more Readings are flagged and fewer failures are
missed, at the price of more false alarms.

The ratio is an **assumption, not a measurement**. The AI4I 2020 data has no cost information, and
only the ratio matters (absolute currency amounts are not needed). It belongs to maintenance
management: replace it with the ratio from real downtime and inspection costs before relying on the
flags. Changing it changes every model's operating threshold, so retrain after editing it.
