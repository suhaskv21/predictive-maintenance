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

- `[missing_data]` (ADR 0008, Proposed): margins for the missing-data policy. All are judgement
  calls fixed before any result.
  - `min_retained_saving`: a pattern of absent inputs is scorable only if, at the frozen Operating
    threshold, it keeps at least this share of the complete-data saving over *flag nothing* (after
    subtracting one corrected SE).
  - `max_recall_drop`: largest absolute fall in recall allowed for a scorable pattern.
  - `flag_rate_ratio_min`, `flag_rate_ratio_max`: allowed band for the Flag rate relative to
    complete Readings.
  - `challenger_margin`: retained saving a more complex strategy must add over a simpler one.
  - `stress_speed_quantile`: what counts as "high speed" in the torque-absent-at-high-speed stress.
  - `assumed_missing_rates`: per-input missing rates (ascending, in (0, 1)) for the data-loss check.
    These are **assumptions**: the raw data has no nulls.
- `[pipeline_health]` (ADR 0008): review triggers for the data-pipeline health tab. Defaults set by
  the project owner, not measured; the plant can change them. Rates are shares of all Readings.
  - `review_window_weeks`, `min_weekly_readings`: a trigger needs this many consecutive weeks, each
    with at least this many Readings.
  - `recoverable_refusal_rate`: review building imputation when refusals the measured imputation
    would have scored reach this share (and are not concentrated, see below).
  - `unrecoverable_refusal_rate`: at or above this share, imputation is not the lever.
  - `top_sources`, `max_source_concentration`: refusals held by the `top_sources` most-refusing
    sources at or above this share mean a broken sensor to fix, not an imputation problem.
  - `informative_band_share`, `informative_failure_rate_ratio`: absence looks informative when one
    training-quartile band holds more than this share of refused Readings, or refused Readings fail
    this many times as often as complete ones.
- `[failure_modes]` (ADR 0009, Proposed): the pre-declared rule for the per-mode predictability
  check. Judgement calls fixed before any number. Failure modes are never features.
  - `min_pr_auc`: smallest mean repeated-CV PR-AUC for a mode to count as predictable.
  - `min_lift`: smallest ratio of that PR-AUC to the mode's prevalence. The PR-AUC must also exceed
    the prevalence by more than one corrected standard error.
- `[app]` (ADR 0009): `example_count` = how many Example readings the app offers, all
  drawn from the train partition (never the holdout); `example_failures` = how many of those are
  recorded failures (the rest are non-failures).
- `[deploy]` (ADR 0010, ADR 0012): checks the training command and the build enforce.
  - `accepted_thresholds`: the Operating threshold accepted in ADR 0007 at the default cost ratio,
    per model name (forest 0.218). Training re-derives it and fails if it differs by more than
    `threshold_tolerance`, so a silent drift in the model or library versions fails loudly.
  - `threshold_tolerance`: largest allowed difference, compared after rounding the re-derived
    threshold to 3 places (the precision the ADR quotes).
  - `site_budget_mb`: largest exported site, in MB, the build accepts.
- `[plausibility]` (ADR 0013, Proposed): which input values are physically possible at assessment time. Only
  impossible values are treated as absent (reason "out of range"), and the Reading is refused. A
  value inside these ranges but outside the training data's range is still scored. Each numeric
  input is a table with `max` and either `min` (inclusive) or `above` (exclusive):

  | Key | Input | Valid range |
  |---|---|---|
  | `air_temperature` | Air temperature [K] | 200 to 400 |
  | `process_temperature` | Process temperature [K] | 200 to 450 |
  | `rotational_speed` | Rotational speed [rpm] | above 0 to 10,000 |
  | `torque` | Torque [Nm] | 0 to 500 |
  | `tool_wear` | Tool wear [min] | 0 to 1,000 |

  `types` lists the valid product Types (`L`, `M`, `H`; any of the dataset's Types). Anything else
  is "out of range". Absent, not a number and not finite values are refused whatever the ranges.

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
