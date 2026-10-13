# Configs

Run configuration, loaded by `predictive_maintenance.config.load_config`.

- `default.toml` — `seed`: the single source of randomness for the pipeline. `[split]`: the split and
  validation settings decided in ADR 0005 (`test_size` = test share of the stratified holdout,
  `cv_folds` × `cv_repeats` = repeated stratified CV on the train partition). Further settings
  (hyperparameters, threshold cost ratio) are added when the ADRs that decide them are accepted.
  `load_config` requires every setting; the file is the only place the values live.
