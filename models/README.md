# Models

Serialized pipeline artifacts (preprocessing + model together) are written here by the training command. Git-ignored by default.

Note: the Pages build (`.github/workflows/pages.yml`) trains the artifact in CI and publishes it with the site, so nothing in this folder is committed.

`operating-metrics.json` is written by training with CV figures (`source: cv_train`). The build then
applies the committed Holdout report (`reports/holdout-metrics.json`, `just holdout-apply`), which
relabels the figures `holdout` with intervals and keeps the CV figures under `sensitivity`.

`manifest.json` also holds the Example readings the app offers (`examples`): drawn by training from
the train partition only, with the config seed, `[app] example_failures` recorded failures and the
rest non-failures. Each has the six inputs and its `Machine failure` outcome, so the app needs no
dataset.

`manifest.json` records `git`, the commit the model was trained from (ADR 0015). Training refuses a
dirty working tree; with `--allow-dirty` it records `null`, and the build contract rejects such a unit.
