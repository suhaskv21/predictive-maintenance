# Models

Serialized pipeline artifacts (preprocessing + model together) are written here by the training command. Git-ignored by default.

Note: Streamlit Community Cloud deploys from the repository, so either the final artifact must be committed (force-add, record in an ADR) or the app must build it on first start.

`operating-metrics.json` is written by training with CV figures (`source: cv_train`). The build then
applies the committed Holdout report (`reports/holdout-metrics.json`, `just holdout-apply`), which
relabels the figures `holdout` with intervals and keeps the CV figures under `sensitivity`.
