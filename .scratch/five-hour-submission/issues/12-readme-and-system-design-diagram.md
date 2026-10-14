# 12: README and system-design diagram

**What to build:** A reader can understand, run, reproduce and critique the system from the README, and can see in one diagram what is built and what a production system would add.

Parent: `.scratch/five-hour-submission/spec.md` (spec §4.1, §4.4, §5, §8).

**Blocked by:** 05 (Build workflow and first live deploy), 07 (Holdout from the CI-built artifact)

**Status:** resolved

- [x] README opens with the live URL, a one-paragraph problem and users statement, and how predictions reach engineers and what a useful outcome looks like.
- [x] It states every spec §8 decision with its rationale: data source, target, features, excluded columns, missing-data handling (`reject`), split and validation, primary metric and threshold, model and interpretability trade-off, the output shown and how to act on it, deployment provider and limitations.
- [x] It reports the Holdout results with their intervals, the metric rationale, and the implications of class imbalance and false-negative and false-alarm costs, with the Cost ratio labelled as an assumption.
- [x] Setup and run instructions: clean install, the train command, the Holdout command, seeds, data location and expectations, and how to run the app.
- [x] A production-readiness critique: how the system would be monitored; at least two defined metrics (refusal rate and Flag rate against expected) plus post-outcome precision and recall with the label delay; a retraining strategy with validation on a fresh holdout and a shadow period before replacing the live model; deployment risks with mitigations (including the ADR 0010 list: heavy first load, Pyodide host dependency, Pyodide moves, public model files, per-browser state, stale thresholds when pins move, single browser checked).
- [x] A "not implemented" list covering the ADR 0012 deferred items and the refit on the full data, and a pointer to ADR 0012.
- [x] A Mermaid system-design diagram covering ingestion, training, serving and monitoring, with built parts solid and described-only production additions dashed, linked from the README.
- [x] The README states the browser evidence from ticket 11 plainly, including its limits.
- [x] `just check` passes and its result is shown.
