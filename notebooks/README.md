# Notebooks

[marimo](https://marimo.io) notebooks (plain `.py`, diff-friendly). Run with `just notebook`.

Rules:
- Exploration and communication only; the pipeline must never depend on them.
- Import logic from `predictive_maintenance`; do not duplicate it.
- Naming: `NN_topic.py` (e.g. `01_eda.py`).
