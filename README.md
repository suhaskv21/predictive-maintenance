# Predictive Maintenance

Failure-risk assessment for industrial machines from a single sensor reading (AI4I 2020 dataset).
Requirements: [`../TECHNICAL_SPEC.md`](../TECHNICAL_SPEC.md) (to be copied to `docs/` once stable).

> Status: scaffold only. No implementation yet.

**Live app:** _TODO (required by spec §4.3)_

## Setup

```bash
uv sync            # install runtime + dev dependencies
just check         # lint, type-check, test
```

## Commands

| Command | Purpose |
|---|---|
| `just test` | Run tests |
| `just lint` | Ruff lint + format check |
| `just typecheck` | mypy (strict) on `src/` |
| `just check` | All of the above |
| `just app` | Run the Streamlit app locally |
| `just notebook` | Open marimo notebook editor |

Training/evaluation commands: _TODO once the pipeline exists._

## Layout

```
src/predictive_maintenance/   package (one subpackage per pipeline stage)
  data/            loading + schema validation of the raw CSV
  preprocessing/   feature selection, encoding, scaling (fit on train only)
  training/        splitting, model fitting, artifact serialization
  evaluation/      metrics, threshold selection, reports
  inference/       load artifact + predict (the only path the UI uses)
  monitoring/      drift / prediction-distribution metrics (design + light impl)
  app/             Streamlit UI code
tests/predictive_maintenance/ mirrors src/ one-to-one (tests/.../x/test_y.py <-> src/.../x/y.py)
streamlit_app.py              thin deployment entrypoint -> predictive_maintenance.app
notebooks/                    marimo notebooks (exploration only, never required to run the pipeline)
docs/adr/                     architecture decision records
docs/design/                  system-design diagram and notes
configs/                      run configuration (seeds, split, hyperparameters)
data/raw/                     immutable source data (ai4i2020.csv)
models/                       serialized pipeline artifacts (generated, git-ignored)
reports/                      generated evaluation output
```

## Working agreement (human + AI agent)

See [`AGENTS.md`](AGENTS.md).
