# 0014. Training-only libraries move to the dev group; Streamlit is removed

- Status: Accepted
- Date: 2026-10-14
- Deciders: Suhas (accepted 2026-10-13)
- Refines: [0012](0012-five-hour-submission-scope.md) (the `streamlit` extra and the kept
  `streamlit_app.py` stub). Does not change the hosting decision in [0010](0010-deployment.md).

## Context

The marimo WebAssembly app is deployed and tested on GitHub Pages (ADR 0010, 0012). The notebook
declares the built wheel as its dependency, so the wheel's `Requires-Dist` is what the browser may
install. An older device showed noticeable load latency.

- `xgboost` and `lightgbm` are imported lazily, only when training builds those candidates
  (`training/candidates.py`). The shipped `Pipeline` is a random forest that needs only scikit-learn.
- `streamlit_app.py` held one docstring and no code. The `streamlit` extra existed only so the wheel
  did not declare it (ADR 0012). The marimo app is the only app.

## Options considered

1. **Keep both libraries in `dependencies` and the Streamlit stub.** No change; the wheel keeps
   declaring libraries the app never imports.
2. **Move `xgboost` and `lightgbm` to the dev group; delete the stub and the extra** (chosen).
3. **Move only `xgboost`.** Leaves `lightgbm` with the same problem.

## Decision

- `xgboost==2.1.4` and `lightgbm==4.6.0` move to the `dev` group, with the same pins. The wheel now
  declares `joblib`, `polars` and `scikit-learn` only.
- `streamlit_app.py`, the `streamlit` extra, the `just app` recipe and the `.streamlit` ignore line
  are removed. `tests/test_dependencies.py` now asserts that no optional dependencies are declared.
- Accepted ADRs 0009, 0010 and 0012 keep their text, which still mentions Streamlit as an option
  and as a stub; this ADR supersedes those mentions.

## Consequences

- The browser has fewer declared packages to fetch. **Not measured:** whether marimo installs the
  wheel's dependencies (open in ADR 0012), so the latency gain is unconfirmed. Time a cold load.
- A server-side fallback (ADR 0010's next host after molab) would add its entry point again.
- If the model dropdown ships LightGBM or XGBoost in the browser (ADR 0012 deferred item 5), the
  wheel or the notebook must name them again.
- The spec (§4.3) names Streamlit Community Cloud; the README already states that a static
  WebAssembly host is read as an "equivalent service".
