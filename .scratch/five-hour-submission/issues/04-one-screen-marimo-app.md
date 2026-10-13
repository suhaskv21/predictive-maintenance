# 04: One-screen marimo app (local)

**What to build:** An engineer opens the app, enters the six inputs and sees Inspect, No inspection flagged or Not assessed with the words from ticket 03. It runs locally in app mode and renders strings only; it holds no threshold, rounding, wording or preprocessing logic. Beside it, the dependency changes that make the browser build possible.

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 03 (Presentation and banned-wording test)

**Status:** resolved

- [x] First step: verify how marimo's WebAssembly export installs a local wheel and its declared dependencies, and record the finding in the ticket's comments.
- [x] A marimo entry notebook at the repository root, beside the Streamlit stub, with the input form (Type as a dropdown, five numeric inputs), the three headline states, the error state, and a footer showing the build commit.
- [x] The notebook imports only the inference layer's loader, assessment and Presentation; no logic of its own; excluded from the linter (approved in ADR 0012).
- [x] `streamlit` moves from the main dependencies to an optional extra named `streamlit`; the app command in the justfile uses the extra; the Streamlit stub still starts.
- [x] `marimo` is pinned to exactly `0.25.1` in the dev group; `uv.lock` is re-locked and library pins are otherwise unchanged.
- [x] The built wheel's declared dependencies name only packages Pyodide bundles.
- [x] A complete Reading, a Reading with one input absent, and one with an out-of-range value each show the right state when run locally.
- [x] `just check` passes and its result is shown.

## Comments

Implemented (not committed): `marimo_app.py`, `tests/test_marimo_app.py`, `tests/test_dependencies.py`, `streamlit` extra,
`marimo==0.25.1`, `uv.lock` re-locked (only the streamlit and marimo entries changed), ruff excludes `marimo_app.py`,
`just marimo-app`. `just check`: ruff, mypy strict, 610 tests pass.

**WASM install finding** (read from marimo 0.25.1's source, `_export/local_wheels.py` and
`_runtime/packages/_micropip_streaming.py`; not run in a browser): `html-wasm` export reads PEP 723 metadata in the
notebook; an entry such as `predictive-maintenance @ ./dist/<wheel>.whl` is copied to `public/wheels/` and rewritten to
`../public/wheels/<wheel>.whl`, and Pyodide installs it with micropip at worker start. marimo's streaming installer
calls micropip's `Transaction` with `deps=True`, so the wheel's `Requires-Dist` ARE resolved: anything not bundled in
Pyodide comes from PyPI. That is why `streamlit` had to leave the main dependencies. The built wheel now declares only
joblib, lightgbm, polars, scikit-learn and xgboost (all bundled); `tests/test_dependencies.py` guards it.

For ticket 05 (not done here): add the PEP 723 header with the wheel reference at export time; ship `models/` and
`configs/` so `mo.notebook_dir()` finds them in the browser; stamp `BUILD_COMMIT` in `marimo_app.py`. Checked locally:
complete, one-input-absent (each of the six) and out-of-range Readings give Inspect, Not assessed and Not assessed;
`marimo run` and `just app` (Streamlit with the extra) both start. Not checked: clearing a number field in the browser
yields an absent value (the notebook relies on `mo.ui.number` returning None).

Review notes: the notebook also imports `config` (plausibility ranges, Cost ratio, product Types) and `data.schema`
(input names), which the checkbox does not list; `assess` and `present` need them. It branches on `DisabledUnit`
to pick `present_unavailable`, and maps a state to a callout colour (style only). Form defaults and `MODEL` are in
the notebook. A wheel-level check of `Requires-Dist` (the built file, not `pyproject.toml`) belongs to ticket 05's
contract checks. An emptied number field arrives as `""` (verified on `mo.ui.number._convert_value`), which the
assessment treats as absent; a test covers it.
