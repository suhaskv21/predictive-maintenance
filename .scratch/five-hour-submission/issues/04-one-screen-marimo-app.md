# 04: One-screen marimo app (local)

**What to build:** An engineer opens the app, enters the six inputs and sees Inspect, No inspection flagged or Not assessed with the words from ticket 03. It runs locally in app mode and renders strings only; it holds no threshold, rounding, wording or preprocessing logic. Beside it, the dependency changes that make the browser build possible.

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 03 (Presentation and banned-wording test)

**Status:** ready-for-agent

- [ ] First step: verify how marimo's WebAssembly export installs a local wheel and its declared dependencies, and record the finding in the ticket's comments.
- [ ] A marimo entry notebook at the repository root, beside the Streamlit stub, with the input form (Type as a dropdown, five numeric inputs), the three headline states, the error state, and a footer showing the build commit.
- [ ] The notebook imports only the inference layer's loader, assessment and Presentation; no logic of its own; excluded from the linter (approved in ADR 0012).
- [ ] `streamlit` moves from the main dependencies to an optional extra named `streamlit`; the app command in the justfile uses the extra; the Streamlit stub still starts.
- [ ] `marimo` is pinned to exactly `0.25.1` in the dev group; `uv.lock` is re-locked and library pins are otherwise unchanged.
- [ ] The built wheel's declared dependencies name only packages Pyodide bundles.
- [ ] A complete Reading, a Reading with one input absent, and one with an out-of-range value each show the right state when run locally.
- [ ] `just check` passes and its result is shown.
