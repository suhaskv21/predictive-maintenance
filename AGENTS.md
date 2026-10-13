# AGENTS.md — working agreement for humans and AI agents

## Project
Predictive-maintenance ML system (AI4I 2020). Source of truth for requirements: `../TECHNICAL_SPEC.md`.
Priorities: leakage-safe evaluation, reproducibility, clear operational communication — not model score.

## Commands
- `uv sync` — install
- `just check` — lint + typecheck + tests (must pass before declaring work done)
- `uv run pytest tests/predictive_maintenance/<pkg>` — targeted tests

## Conventions
- Source in `src/predictive_maintenance/<stage>/`; tests mirror it at `tests/predictive_maintenance/<stage>/test_<module>.py`. New module => new mirrored test file.
- DataFrames: **polars** (not pandas). Convert to numpy/pandas only at the scikit-learn boundary.
- Pipeline logic lives in the package, never only in notebooks. Notebooks (marimo) import from the package.
- The UI must call `inference/` only; no preprocessing logic in `app/`.
- Split train/test **before** fitting anything. Fit preprocessing on train only. Failure-mode columns (`TWF, HDF, PWF, OSF, RNF`) are never features for `Machine failure`.
- Preprocessing + model are serialized together as one scikit-learn `Pipeline` artifact.
- All randomness is seeded from `configs/` (single seed source).
- Type hints required (mypy strict); public modules/functions have docstrings.

## Decisions
- Any non-trivial decision (model choice, metric, threshold, deployment target, library) gets an ADR in `docs/adr/` (copy `0000-template.md`, next number, status `Proposed`).
- Agents may propose ADRs (status `Proposed`); a human moves them to `Accepted`.
- Do not change an `Accepted` ADR; supersede it with a new one.

## Agent boundaries
- Ask before: adding dependencies, changing the target/feature set/split strategy, deleting files, changing CI or deployment config.
- Never commit secrets, `.env`, or generated artifacts in `models/`.
- Keep changes small and reviewable; one concern per commit.
- Do not claim done without running `just check` and showing the result.

## Agent skills

### Issue tracker

Issues and specs are local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` + `docs/adr/`. See `docs/agents/domain.md`.
