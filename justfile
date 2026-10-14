default:
    @just --list

install:
    uv sync

test:
    uv run pytest

lint:
    uv run ruff check . && uv run ruff format --check .

format:
    uv run ruff check --fix . && uv run ruff format .

typecheck:
    uv run mypy

check: lint typecheck test

train:
    uv run python -m predictive_maintenance.training.train

# One-off: scores the Holdout once. A second run refuses; rescoring needs a new ADR.
holdout:
    uv run python -m predictive_maintenance.evaluation.holdout

# Build check: the committed Holdout report belongs to the Pipeline in models/.
holdout-check:
    uv run python -m predictive_maintenance.evaluation.holdout --check

# After `just train`: check the report, then make operating-metrics quote the Holdout figures.
holdout-apply:
    uv run python -m predictive_maintenance.evaluation.holdout --apply

# The one-screen marimo app (ADR 0012) in app mode.
marimo-app:
    uv run marimo run marimo_app.py

# Rehearse the Pages build into build/site (as pages.yml does); needs `just train` first.
site:
    rm -rf dist build
    uv build --wheel
    uv run python -m predictive_maintenance.deploy.stage --wheel dist/*.whl --release "$(git describe --tags --always)"
    uv run marimo export html-wasm build/site-src/marimo_app.py -o build/site --mode run
    uv run python -m predictive_maintenance.deploy.contract --wheel dist/*.whl --site build/site

notebook:
    uv run marimo edit notebooks
