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

app:
    uv run streamlit run streamlit_app.py

notebook:
    uv run marimo edit notebooks
