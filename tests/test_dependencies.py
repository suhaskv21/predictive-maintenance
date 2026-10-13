"""Guard: the wheel's declared dependencies are installable in marimo's Pyodide (ADR 0012)."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())

# Packages marimo's Pyodide bundles (ADR 0009 table); micropip installs any other declared
# dependency from PyPI, which can fail in the browser.
PYODIDE_BUNDLED = {"joblib", "lightgbm", "numpy", "polars", "scikit-learn", "scipy", "xgboost"}


def _name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match is not None
    return match.group(0).lower().replace("_", "-")


def test_declared_dependencies_name_only_pyodide_bundled_packages() -> None:
    names = {_name(r) for r in PYPROJECT["project"]["dependencies"]}
    assert names <= PYODIDE_BUNDLED, sorted(names - PYODIDE_BUNDLED)


def test_streamlit_is_an_optional_extra_not_a_main_dependency() -> None:
    assert "streamlit" not in {_name(r) for r in PYPROJECT["project"]["dependencies"]}
    extra = PYPROJECT["project"]["optional-dependencies"]["streamlit"]
    assert {_name(r) for r in extra} == {"streamlit"}


def test_marimo_is_pinned_exactly_in_the_dev_group() -> None:
    assert "marimo==0.25.1" in PYPROJECT["dependency-groups"]["dev"]
