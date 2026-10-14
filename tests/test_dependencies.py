"""Guard: the wheel's declared dependencies are installable in marimo's Pyodide (ADR 0012)."""

import tomllib
from pathlib import Path

from predictive_maintenance.deploy.contract import PYODIDE_BUNDLED, requirement_name

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_declared_dependencies_name_only_pyodide_bundled_packages() -> None:
    names = {requirement_name(r) for r in PYPROJECT["project"]["dependencies"]}
    assert names <= PYODIDE_BUNDLED, sorted(names - PYODIDE_BUNDLED)


def test_streamlit_is_an_optional_extra_not_a_main_dependency() -> None:
    assert "streamlit" not in {requirement_name(r) for r in PYPROJECT["project"]["dependencies"]}
    extra = PYPROJECT["project"]["optional-dependencies"]["streamlit"]
    assert {requirement_name(r) for r in extra} == {"streamlit"}


def test_marimo_is_pinned_exactly_in_the_dev_group() -> None:
    assert "marimo==0.25.1" in PYPROJECT["dependency-groups"]["dev"]
