"""Guard: the wheel's declared dependencies are installable in marimo's Pyodide (ADR 0012)."""

import tomllib
from pathlib import Path

from predictive_maintenance.deploy.contract import PYODIDE_BUNDLED, requirement_name

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_declared_dependencies_name_only_pyodide_bundled_packages() -> None:
    names = {requirement_name(r) for r in PYPROJECT["project"]["dependencies"]}
    assert names <= PYODIDE_BUNDLED, sorted(names - PYODIDE_BUNDLED)


def test_no_optional_dependencies_are_declared() -> None:
    assert "optional-dependencies" not in PYPROJECT["project"]


def test_marimo_is_pinned_exactly_in_the_dev_group() -> None:
    assert "marimo==0.25.1" in PYPROJECT["dependency-groups"]["dev"]
