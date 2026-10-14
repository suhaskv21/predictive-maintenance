"""The build contract: cheap checks that fail a build before a bad site deploys (ADR 0010, 0012).

``python -m predictive_maintenance.deploy.contract`` runs every check below against the built
unit, wheel and exported site and exits non-zero, printing one line per failure, if any fails.
The re-derivation of the Operating threshold and the proof that training never reads the Holdout
are done by training and its tests; here the written unit is checked against the accepted figure.
"""

import argparse
import re
import zipfile
from collections.abc import Sequence
from pathlib import Path

from predictive_maintenance.config import PROJECT_ROOT, Config, load_config
from predictive_maintenance.inference.artifact import DisabledUnit, load_unit
from predictive_maintenance.training.provenance import is_commit_sha
from predictive_maintenance.training.train import (
    DEFAULT_MODEL,
    ThresholdDriftError,
    check_threshold,
)

# Packages marimo's Pyodide bundles (ADR 0009 table); micropip installs any other declared
# dependency from PyPI, which can fail in the browser.
PYODIDE_BUNDLED = frozenset(
    {"joblib", "lightgbm", "numpy", "polars", "scikit-learn", "scipy", "xgboost"}
)

_BYTES_PER_MB = 1024 * 1024
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def requirement_name(requirement: str) -> str:
    """The normalised distribution name a requirement string starts with."""
    match = _NAME.match(requirement)
    return match.group(0).lower().replace("_", "-") if match else requirement


def check_unit(models_dir: Path, model: str, config: Config) -> list[str]:
    """Failures of the unit of ``model``: files, manifest, commit, versions and the threshold."""
    unit = load_unit(models_dir, model)
    if isinstance(unit, DisabledUnit):
        return [f"unit {model}: {unit.reason}"]
    failures = []
    if not is_commit_sha(unit.manifest.get("git")):
        failures.append(
            f"unit {model}: the manifest records no Git commit (trained with --allow-dirty or "
            "before ADR 0015); retrain from a clean commit"
        )
    try:
        check_threshold(config, model, unit.operating_point.threshold)
    except ThresholdDriftError as error:
        failures.append(f"unit {model}: Operating threshold is not the accepted one: {error}")
    return failures


def check_site_size(site: Path, budget_mb: float) -> list[str]:
    """Failures if ``site`` is missing or holds more than ``budget_mb`` megabytes."""
    if not site.is_dir():
        return [f"site {site} was not exported"]
    size = sum(path.stat().st_size for path in site.rglob("*") if path.is_file())
    if size > budget_mb * _BYTES_PER_MB:
        return [f"site is {size / _BYTES_PER_MB:.1f} MB, over the {budget_mb:g} MB budget"]
    return []


def check_wheel_dependencies(wheel: Path) -> list[str]:
    """Failures for each dependency the wheel installs by default that Pyodide does not bundle."""
    with zipfile.ZipFile(wheel) as archive:
        metadata = next(
            (name for name in archive.namelist() if name.endswith(".dist-info/METADATA")), None
        )
        if metadata is None:
            return [f"wheel {wheel.name} has no METADATA"]
        text = archive.read(metadata).decode()
    failures = []
    for line in text.splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        requirement = line.removeprefix("Requires-Dist:").strip()
        if "extra ==" in requirement:
            continue
        name = requirement_name(requirement)
        if name not in PYODIDE_BUNDLED:
            failures.append(f"wheel depends on {name}, which Pyodide does not bundle")
    return failures


def run_contract(
    *, models_dir: Path, model: str, wheel: Path, site: Path, config: Config
) -> list[str]:
    """Every failure of every check, so one build run shows them all."""
    return [
        *check_unit(models_dir, model, config),
        *check_wheel_dependencies(wheel),
        *check_site_size(site, config.deploy.site_budget_mb),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the contract; return the process exit code (1 if any check fails)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args(argv)
    failures = run_contract(
        models_dir=args.models,
        model=args.model,
        wheel=args.wheel,
        site=args.site,
        config=load_config(),
    )
    for failure in failures:
        print(f"CONTRACT FAILED: {failure}")
    if not failures:
        print("build contract passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
