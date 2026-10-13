"""Load and check one model's artifact unit (ADR 0009, ADR 0012); imports no UI framework.

``load_unit`` returns a ``LoadedUnit`` only if all five files are present, name the requested
model, match the SHA-256 hashes in the manifest and were written under the installed library
versions. Any failure returns a ``DisabledUnit`` with a one-line reason instead. A disabled unit
is never scored with a fallback threshold, and a failed model is never replaced by another one:
the caller shows the error state.
"""

import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from predictive_maintenance.evaluation.missingness import MissingDataPolicy
from predictive_maintenance.evaluation.threshold import OperatingPoint
from predictive_maintenance.training.unit import LIBRARIES, REJECT, file_sha256, unit_paths

# A model needs these to run at all; the other recorded libraries (gradient boosting packages) are
# compared only where installed, because the forest does not import them.
REQUIRED_LIBRARIES = ("scikit-learn", "polars", "joblib", "numpy")


@dataclass(frozen=True)
class LoadedUnit:
    """A checked unit: the Pipeline and the figures that belong to it."""

    model: str
    pipeline: Pipeline
    operating_point: OperatingPoint
    policy: MissingDataPolicy
    operating_metrics: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class DisabledUnit:
    """A unit that failed a check: it must not be used to score, and ``reason`` is one line."""

    model: str
    reason: str


class _CheckError(Exception):
    """A failed check; the message is the one-line reason."""


def _installed(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _check_versions(recorded: dict[str, Any]) -> None:
    for name in LIBRARIES:
        wanted, installed = recorded.get(name), _installed(name)
        required = name in REQUIRED_LIBRARIES
        if wanted is None and required:
            raise _CheckError(f"the manifest records no version of {name}")
        if installed is None and required:
            raise _CheckError(f"{name} {wanted} is needed but not installed")
        if wanted is not None and installed is not None and wanted != installed:
            raise _CheckError(
                f"{name} {installed} is installed but the unit was built with {wanted}"
            )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise _CheckError(f"{path.name} cannot be read: {error}") from error


def _load(models_dir: Path, model: str) -> LoadedUnit:
    unit = unit_paths(models_dir, model)
    for path in (*unit.sidecars, unit.manifest):
        if not path.is_file():
            raise _CheckError(f"missing file {path.name}")

    manifest = _read_json(unit.manifest)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    libraries = manifest.get("libraries") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not isinstance(libraries, dict):
        raise _CheckError(f"{unit.manifest.name} is not a manifest")
    if manifest.get("model") != model:
        raise _CheckError(
            f"{unit.manifest.name} is for model {manifest.get('model')!r}, not {model!r}"
        )
    if set(files) != {path.name for path in unit.sidecars}:
        raise _CheckError(f"{unit.manifest.name} lists unexpected file names")
    for path in unit.sidecars:
        if file_sha256(path) != files[path.name]:
            raise _CheckError(f"SHA-256 of {path.name} does not match the manifest")
    _check_versions(libraries)

    try:
        point = OperatingPoint.from_json(unit.operating_point.read_text())
        policy = MissingDataPolicy.from_json(unit.missing.read_text())
    except (ValueError, KeyError, TypeError) as error:
        raise _CheckError(f"the operating point or policy cannot be read: {error!r}") from error
    metrics = _read_json(unit.operating_metrics)
    named = {
        unit.operating_point.name: point.model,
        unit.missing.name: policy.model,
        unit.operating_metrics.name: metrics.get("model") if isinstance(metrics, dict) else None,
    }
    for name, other in named.items():
        if other != model:
            raise _CheckError(f"{name} is for model {other!r}, not {model!r}")
    if policy.strategy != REJECT:
        raise _CheckError(
            f"{unit.missing.name} strategy is {policy.strategy!r}; only {REJECT!r} is supported"
        )

    try:
        pipeline = joblib.load(unit.pipeline)
    except Exception as error:
        raise _CheckError(f"{unit.pipeline.name} cannot be loaded: {error!r}") from error
    return LoadedUnit(model, pipeline, point, policy, metrics, manifest)


def load_unit(models_dir: Path, model: str) -> LoadedUnit | DisabledUnit:
    """Load and check the unit of ``model`` under ``models_dir``; never falls back to another."""
    try:
        return _load(models_dir, model)
    except _CheckError as failure:
        return DisabledUnit(model, " ".join(str(failure).split()))
