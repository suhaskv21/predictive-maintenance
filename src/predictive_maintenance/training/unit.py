"""The five-file artifact unit of one model: its layout, contents and writer (ADR 0009, ADR 0012).

The unit lives under ``models/`` (git-ignored) as ``<model>.<part>``:

- ``pipeline.joblib``: preprocessing and model as one scikit-learn ``Pipeline``;
- ``operating-point.json``: ``OperatingPoint`` (default ratio, thresholds by ratio, no bands);
- ``missing.json``: the missing-data policy, strategy ``reject`` with an empty reference;
- ``operating-metrics.json``: CV figures per Cost ratio labelled ``cv_train``, and training ranges;
- ``manifest.json``: model name, library versions, SHA-256 of the other four files and the scores of
  a few fixed train Readings (the golden check).

This module knows the layout and the file formats, not how a model is trained, so the training
command writes the unit and the inference layer can find and read it through the same names.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.missingness import MissingDataPolicy
from predictive_maintenance.evaluation.threshold import OperatingPoint, cv_operating_metrics
from predictive_maintenance.preprocessing.imputation import STRATEGIES

REJECT = STRATEGIES[0]
SOURCE_LABEL = "cv_train"
NOT_MEASURED = "not measured in this build"
LIBRARIES = ("scikit-learn", "xgboost", "lightgbm", "polars", "joblib", "numpy")


@dataclass(frozen=True)
class ArtifactUnit:
    """The five files of one model's unit."""

    model: str
    pipeline: Path
    operating_point: Path
    missing: Path
    operating_metrics: Path
    manifest: Path

    @property
    def sidecars(self) -> tuple[Path, ...]:
        """The four files whose SHA-256 the manifest records."""
        return (self.pipeline, self.operating_point, self.missing, self.operating_metrics)


def unit_paths(models_dir: Path, model: str) -> ArtifactUnit:
    """Where the unit of ``model`` lives under ``models_dir``."""
    return ArtifactUnit(
        model=model,
        pipeline=models_dir / f"{model}.pipeline.joblib",
        operating_point=models_dir / f"{model}.operating-point.json",
        missing=models_dir / f"{model}.missing.json",
        operating_metrics=models_dir / f"{model}.operating-metrics.json",
        manifest=models_dir / f"{model}.manifest.json",
    )


def ratio_key(cost_ratio: float) -> str:
    """The JSON key of a Cost ratio, as ``OperatingPoint.to_json`` writes it (``"10.0"``)."""
    return repr(cost_ratio)


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_policy(model: str, cost_ratio: float) -> MissingDataPolicy:
    """The policy of this build: every Incomplete reading is refused and nothing is measured."""
    return MissingDataPolicy(model=model, strategy=REJECT, cost_ratio=cost_ratio, scorable=())


def _none_if_nan(value: float) -> float | None:
    return None if math.isnan(value) else value


def build_operating_metrics(
    model: str,
    y: Sequence[int] | NDArray[Any],
    repeat_scores: NDArray[np.float64],
    thresholds: Mapping[float, float],
    train: pl.DataFrame,
) -> dict[str, Any]:
    """CV figures per Cost ratio (labelled ``cv_train``) and the training range of each input.

    ``repeat_scores`` are the out-of-fold scores of the train partition's CV repeats and
    ``thresholds`` the Operating threshold per Cost ratio. Figures that are undefined (nothing
    flagged, no failure caught) are ``None``, never NaN, so the file is valid JSON.
    """
    ratios = {}
    for ratio, threshold in thresholds.items():
        figures = cv_operating_metrics(y, repeat_scores, threshold, ratio)
        ratios[ratio_key(ratio)] = {
            "recall": _none_if_nan(figures.recall),
            "precision": _none_if_nan(figures.precision),
            "flags_per_1000": _none_if_nan(figures.alerts_per_1000),
            "false_alarms_per_caught": _none_if_nan(figures.false_alarms_per_caught),
        }
    return {
        "model": model,
        "source": SOURCE_LABEL,
        "ratios": ratios,
        "training_ranges": {
            name: {"min": train[name].min(), "max": train[name].max()}
            for name in schema.NUMERIC_FEATURES
        },
        "training_types": list(schema.TYPE_LEVELS),
    }


def _dump(path: Path, content: Mapping[str, Any] | str) -> None:
    text = content if isinstance(content, str) else json.dumps(content, indent=2)
    path.write_text(text + "\n")


def write_unit(
    models_dir: Path,
    *,
    pipeline: Pipeline,
    point: OperatingPoint,
    policy: MissingDataPolicy,
    metrics: Mapping[str, Any],
    golden: pl.DataFrame,
    golden_scores: Sequence[float],
) -> ArtifactUnit:
    """Write the five files of ``point.model`` under ``models_dir`` and return their paths.

    ``golden`` are fixed train Readings (the six inputs) and ``golden_scores`` the Pipeline's
    scores on them, stored in the manifest for the golden check.

    Raises:
        ValueError: if the policy or metrics name a different model than the operating point, or
            ``golden`` and ``golden_scores`` differ in length.
    """
    model = point.model
    if policy.model != model or metrics["model"] != model:
        raise ValueError("operating point, policy and metrics must name the same model")
    if golden.height != len(golden_scores):
        raise ValueError("golden Readings and golden scores must have the same length")
    models_dir.mkdir(parents=True, exist_ok=True)
    unit = unit_paths(models_dir, model)
    joblib.dump(pipeline, unit.pipeline)
    _dump(unit.operating_point, point.to_json())
    _dump(unit.missing, {**json.loads(policy.to_json()), "reference_note": NOT_MEASURED})
    _dump(unit.operating_metrics, metrics)
    _dump(
        unit.manifest,
        {
            "model": model,
            "libraries": {name: version(name) for name in LIBRARIES},
            "files": {path.name: file_sha256(path) for path in unit.sidecars},
            "golden": {"readings": golden.to_dicts(), "scores": list(golden_scores)},
        },
    )
    return unit
