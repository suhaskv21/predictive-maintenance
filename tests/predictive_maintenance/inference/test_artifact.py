import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from predictive_maintenance.evaluation.threshold import OperatingPoint
from predictive_maintenance.inference.artifact import DisabledUnit, LoadedUnit, load_unit
from predictive_maintenance.training.unit import (
    ArtifactUnit,
    build_operating_metrics,
    file_sha256,
    reject_policy,
    unit_paths,
    write_unit,
)

MODEL = "toy"
THRESHOLD = 0.2


def readings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Type": ["L", "M", "H", "L"],
            "Air temperature [K]": [298.0, 300.0, 302.0, 299.0],
            "Process temperature [K]": [308.0, 310.0, 312.0, 309.0],
            "Rotational speed [rpm]": [1400, 1500, 1600, 1450],
            "Torque [Nm]": [30.0, 40.0, 50.0, 35.0],
            "Tool wear [min]": [0, 100, 200, 50],
        }
    )


def write(models_dir: Path, model: str = MODEL) -> ArtifactUnit:
    frame = readings()
    pipeline = Pipeline([("model", DummyClassifier(strategy="prior"))]).fit(frame, [0, 1, 0, 1])
    thresholds = {5.0: 0.3, 10.0: THRESHOLD}
    return write_unit(
        models_dir,
        pipeline=pipeline,
        point=OperatingPoint(
            model=model,
            default_cost_ratio=10.0,
            thresholds=thresholds,
            bands=None,
            risk_floor=0.01,
        ),
        policy=reject_policy(model, 10.0),
        metrics=build_operating_metrics(
            model, [0, 0, 1, 1], np.array([[0.1, 0.2, 0.6, 0.9]]), thresholds, frame
        ),
        golden=frame.head(2),
        golden_scores=[0.5, 0.5],
    )


def edit_manifest(unit: ArtifactUnit, change: Callable[[dict[str, Any]], None]) -> None:
    manifest = json.loads(unit.manifest.read_text())
    change(manifest)
    unit.manifest.write_text(json.dumps(manifest))


def rewrite_sidecar(
    unit: ArtifactUnit, path: Path, change: Callable[[dict[str, Any]], None]
) -> None:
    """Change a JSON sidecar and keep its manifest hash right, to reach the content checks."""
    content = json.loads(path.read_text())
    change(content)
    path.write_text(json.dumps(content))
    edit_manifest(unit, lambda m: m["files"].__setitem__(path.name, file_sha256(path)))


def disabled(models_dir: Path, model: str = MODEL) -> DisabledUnit:
    result = load_unit(models_dir, model)
    assert isinstance(result, DisabledUnit)
    return result


def test_a_valid_unit_loads_with_its_pipeline_threshold_policy_and_figures(tmp_path: Path) -> None:
    write(tmp_path)
    unit = load_unit(tmp_path, MODEL)
    assert isinstance(unit, LoadedUnit)
    assert unit.model == MODEL
    assert unit.operating_point.threshold == THRESHOLD
    assert unit.policy.strategy == "reject"
    assert unit.operating_metrics["source"] == "cv_train"
    assert unit.pipeline.predict_proba(readings())[:, 1].tolist() == [0.5] * 4


def test_a_disabled_unit_names_its_model_and_gives_a_one_line_reason(tmp_path: Path) -> None:
    result = disabled(tmp_path)
    assert result.model == MODEL
    assert result.reason and "\n" not in result.reason


@pytest.mark.parametrize(
    "part", ["pipeline", "operating_point", "missing", "operating_metrics", "manifest"]
)
def test_a_missing_file_disables_the_unit_and_names_the_file(tmp_path: Path, part: str) -> None:
    unit = write(tmp_path)
    path = getattr(unit, part)
    path.unlink()
    assert "missing file" in disabled(tmp_path).reason
    assert path.name in disabled(tmp_path).reason


def test_a_manifest_for_another_model_disables_the_unit(tmp_path: Path) -> None:
    edit_manifest(write(tmp_path), lambda m: m.update(model="other"))
    assert "other" in disabled(tmp_path).reason


def test_a_manifest_that_names_different_files_disables_the_unit(tmp_path: Path) -> None:
    def rename(manifest: dict[str, Any]) -> None:
        files = manifest["files"]
        files["other.missing.json"] = files.pop(f"{MODEL}.missing.json")

    edit_manifest(write(tmp_path), rename)
    assert "file names" in disabled(tmp_path).reason


@pytest.mark.parametrize("part", ["operating_point", "missing", "operating_metrics"])
def test_a_sidecar_that_names_another_model_disables_the_unit(tmp_path: Path, part: str) -> None:
    unit = write(tmp_path)
    rewrite_sidecar(unit, getattr(unit, part), lambda content: content.update(model="other"))
    assert "other" in disabled(tmp_path).reason


@pytest.mark.parametrize("part", ["pipeline", "operating_point", "missing", "operating_metrics"])
def test_a_changed_file_fails_its_hash_check_and_disables_the_unit(
    tmp_path: Path, part: str
) -> None:
    path = getattr(write(tmp_path), part)
    path.write_bytes(path.read_bytes() + b" ")
    reason = disabled(tmp_path).reason
    assert "SHA-256" in reason and path.name in reason


def test_a_library_version_that_differs_from_the_manifest_disables_the_unit(
    tmp_path: Path,
) -> None:
    edit_manifest(write(tmp_path), lambda m: m["libraries"].update({"scikit-learn": "0.0.1"}))
    reason = disabled(tmp_path).reason
    assert "scikit-learn" in reason and "0.0.1" in reason


def test_an_optional_library_that_is_recorded_but_not_installed_does_not_disable_the_unit(
    tmp_path: Path,
) -> None:
    edit_manifest(write(tmp_path), lambda m: m["libraries"].update({"no-such-library": "1.0"}))
    assert isinstance(load_unit(tmp_path, MODEL), LoadedUnit)
    unit = unit_paths(tmp_path, MODEL)
    edit_manifest(unit, lambda m: m["libraries"].pop("xgboost"))
    assert isinstance(load_unit(tmp_path, MODEL), LoadedUnit)


def test_a_manifest_that_records_no_version_of_a_required_library_disables_the_unit(
    tmp_path: Path,
) -> None:
    edit_manifest(write(tmp_path), lambda m: m["libraries"].pop("polars"))
    assert "polars" in disabled(tmp_path).reason


def test_a_manifest_that_is_not_json_disables_the_unit(tmp_path: Path) -> None:
    write(tmp_path)
    unit_paths(tmp_path, MODEL).manifest.write_text("{not json")
    assert "manifest" in disabled(tmp_path).reason


def test_a_pipeline_that_cannot_be_loaded_disables_the_unit_with_one_line(tmp_path: Path) -> None:
    unit = write(tmp_path)
    unit.pipeline.write_bytes(b"not a pickle")
    edit_manifest(
        unit, lambda m: m["files"].update({unit.pipeline.name: file_sha256(unit.pipeline)})
    )
    result = disabled(tmp_path)
    assert unit.pipeline.name in result.reason and "\n" not in result.reason


def test_a_policy_that_is_not_reject_disables_the_unit(tmp_path: Path) -> None:
    unit = write(tmp_path)
    rewrite_sidecar(unit, unit.missing, lambda content: content.update(strategy="median"))
    assert "reject" in disabled(tmp_path).reason


def test_a_failed_model_is_never_replaced_by_another_valid_model(tmp_path: Path) -> None:
    write(tmp_path, model="other")
    broken = write(tmp_path)
    broken.operating_point.unlink()
    assert isinstance(load_unit(tmp_path, "other"), LoadedUnit)
    assert disabled(tmp_path).model == MODEL
