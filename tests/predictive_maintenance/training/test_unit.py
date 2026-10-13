import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.threshold import OperatingPoint
from predictive_maintenance.training.unit import (
    NOT_MEASURED,
    build_operating_metrics,
    file_sha256,
    ratio_key,
    reject_policy,
    unit_paths,
    write_unit,
)

MODEL = "toy"


def train_frame() -> pl.DataFrame:
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


def point(model: str = MODEL) -> OperatingPoint:
    return OperatingPoint(
        model=model,
        default_cost_ratio=10.0,
        thresholds={5.0: 0.3, 10.0: 0.2},
        bands=None,
        risk_floor=0.01,
    )


def metrics(model: str = MODEL) -> dict[str, object]:
    return build_operating_metrics(
        model,
        np.array([0, 0, 1, 1]),
        np.array([[0.1, 0.2, 0.6, 0.9]]),
        {5.0: 0.3, 10.0: 0.2},
        train_frame(),
    )


def write(tmp_path: Path, *, model: str = MODEL, policy_model: str = MODEL) -> Path:
    frame = train_frame()
    pipeline = Pipeline([("model", DummyClassifier())]).fit(frame, [0, 1, 0, 1])
    write_unit(
        tmp_path,
        pipeline=pipeline,
        point=point(model),
        policy=reject_policy(policy_model, 10.0),
        metrics=metrics(model),
        golden=frame.head(2),
        golden_scores=[0.5, 0.5],
    )
    return tmp_path


def test_unit_paths_name_the_five_files_after_the_model(tmp_path: Path) -> None:
    unit = unit_paths(tmp_path, MODEL)
    names = [p.name for p in (*unit.sidecars, unit.manifest)]
    assert names == [
        "toy.pipeline.joblib",
        "toy.operating-point.json",
        "toy.missing.json",
        "toy.operating-metrics.json",
        "toy.manifest.json",
    ]
    assert unit.model == MODEL


def test_ratio_key_matches_how_the_operating_point_writes_its_ratios() -> None:
    assert ratio_key(10.0) == "10.0"
    assert set(json.loads(point().to_json())["thresholds"]) == {ratio_key(5.0), ratio_key(10.0)}


def test_the_reject_policy_scores_nothing_and_has_an_empty_reference() -> None:
    policy = reject_policy(MODEL, 10.0)
    assert (policy.strategy, policy.scorable, policy.measured) == ("reject", (), ())


def test_write_unit_creates_the_directory_and_records_the_hash_of_each_sidecar(
    tmp_path: Path,
) -> None:
    unit = unit_paths(write(tmp_path / "nested" / "models"), MODEL)
    manifest = json.loads(unit.manifest.read_text())
    assert manifest["files"] == {p.name: file_sha256(p) for p in unit.sidecars}
    assert manifest["golden"]["scores"] == [0.5, 0.5]
    assert len(manifest["golden"]["readings"]) == 2


def test_write_unit_marks_the_policy_reference_as_not_measured(tmp_path: Path) -> None:
    unit = unit_paths(write(tmp_path), MODEL)
    assert json.loads(unit.missing.read_text())["reference_note"] == NOT_MEASURED


def test_a_changed_file_no_longer_matches_its_manifest_hash(tmp_path: Path) -> None:
    unit = unit_paths(write(tmp_path), MODEL)
    recorded = json.loads(unit.manifest.read_text())["files"][unit.operating_point.name]
    unit.operating_point.write_text(unit.operating_point.read_text() + " ")
    assert file_sha256(unit.operating_point) != recorded


def test_write_unit_rejects_files_that_name_different_models(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same model"):
        write(tmp_path, policy_model="other")
    assert not list(tmp_path.iterdir())


def test_write_unit_rejects_golden_readings_and_scores_of_different_length(tmp_path: Path) -> None:
    pipeline = Pipeline([("model", DummyClassifier())]).fit(train_frame(), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="same length"):
        write_unit(
            tmp_path,
            pipeline=pipeline,
            point=point(),
            policy=reject_policy(MODEL, 10.0),
            metrics=metrics(),
            golden=train_frame().head(2),
            golden_scores=[0.5],
        )


def test_operating_metrics_are_labelled_cv_train_with_figures_per_ratio_and_training_ranges() -> (
    None
):
    result = metrics()
    assert result["source"] == "cv_train"
    ratios = result["ratios"]
    assert isinstance(ratios, dict) and set(ratios) == {"5.0", "10.0"}
    # At 0.3: flags the two highest of four, both failures.
    assert ratios["5.0"] == {
        "recall": 1.0,
        "precision": 1.0,
        "flags_per_1000": 500.0,
        "false_alarms_per_caught": 0.0,
    }
    ranges = result["training_ranges"]
    assert isinstance(ranges, dict) and set(ranges) == set(schema.NUMERIC_FEATURES)
    assert ranges["Tool wear [min]"] == {"min": 0, "max": 200}


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_undefined_operating_figures_are_null_not_nan() -> None:
    result = build_operating_metrics(
        MODEL, np.array([0, 0, 1, 1]), np.array([[0.1, 0.2, 0.3, 0.4]]), {5.0: 0.99}, train_frame()
    )
    ratios = result["ratios"]
    assert isinstance(ratios, dict)
    assert ratios["5.0"]["precision"] is None and ratios["5.0"]["false_alarms_per_caught"] is None
    assert not math.isnan(ratios["5.0"]["recall"])
    json.dumps(result, allow_nan=False)
