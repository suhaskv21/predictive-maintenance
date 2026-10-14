"""The entry notebook renders each state from the inference layer and holds no logic of its own."""

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.threshold import OperatingPoint
from predictive_maintenance.training.unit import (
    build_operating_metrics,
    reject_policy,
    write_unit,
)

APP = Path(__file__).resolve().parents[1] / "marimo_app.py"
MODEL = "random_forest_unweighted"
AIR, PROCESS, SPEED, TORQUE, WEAR = schema.NUMERIC_FEATURES
READING: dict[str, Any] = {
    schema.CATEGORICAL_FEATURE: "M",
    AIR: 300.0,
    PROCESS: 310.0,
    SPEED: 1500,
    TORQUE: 40.0,
    WEAR: 100,
}
# Only the framework, the loader, the assessment, the Presentation and the config they need.
ALLOWED_IMPORTS = {
    "marimo",
    "predictive_maintenance.config",
    "predictive_maintenance.data",
    "predictive_maintenance.inference.artifact",
    "predictive_maintenance.inference.assessment",
    "predictive_maintenance.inference.golden",
    "predictive_maintenance.inference.presentation",
    "predictive_maintenance.inference.site",
}


def load_app() -> ModuleType:
    spec = importlib.util.spec_from_file_location("marimo_app_under_test", APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_constant_unit(models_dir: Path, *, threshold: float, centre: float = 100) -> None:
    """A unit whose Pipeline scores every Reading 0.3, so the Flag depends on ``threshold`` only."""
    rng = np.random.default_rng(1)
    n = 1000
    frame = pl.DataFrame(
        {
            schema.CATEGORICAL_FEATURE: rng.choice(list(schema.TYPE_LEVELS), n),
            **{name: rng.normal(centre, 10, n) for name in schema.NUMERIC_FEATURES},
        }
    )
    y = np.array([1] * 300 + [0] * 700)
    pipeline = Pipeline([("model", DummyClassifier(strategy="prior"))]).fit(frame, y)
    thresholds = {5.0: threshold * 1.5, 10.0: threshold}
    write_unit(
        models_dir,
        pipeline=pipeline,
        point=OperatingPoint(
            model=MODEL,
            default_cost_ratio=10.0,
            thresholds=thresholds,
            bands=None,
            risk_floor=0.01,
        ),
        policy=reject_policy(MODEL, 10.0),
        metrics=build_operating_metrics(MODEL, y, np.array([y * 0.5]), thresholds, frame),
        golden=frame.head(1),
        golden_scores=[0.3],
    )


def run(models_dir: Path, reading: dict[str, Any] | None = None) -> tuple[Any, Any]:
    defs: dict[str, Any] = {"unit_dir": models_dir}
    if reading is not None:
        defs["reading"] = reading
    outputs, definitions = load_app().app.run(defs=defs)
    return outputs[-1].text, definitions["presentation"]


@pytest.fixture
def flagging_dir(tmp_path: Path) -> Path:
    write_constant_unit(tmp_path, threshold=0.2)
    return tmp_path


def test_a_complete_reading_above_the_threshold_shows_inspect(flagging_dir: Path) -> None:
    html, presentation = run(flagging_dir, READING)
    assert presentation.state == "inspect"
    assert "Inspect" in html


def test_a_complete_reading_below_the_threshold_shows_no_inspection_flagged(
    tmp_path: Path,
) -> None:
    write_constant_unit(tmp_path, threshold=0.5)
    html, presentation = run(tmp_path, READING)
    assert presentation.state == "no_inspection_flagged"
    assert "No inspection flagged" in html


def test_the_form_defaults_are_a_complete_reading(flagging_dir: Path) -> None:
    _, presentation = run(flagging_dir)
    assert presentation.state in {"inspect", "no_inspection_flagged"}


@pytest.mark.parametrize("name", list(READING))
def test_a_reading_with_one_input_absent_shows_not_assessed(flagging_dir: Path, name: str) -> None:
    html, presentation = run(flagging_dir, {**READING, name: None})
    assert presentation.state == "not_assessed"
    assert "Not assessed" in html
    assert f"{name}: absent" in html
    assert presentation.failure_risk is None


def test_an_emptied_number_field_is_absent_not_a_number(flagging_dir: Path) -> None:
    _, presentation = run(flagging_dir, {**READING, TORQUE: ""})
    assert presentation.state == "not_assessed"
    assert f"{TORQUE}: absent" in presentation.refusal


def test_a_reading_with_an_out_of_range_value_shows_not_assessed(flagging_dir: Path) -> None:
    html, presentation = run(flagging_dir, {**READING, SPEED: -5})
    assert presentation.state == "not_assessed"
    assert f"{SPEED}: out of range" in html


def test_a_unit_that_fails_its_checks_shows_the_error_state(tmp_path: Path) -> None:
    html, presentation = run(tmp_path, READING)
    assert presentation.state == "model_unavailable"
    assert "Model unavailable" in html
    assert "Failure risk" not in html
    assert "Scoring check failed" not in html


def test_every_state_shows_the_caveats_and_the_build_commit(flagging_dir: Path) -> None:
    for reading in (READING, {**READING, TORQUE: None}):
        html, presentation = run(flagging_dir, reading)
        assert all(caveat in html for caveat in presentation.caveats)
        assert "Build commit" in html


def test_a_fresh_unit_shows_no_scoring_check_banner(flagging_dir: Path) -> None:
    html, _ = run(flagging_dir, READING)
    assert "Scoring check failed" not in html


def test_a_stored_score_that_differs_shows_the_scoring_check_banner(flagging_dir: Path) -> None:
    manifest = flagging_dir / f"{MODEL}.manifest.json"
    content = json.loads(manifest.read_text())
    content["golden"]["scores"][0] = 0.9
    manifest.write_text(json.dumps(content))
    html, presentation = run(flagging_dir, READING)
    assert "Scoring check failed" in html
    assert "Reading 1: stored score 0.9" in html
    assert presentation.state == "inspect"


RANGE_PHRASE = "outside the range of the training data, treat the result as unreliable"
SENSITIVITY_TITLE = "How the Flag changes with the cost assumption"


def test_a_plausible_reading_outside_the_training_data_shows_the_note_and_is_scored(
    flagging_dir: Path,
) -> None:
    html, presentation = run(flagging_dir, READING)
    assert presentation.state == "inspect"
    assert RANGE_PHRASE in html


def test_a_reading_inside_the_training_data_shows_no_note(tmp_path: Path) -> None:
    write_constant_unit(tmp_path, threshold=0.2, centre=300)
    html, presentation = run(
        tmp_path, {**READING, **{name: 300 for name in schema.NUMERIC_FEATURES}}
    )
    assert presentation.state == "inspect"
    assert RANGE_PHRASE not in html


def test_a_result_shows_the_sensitivity_expander_with_each_cost_ratio(flagging_dir: Path) -> None:
    html, presentation = run(flagging_dir, READING)
    assert SENSITIVITY_TITLE in html
    for row in presentation.sensitivity:
        assert row.text.split(":")[0] in html
    assert [row.ratio for row in presentation.sensitivity] == ["5:1", "10:1"]


def test_a_refused_reading_and_an_error_state_show_no_expander(
    flagging_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    refused, _ = run(flagging_dir, {**READING, TORQUE: None})
    unavailable, _ = run(tmp_path_factory.mktemp("empty"), READING)
    assert SENSITIVITY_TITLE not in refused
    assert SENSITIVITY_TITLE not in unavailable


def test_the_notebook_imports_only_the_inference_layer_and_its_config() -> None:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(APP.read_text())):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= ALLOWED_IMPORTS, sorted(imported - ALLOWED_IMPORTS)
