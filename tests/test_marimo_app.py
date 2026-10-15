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
from predictive_maintenance.inference.artifact import LoadedUnit, load_unit
from predictive_maintenance.inference.examples import load_examples
from predictive_maintenance.inference.presentation import banned_wording_in
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
    "predictive_maintenance.inference.examples",
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


def write_constant_unit(
    models_dir: Path,
    *,
    threshold: float,
    centre: float = 100,
    examples: pl.DataFrame | None = None,
) -> None:
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
        examples=examples,
    )


def run(
    models_dir: Path, reading: dict[str, Any] | None = None, example: Any = None
) -> tuple[Any, Any]:
    return run_app(models_dir, reading, example)[:2]


def run_app(
    models_dir: Path,
    reading: dict[str, Any] | None = None,
    example: Any = None,
    extra: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any]:
    defs: dict[str, Any] = {"unit_dir": models_dir, **(extra or {})}
    if reading is not None:
        defs["reading"] = reading
    if example is not None:
        defs["example"] = example
    outputs, definitions = load_app().app.run(defs=defs)
    return outputs[-1].text, definitions["presentation"], definitions


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


def test_the_model_name_and_caveats_sit_in_a_model_card_expander(flagging_dir: Path) -> None:
    for reading in (READING, {**READING, TORQUE: None}):
        html, presentation = run(flagging_dir, reading)
        assert "Model card: Random forest (unweighted)" in html
        assert presentation.model_card_title in html


def test_every_state_shows_the_caveats_and_the_release(flagging_dir: Path) -> None:
    for reading in (READING, {**READING, TORQUE: None}):
        html, presentation = run(flagging_dir, reading)
        assert all(caveat in html for caveat in presentation.caveats)
        assert "Release:" in html


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


def test_the_cost_ratio_dropdown_changes_the_flag(tmp_path: Path) -> None:
    import marimo as mo

    write_constant_unit(tmp_path, threshold=0.25)  # score 0.3: flagged at 10:1, not at 5:1 (0.375)
    html, default = run(tmp_path, READING)
    assert default.state == "inspect"  # the dropdown starts at the configured 10:1
    assert "Cost ratio" in html
    picker = mo.ui.dropdown({"5:1": 5.0, "10:1": 10.0}, value="5:1")
    _, relaxed, _ = run_app(tmp_path, READING, extra={"cost_ratio_picker": picker})
    assert relaxed.state == "no_inspection_flagged"
    assert any("as much as 5 false alarms" in caveat for caveat in relaxed.caveats)


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


def test_what_a_flag_means_is_inside_an_expander(flagging_dir: Path) -> None:
    html, presentation = run(flagging_dir, READING)
    assert presentation.meaning_title is not None and presentation.meaning_title in html
    assert all(line in html for line in presentation.meaning)
    refused, _ = run(flagging_dir, {**READING, TORQUE: None})
    assert "What this Flag means" not in refused


EXAMPLES = pl.DataFrame(
    {
        schema.CATEGORICAL_FEATURE: ["L", "H"],
        AIR: [298.5, 301.0],
        PROCESS: [308.5, 311.0],
        SPEED: [1400, 2100],
        TORQUE: [30.5, 65.0],
        WEAR: [10, 220],
        schema.TARGET: [0, 1],
    }
)


@pytest.fixture
def examples_dir(tmp_path: Path) -> Path:
    write_constant_unit(tmp_path, threshold=0.2, examples=EXAMPLES)
    return tmp_path


def load_app_examples(models_dir: Path) -> tuple[Any, ...]:
    unit = load_unit(models_dir, MODEL)
    assert isinstance(unit, LoadedUnit)
    return load_examples(unit)


def example_reading(row: int) -> dict[str, Any]:
    return {name: EXAMPLES[name][row] for name in schema.FEATURE_COLUMNS}


def test_the_example_selector_offers_each_example_without_its_outcome(
    examples_dir: Path,
) -> None:
    html, _, definitions = run_app(examples_dir, READING)
    assert "Example reading" in html
    labels = [example.label for example in definitions["examples"]]
    assert labels == ["Example reading 1", "Example reading 2"]
    assert all(label in html for label in labels)
    assert "Recorded in the data" not in html


def test_a_unit_without_examples_shows_no_selector(flagging_dir: Path) -> None:
    html, _ = run(flagging_dir, READING)
    assert "Example reading" not in html


@pytest.mark.parametrize(("row", "outcome"), [(0, "no failure"), (1, "Machine failure")])
def test_choosing_an_example_fills_the_form_and_shows_its_recorded_outcome(
    examples_dir: Path, row: int, outcome: str
) -> None:
    chosen = load_app_examples(examples_dir)[row]
    html, presentation, definitions = run_app(examples_dir, example=chosen)
    assert definitions["inputs"].value == example_reading(row)
    assert f"Recorded in the data: {outcome}." in html
    assert presentation.outcome_line in html
    assert "so agreement proves nothing" in html
    assert not any(mode in presentation.outcome_line for mode in schema.FAILURE_MODES)


def test_the_outcome_line_text_passes_the_banned_wording_test(examples_dir: Path) -> None:
    for chosen in load_app_examples(examples_dir):
        _, presentation = run(examples_dir, example=chosen)
        assert banned_wording_in(presentation.outcome_line) == []


@pytest.mark.parametrize("name", list(READING))
def test_editing_any_input_clears_the_outcome_line(examples_dir: Path, name: str) -> None:
    chosen = load_app_examples(examples_dir)[1]
    edited = {**example_reading(1), name: "M" if name == "Type" else 123.0}
    html, presentation = run(examples_dir, edited, example=chosen)
    assert presentation.outcome_line is None
    assert "Recorded in the data" not in html


def test_emptying_an_input_clears_the_outcome_line(examples_dir: Path) -> None:
    chosen = load_app_examples(examples_dir)[1]
    html, _ = run(examples_dir, {**example_reading(1), TORQUE: None}, example=chosen)
    assert "Recorded in the data" not in html


def test_without_a_chosen_example_no_outcome_is_shown(examples_dir: Path) -> None:
    html, presentation = run(examples_dir, example_reading(1))
    assert presentation.outcome_line is None
    assert "Recorded in the data" not in html
