from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.threshold import OperatingPoint
from predictive_maintenance.inference.artifact import DisabledUnit, LoadedUnit, load_unit
from predictive_maintenance.inference.examples import (
    find_example,
    load_examples,
    recorded_outcome,
)
from predictive_maintenance.inference.presentation import banned_wording_in, example_outcome_line
from predictive_maintenance.training.unit import (
    build_operating_metrics,
    reject_policy,
    write_unit,
)

MODEL = "random_forest_unweighted"
AIR, PROCESS, SPEED, TORQUE, WEAR = schema.NUMERIC_FEATURES


def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Type": ["L", "M", "H", "L"],
            AIR: [298.0, 300.0, 302.0, 299.0],
            PROCESS: [308.0, 310.0, 312.0, 309.0],
            SPEED: [1400, 1500, 1600, 1450],
            TORQUE: [30.0, 40.0, 50.0, 35.0],
            WEAR: [0, 100, 200, 50],
        }
    )


def write(models_dir: Path) -> LoadedUnit:
    train = frame()
    pipeline = Pipeline([("model", DummyClassifier())]).fit(train, [0, 1, 0, 1])
    thresholds = {10.0: 0.2}
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
        metrics=build_operating_metrics(
            MODEL, [0, 1, 0, 1], [[0.1, 0.6, 0.2, 0.9]], thresholds, train
        ),
        golden=train.head(1),
        golden_scores=[0.5],
        examples=train.head(3).with_columns(pl.Series(schema.TARGET, [0, 1, 0])),
    )
    unit = load_unit(models_dir, MODEL)
    assert isinstance(unit, LoadedUnit)
    return unit


@pytest.fixture
def unit(tmp_path: Path) -> LoadedUnit:
    return write(tmp_path)


def test_the_examples_come_from_the_manifest_with_their_recorded_outcome(unit: LoadedUnit) -> None:
    examples = load_examples(unit)
    assert [example.machine_failure for example in examples] == [False, True, False]
    assert examples[1].reading == {
        "Type": "M",
        AIR: 300.0,
        PROCESS: 310.0,
        SPEED: 1500,
        TORQUE: 40.0,
        WEAR: 100,
    }


def test_the_labels_are_distinct_and_do_not_reveal_the_outcome(unit: LoadedUnit) -> None:
    labels = [example.label for example in load_examples(unit)]
    assert len(set(labels)) == len(labels)
    assert all("failure" not in label.lower() for label in labels)


def test_a_disabled_unit_offers_no_examples() -> None:
    assert load_examples(DisabledUnit(MODEL, "missing file")) == ()


def test_a_manifest_without_examples_offers_none(unit: LoadedUnit) -> None:
    manifest = {key: value for key, value in unit.manifest.items() if key != "examples"}
    assert load_examples(replace(unit, manifest=manifest)) == ()


def test_a_malformed_example_is_left_out(unit: LoadedUnit) -> None:
    manifest = {**unit.manifest, "examples": [{"Type": "M"}, *unit.manifest["examples"]]}
    assert len(load_examples(replace(unit, manifest=manifest))) == 3


def test_the_outcome_is_recorded_only_for_the_unedited_example(unit: LoadedUnit) -> None:
    example = load_examples(unit)[1]
    assert recorded_outcome(example, dict(example.reading)) is True
    assert recorded_outcome(load_examples(unit)[0], dict(load_examples(unit)[0].reading)) is False


@pytest.mark.parametrize("name", list(frame().columns))
def test_editing_any_input_clears_the_outcome(unit: LoadedUnit, name: str) -> None:
    example = load_examples(unit)[1]
    edited = {**example.reading, name: "H" if name == "Type" else 999.0}
    assert recorded_outcome(example, edited) is None


def test_emptying_an_input_clears_the_outcome(unit: LoadedUnit) -> None:
    example = load_examples(unit)[1]
    assert recorded_outcome(example, {**example.reading, TORQUE: None}) is None


def test_no_example_selected_has_no_outcome() -> None:
    assert recorded_outcome(None, {"Type": "M"}) is None


def test_the_outcome_line_text_passes_the_banned_wording_test_and_names_only_machine_failure(
    unit: LoadedUnit,
) -> None:
    for example in load_examples(unit):
        line = example_outcome_line(example.machine_failure)
        assert banned_wording_in(line) == []
        assert not any(mode in line for mode in schema.FAILURE_MODES)
        assert line.startswith("Recorded in the data: ")


def test_an_example_is_found_by_its_label_and_nothing_chosen_finds_none(unit: LoadedUnit) -> None:
    examples = load_examples(unit)
    assert find_example(examples, "Example reading 2") is examples[1]
    assert find_example(examples, None) is None
    assert find_example(examples, "Example reading 99") is None
