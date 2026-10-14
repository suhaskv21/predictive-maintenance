import re
from typing import Any

import pytest

from predictive_maintenance.evaluation.threshold import OperatingPoint
from predictive_maintenance.inference.artifact import DisabledUnit
from predictive_maintenance.inference.assessment import (
    ABSENT,
    OUT_OF_RANGE,
    Assessment,
    MissingInput,
)
from predictive_maintenance.inference.golden import GoldenResult
from predictive_maintenance.inference.presentation import (
    BANNED_WORDING,
    Presentation,
    banned_wording_in,
    example_outcome_line,
    flag_line_text,
    golden_banner,
    present,
    present_unavailable,
    risk_text,
)

FLOOR = 0.01


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "under 1 %"),
        (0.0099999, "under 1 %"),
        (0.01, "about 1 %"),
        (0.2183, "about 22 %"),
        (0.125, "about 13 %"),  # halves go up
        (0.285, "about 29 %"),  # 0.285 * 100 is 28.499999999999996 in binary floating point
        (0.95, "about 95 %"),
        (0.9500001, "over 95 %"),
        (1.0, "over 95 %"),
    ],
)
def test_risk_text_rules(score: float, expected: str) -> None:
    assert risk_text(score, FLOOR) == expected


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_risk_text_rejects_non_probabilities(score: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        risk_text(score, FLOOR)


def test_a_floor_that_rounds_to_zero_never_shows_zero() -> None:
    assert risk_text(0.001, 0.001) == "under 1 %"


def test_flag_line_is_the_models_own_line_rounded_the_same_way() -> None:
    assert flag_line_text(0.21833333, FLOOR) == "This model flags from about 22 %."


# --- states -----------------------------------------------------------------------------------

MODEL = "random_forest_unweighted"
COST_RATIO = 10.0
POINT = OperatingPoint(
    model=MODEL,
    default_cost_ratio=COST_RATIO,
    thresholds={5.0: 0.24, 10.0: 0.2183, 20.0: 0.15},
    bands=None,
    risk_floor=FLOOR,
)
FIGURES = {
    "recall": 0.8557,
    "precision": 0.7978,
    "flags_per_1000": 36.31,
    "false_alarms_per_caught": 0.25,
}
METRICS: dict[str, Any] = {"model": MODEL, "source": "cv_train", "ratios": {"10.0": FIGURES}}


def assessment(score: float, flag: bool) -> Assessment:
    return Assessment("complete", (), None, False, score, flag)


REFUSED = Assessment(
    "refused",
    (MissingInput("Torque [Nm]", ABSENT), MissingInput("Rotational speed [rpm]", OUT_OF_RANGE)),
    "inputs absent or unusable: Torque [Nm], Rotational speed [rpm]",
    False,
    None,
    None,
)


def show(assessed: Assessment, *, recorded_outcome: bool | None = None) -> Presentation:
    return present(
        assessed, POINT, METRICS, cost_ratio=COST_RATIO, recorded_outcome=recorded_outcome
    )


def all_states() -> dict[str, Presentation]:
    return {
        "inspect": show(assessment(0.4, True)),
        "no_inspection_flagged": show(assessment(0.03, False)),
        "not_assessed": show(REFUSED),
        "model_unavailable": present_unavailable(
            DisabledUnit(
                MODEL, "SHA-256 of random_forest_unweighted.pipeline.joblib does not match"
            ),
            cost_ratio=COST_RATIO,
        ),
        "example_reading": show(assessment(0.4, True), recorded_outcome=True),
    }


def test_flagged_reading_headline_is_inspect_and_risk_is_secondary() -> None:
    shown = show(assessment(0.4, True))
    assert shown.state == "inspect"
    assert shown.headline == "Inspect"
    assert shown.failure_risk == "Failure risk (rough estimate, not calibrated): about 40 %"
    assert shown.flag_line == "This model flags from about 22 %."
    assert "within a percent" in (shown.flag_line_note or "")
    assert "either way" in (shown.flag_line_note or "")


def test_flagged_meaning_states_false_alarms() -> None:
    meaning = " ".join(show(assessment(0.4, True)).meaning)
    assert "Of every 100 machine failures, about 86 are flagged." in meaning
    assert "Of every 100 Flags, about 80 are real failures." in meaning
    assert "About 36 of every 1,000 Readings are flagged." in meaning
    assert "About 20 of every 100 Flags are false alarms" in meaning
    assert "cross-validation on the training data" in meaning


def test_unflagged_meaning_states_missed_failures_and_is_no_clearance() -> None:
    shown = show(assessment(0.03, False))
    assert shown.state == "no_inspection_flagged"
    assert shown.headline == "No inspection flagged"
    meaning = " ".join(shown.meaning)
    assert "About 14 of every 100 real failures are not flagged" in meaning
    assert "false alarms" not in meaning


def test_the_flag_decides_the_headline_not_the_rounded_risk() -> None:
    # 21.7 % and 21.9 % both read "about 22 %", but only one is at or above the 21.83 % line.
    below, above = show(assessment(0.217, False)), show(assessment(0.219, True))
    assert below.failure_risk == above.failure_risk
    assert (below.headline, above.headline) == ("No inspection flagged", "Inspect")


def test_extreme_figures_never_read_as_certainty() -> None:
    perfect = {**METRICS, "ratios": {"10.0": {**FIGURES, "recall": 1.0}}}
    meaning = " ".join(
        present(assessment(0.4, True), POINT, perfect, cost_ratio=COST_RATIO).meaning
    )
    assert "more than 99 are flagged" in meaning
    assert "100" not in re.sub(r"[Oo]f every 100", "", meaning)


def test_undefined_figures_say_so_instead_of_inventing_numbers() -> None:
    undefined = {
        **METRICS,
        "ratios": {"10.0": {"recall": None, "precision": None, "flags_per_1000": None}},
    }
    shown = present(assessment(0.4, True), POINT, undefined, cost_ratio=COST_RATIO)
    assert shown.meaning == ("What a Flag means is not available for this model.",)


def test_not_assessed_lists_inputs_and_shows_no_number() -> None:
    shown = show(REFUSED)
    assert shown.state == "not_assessed"
    assert shown.headline == "Not assessed"
    assert shown.refusal == (
        "Torque [Nm]: absent",
        "Rotational speed [rpm]: out of range",
        "No risk or Flag is given. This does not mean the machine is fine. Enter the missing "
        "values and assess again; if a sensor is offline, treat this machine as unassessed.",
    )
    assert shown.failure_risk is None
    assert shown.flag_line is None
    assert shown.flag_line_note is None
    assert shown.meaning == ()
    rendered = "\n".join(shown.lines())
    assert "recoverable" not in rendered
    assert "22" not in rendered  # the threshold
    assert "%" not in rendered


def test_unavailable_model_is_an_error_state_without_numbers() -> None:
    shown = all_states()["model_unavailable"]
    assert shown.state == "model_unavailable"
    assert "does not match" in " ".join(shown.refusal)
    assert "This does not mean the machine is fine" in " ".join(shown.refusal)
    assert shown.failure_risk is None
    assert shown.flag_line is None


def test_a_cost_ratio_that_differs_from_the_unit_is_refused() -> None:
    with pytest.raises(ValueError, match="cost ratio"):
        present(assessment(0.4, True), POINT, METRICS, cost_ratio=20.0)


def test_unknown_metric_source_is_refused() -> None:
    with pytest.raises(ValueError, match="source"):
        present(assessment(0.4, True), POINT, {**METRICS, "source": "vibes"}, cost_ratio=COST_RATIO)


def test_holdout_figures_are_labelled_as_such() -> None:
    holdout = {**METRICS, "source": "holdout"}
    meaning = " ".join(
        present(assessment(0.4, True), POINT, holdout, cost_ratio=COST_RATIO).meaning
    )
    assert "held-out test data" in meaning


def test_example_reading_outcome_line() -> None:
    suffix = " This Reading was in the training data, so agreement proves nothing."
    assert example_outcome_line(True) == "Recorded in the data: Machine failure." + suffix
    assert example_outcome_line(False) == "Recorded in the data: no failure." + suffix
    assert show(assessment(0.4, True), recorded_outcome=False).outcome_line == example_outcome_line(
        False
    )
    assert show(assessment(0.4, True)).outcome_line is None
    assert show(REFUSED, recorded_outcome=True).outcome_line is None


# --- every state: caveats present, no banned wording --------------------------------------------


@pytest.mark.parametrize("state", list(all_states()))
def test_every_state_carries_the_caveats_and_names_the_model(state: str) -> None:
    text = "\n".join(all_states()[state].lines())
    for caveat in (
        "Estimated from historical readings.",
        "Synthetic benchmark data, not your plant.",
        "Assumes one missed failure costs as much as 10 false alarms. "
        "This is an assumption, not a measurement.",
        "An aid for prioritising inspections, not a guarantee.",
        "Model: Random forest (unweighted).",
    ):
        assert caveat in text


def test_the_cost_ratio_caveat_follows_the_config_value() -> None:
    point = OperatingPoint(MODEL, 7.5, {5.0: 0.24, 7.5: 0.2}, None, FLOOR)
    metrics = {**METRICS, "ratios": {"7.5": FIGURES}}
    shown = present(assessment(0.4, True), point, metrics, cost_ratio=7.5)
    assert any("as much as 7.5 false alarms" in caveat for caveat in shown.caveats)


@pytest.mark.parametrize("state", list(all_states()))
def test_no_state_contains_banned_wording(state: str) -> None:
    assert banned_wording_in("\n".join(all_states()[state].lines())) == []


@pytest.mark.parametrize("score", [0.0, 0.0099, 0.01, 0.1, 0.2183, 0.5, 0.95, 0.96, 1.0])
@pytest.mark.parametrize("flag", [True, False])
def test_no_risk_value_can_produce_banned_wording(score: float, flag: bool) -> None:
    assert banned_wording_in("\n".join(show(assessment(score, flag)).lines())) == []


@pytest.mark.parametrize(
    "sample",
    [
        "This machine will fail",
        "The machine is Safe",
        "a healthy machine",
        "Status: OK",
        "There is no risk here",
        "Failure risk: 0 %",
        "Failure risk: 100 %",
        "Failure risk: 100%",
        "Failure risk: 100%",
        "Failure risk: 100.0 %",
        "Failure risk: 0.0 %",
        "Failure risk: 100\u00a0%",
        "There is no-risk here",
        "guaranteed to run",
        "The model predicts failure",
        "the probability of failure is 3 %",
        "an accuracy of 97 %",
    ],
)
def test_the_scanner_catches_each_banned_phrase(sample: str) -> None:
    assert banned_wording_in(sample), sample


def test_an_unavailable_reason_with_banned_wording_is_not_shown() -> None:
    shown = present_unavailable(
        DisabledUnit(MODEL, "cannot be loaded: RuntimeError('ok, 100 % safe')"),
        cost_ratio=COST_RATIO,
    )
    assert banned_wording_in("\n".join(shown.lines())) == []
    assert "safe" not in " ".join(shown.refusal)


@pytest.mark.parametrize(
    "sample",
    ["about 10 %", "about 20 %", "about 100 of every 1,000", "not a guarantee", "Okay"],
)
def test_the_scanner_does_not_flag_innocent_text(sample: str) -> None:
    assert banned_wording_in(sample) == []


def test_every_banned_phrase_is_listed_once() -> None:
    assert len({name for name, _ in BANNED_WORDING}) == len(BANNED_WORDING)


# --- golden check banner -------------------------------------------------------------------------

GOLDEN_FAILED = GoldenResult(differences=("Reading 2: stored score 0.25, scored 0.5",))


def test_a_passing_golden_check_shows_no_banner() -> None:
    assert golden_banner(GoldenResult(differences=())) is None


def test_a_failed_golden_check_says_results_may_be_wrong_and_names_the_difference() -> None:
    banner = golden_banner(GOLDEN_FAILED)
    assert banner is not None
    assert banner.startswith("Scoring check failed.")
    assert "Do not rely on it" in banner
    assert "Reading 2: stored score 0.25, scored 0.5" in banner


def test_a_failed_golden_check_banner_has_no_banned_wording() -> None:
    banner = golden_banner(GOLDEN_FAILED)
    assert banner is not None and banned_wording_in(banner) == []


def test_the_banner_lists_three_differences_and_counts_the_rest() -> None:
    many = GoldenResult(differences=tuple(f"Reading {n}: differs" for n in range(1, 6)))
    banner = golden_banner(many)
    assert banner is not None
    assert "Reading 3: differs" in banner
    assert "Reading 4" not in banner
    assert "2 more" in banner


def test_a_difference_with_banned_wording_is_not_shown() -> None:
    banner = golden_banner(GoldenResult(differences=("ok, 100 % safe",)))
    assert banner is not None
    assert banned_wording_in(banner) == []
    assert "safe" not in banner
    assert "details withheld" in banner


def test_the_banner_never_ends_in_a_doubled_full_stop() -> None:
    banner = golden_banner(GoldenResult(differences=("cannot be read: KeyError('golden').",)))
    assert banner is not None and not banner.endswith("..")
