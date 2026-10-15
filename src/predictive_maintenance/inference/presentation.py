"""All engineer-facing wording of one assessment (ADR 0009); imports no UI framework.

``present`` maps an ``Assessment`` plus the unit's OperatingPoint and operating-metrics to a
``Presentation``; ``present_unavailable`` does the same for a unit that failed its checks. Every
rounding rule and every word an engineer sees lives here, so the app only renders strings. This
module also owns ``BANNED_WORDING``: wording that overstates certainty and must never be rendered.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

from predictive_maintenance.evaluation.threshold import OperatingPoint
from predictive_maintenance.inference.artifact import DisabledUnit
from predictive_maintenance.inference.assessment import Assessment
from predictive_maintenance.inference.golden import GoldenResult
from predictive_maintenance.training.unit import ratio_key

State = Literal["inspect", "no_inspection_flagged", "not_assessed", "model_unavailable"]

HIGH_RISK = 0.95  # above this the risk reads "over 95 %", never "100 %"

# Where a quoted figure comes from, by the ``source`` label of the operating-metrics file.
FIGURE_SOURCES = {
    "cv_train": "cross-validation on the training data",
    "holdout": "held-out test data",
}
MODEL_NAMES = {"random_forest_unweighted": "Random forest (unweighted)"}
# Cost ratios whose best Flag line a model does not pin down (ADR 0007: a non-flat optimum), so
# their sensitivity row is marked.
NOT_PRECISE = {"random_forest_unweighted": (20.0,)}
NOT_PRECISE_NOTE = "not precise"

# Names of phrases that overstate certainty, with the pattern that finds each. "0 %" and "100 %"
# must not match inside "10 %" or "100 of every 1,000". "no risk" skips the mandated sentence "No
# risk or Flag is given". Any accuracy claim is banned outright: figures are quoted as recall and
# precision together with their source instead.
BANNED_WORDING: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("will fail", re.compile(r"\bwill fail\b", re.IGNORECASE)),
    ("safe", re.compile(r"\bsafe(?:ly)?\b", re.IGNORECASE)),
    ("healthy", re.compile(r"\bhealthy\b", re.IGNORECASE)),
    ("OK", re.compile(r"\bOK\b")),
    ("no risk", re.compile(r"\bno[\s-]+risk\b(?! or Flag is given)", re.IGNORECASE)),
    ("0 %", re.compile(r"(?<![\d.,])0(?:\.0+)?[ \u00a0]?%")),
    ("100 %", re.compile(r"(?<![\d.,])100(?:\.0+)?[ \u00a0]?%")),
    ("guaranteed", re.compile(r"\bguaranteed\b", re.IGNORECASE)),
    ("predicts failure", re.compile(r"\bpredicts? failure\b", re.IGNORECASE)),
    ("probability of failure", re.compile(r"\bprobability of failure\b", re.IGNORECASE)),
    ("accuracy claim", re.compile(r"\baccura(?:te|tely|cy)\b", re.IGNORECASE)),
)

NOT_ASSESSED_NOTICE = (
    "No risk or Flag is given. This does not mean the machine is fine. Enter the missing values "
    "and assess again; if a sensor is offline, treat this machine as unassessed."
)
UNAVAILABLE_NOTICE = (
    "No risk or Flag is given. This does not mean the machine is fine. Try again later; until "
    "then treat this machine as unassessed."
)
GOLDEN_FAILED_NOTICE = (
    "Scoring check failed. The stored test scores of this model could not be reproduced here, "
    "so the Failure risk shown below may be wrong. Do not rely on it until this is resolved."
)
GOLDEN_WITHHELD = "details withheld"
GOLDEN_SHOWN = 3  # differences quoted in the banner; the rest are counted
FLAG_LINE_NOTE = "A risk within a percent of this line can go either way."
OUTCOME_SUFFIX = "This Reading was in the training data, so agreement proves nothing."
RANGE_NOTE = "outside the range of the training data, treat the result as unreliable."
MEANING_TITLE = "What this Flag means"
SENSITIVITY_TITLE = "How the Flag changes with the cost assumption"
UNAVAILABLE_FIGURES = "figures not available"


def banned_wording_in(text: str) -> list[str]:
    """The names of the banned phrases found in ``text`` (empty if it is clean)."""
    return [name for name, pattern in BANNED_WORDING if pattern.search(text)]


@dataclass(frozen=True)
class SensitivityRow:
    """The Flag's behaviour at one Cost ratio, in words (metrics only, never a threshold).

    ``note`` is "not precise" where the model's best Flag line at that ratio is not pinned down.
    """

    ratio: str
    text: str
    note: str | None


@dataclass(frozen=True)
class Presentation:
    """Every string for one result; a field that is None or empty is not shown.

    A Not assessed or model-unavailable state has no ``failure_risk``, ``flag_line`` or
    ``meaning``, so no number can be rendered for it.
    """

    state: State
    headline: str
    failure_risk: str | None
    flag_line: str | None
    flag_line_note: str | None
    meaning: tuple[str, ...]
    refusal: tuple[str, ...]
    outcome_line: str | None
    model_line: str
    caveats: tuple[str, ...]
    range_note: str | None = None
    model_card_title: str = ""
    meaning_title: str | None = None
    sensitivity_title: str | None = None
    sensitivity_intro: str | None = None
    sensitivity: tuple[SensitivityRow, ...] = ()
    sensitivity_source: str | None = None

    def sensitivity_lines(self) -> tuple[str, ...]:
        """The strings inside the sensitivity expander, in display order (empty if none)."""
        if not self.sensitivity:
            return ()
        return (
            *((self.sensitivity_intro,) if self.sensitivity_intro else ()),
            *(row.text for row in self.sensitivity),
            *((self.sensitivity_source,) if self.sensitivity_source else ()),
        )

    def lines(self) -> tuple[str, ...]:
        """Every string of the state, in display order."""
        risk = (self.failure_risk, self.flag_line, self.flag_line_note, self.range_note)
        return (
            self.headline,
            *(line for line in risk if line),
            *self.meaning,
            *self.sensitivity_lines(),
            *self.refusal,
            *((self.outcome_line,) if self.outcome_line else ()),
            self.model_line,
            *self.caveats,
        )


def _half_up(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _percent(fraction: float) -> int:
    """Nearest whole percent, halves up (the decimal repr avoids binary rounding surprises)."""
    return _half_up(Decimal(repr(fraction)) * 100)


def risk_text(score: float, floor: float) -> str:
    """A Failure risk in words: "under 1 %" below the Risk floor, "about N %", "over 95 %".

    "0 %" and "100 %" are never produced: they would claim certainty a rough estimate cannot have.
    """
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"a risk must be a number between 0 and 1, got {score!r}")
    if score > HIGH_RISK:
        return "over 95 %"
    percent = _percent(score)
    if score < floor or percent < 1:
        return "under 1 %"
    return f"about {percent} %"


def flag_line_text(threshold: float, floor: float) -> str:
    """The model's own Flag line, rounded the same way as the Failure risk."""
    return f"This model flags from {risk_text(threshold, floor)}."


def _count(n: int, of: int = 100) -> str:
    """A count out of ``of`` in words, never claiming none or all."""
    if n < 1:
        return "fewer than 1"
    if n > of - 1:
        return f"more than {of - 1}"
    return f"about {n}"


def _sentence(text: str) -> str:
    return text[0].upper() + text[1:]


def model_name(model: str) -> str:
    """The name an engineer sees for a model id."""
    return MODEL_NAMES.get(model, model.replace("_", " ").capitalize())


def model_card_title(model: str) -> str:
    """The title of the expander that holds the model's name and the always-on caveats."""
    return f"Model card: {model_name(model)}"


def _caveats(cost_ratio: float) -> tuple[str, ...]:
    return (
        "Estimated from historical readings.",
        f"Assumes one missed failure costs as much as {cost_ratio:g} false alarms. "
        "This is an assumption, not a measurement.",
        "An aid for prioritising inspections, not a guarantee.",
    )


def _source_text(metrics: Mapping[str, Any]) -> str:
    source = FIGURE_SOURCES.get(metrics.get("source", ""))
    if source is None:
        raise ValueError(f"unknown figure source {metrics.get('source')!r} in operating-metrics")
    return source


def _rounded(figures: Mapping[str, Any]) -> tuple[int, int, int] | None:
    """Recall and precision as whole percents and Flags per 1,000, or None if any is undefined."""
    recall, precision, per_1000 = (
        figures.get("recall"),
        figures.get("precision"),
        figures.get("flags_per_1000"),
    )
    if recall is None or precision is None or per_1000 is None:
        return None
    return (
        min(max(_percent(recall), 0), 100),
        min(max(_percent(precision), 0), 100),
        _half_up(Decimal(repr(per_1000))),
    )


def _figures_at(metrics: Mapping[str, Any], ratio: float) -> tuple[str, Mapping[str, Any]]:
    """The source text and figures for ``ratio``: the top-level ones if present, else the CV set.

    Once the Holdout is applied only the default ratio is top-level; the other ratios keep their
    cross-validation figures under ``sensitivity``.
    """
    key = ratio_key(ratio)
    if key in metrics["ratios"]:
        return _source_text(metrics), metrics["ratios"][key]
    stored = metrics.get("sensitivity", {})
    if key not in stored.get("ratios", {}):
        raise ValueError(f"no operating figures for cost ratio {ratio:g}")
    return _source_text(stored), stored["ratios"][key]


def _meaning(metrics: Mapping[str, Any], ratio: float, flagged: bool) -> tuple[str, ...]:
    source, figures = _figures_at(metrics, ratio)
    rounded = _rounded(figures)
    if rounded is None:
        return ("What a Flag means is not available for this model.",)
    caught, real, flags = rounded
    follow_up = (
        f"{_sentence(_count(100 - real))} of every 100 Flags are false alarms, so some "
        "inspections will find nothing."
        if flagged
        else f"{_sentence(_count(100 - caught))} of every 100 real failures are not flagged, so "
        "no Flag is not a clearance."
    )
    return (
        f"Of every 100 machine failures, {_count(caught)} are flagged.",
        f"Of every 100 Flags, {_count(real)} are real failures.",
        f"{_sentence(_count(flags, 1000))} of every 1,000 Readings are flagged.",
        follow_up,
        f"Figures from {source}.",
    )


def _sensitivity_row(model: str, ratio: float, figures: Mapping[str, Any]) -> SensitivityRow:
    label = f"{ratio:g}:1"
    note = NOT_PRECISE_NOTE if ratio in NOT_PRECISE.get(model, ()) else None
    rounded = _rounded(figures)
    if rounded is None:
        text = f"{label}: {UNAVAILABLE_FIGURES}."
    else:
        caught, real, flags = rounded
        text = (
            f"{label}: of every 100 failures, {_count(caught)} are flagged; "
            f"of every 100 Flags, {_count(real)} are real; "
            f"{_count(flags, 1000)} of every 1,000 Readings are flagged."
        )
    return SensitivityRow(label, f"{text} ({note})" if note else text, note)


def _sensitivity(model: str, metrics: Mapping[str, Any], cost_ratio: float) -> dict[str, Any]:
    """The expander's fields: one row per stored Cost ratio, from the cross-validation figures.

    Once the Holdout is applied the top-level ratios hold only the default one, and the full set
    sits under ``sensitivity``.
    """
    stored = metrics.get("sensitivity", metrics)
    source = _source_text(stored)
    rows = tuple(
        _sensitivity_row(model, ratio, figures)
        for ratio, figures in sorted((float(key), value) for key, value in stored["ratios"].items())
    )
    return {
        "sensitivity_title": SENSITIVITY_TITLE,
        "sensitivity_intro": (
            f"For information only: the Flag above uses the {cost_ratio:g}:1 ratio chosen "
            "above; the rows below show every ratio."
        ),
        "sensitivity": rows,
        "sensitivity_source": f"Figures from {source}.",
    }


def _range_note(assessment: Assessment) -> str | None:
    if not assessment.outside_training:
        return None
    return f"{', '.join(assessment.outside_training)}: {RANGE_NOTE}"


def example_outcome_line(machine_failure: bool) -> str:
    """The muted line for an Example reading: its recorded outcome, and why it proves nothing."""
    outcome = "Machine failure" if machine_failure else "no failure"
    return f"Recorded in the data: {outcome}. {OUTCOME_SUFFIX}"


def present(
    assessment: Assessment,
    operating_point: OperatingPoint,
    operating_metrics: Mapping[str, Any],
    *,
    cost_ratio: float,
    recorded_outcome: bool | None = None,
) -> Presentation:
    """The strings for one assessment from a loaded unit.

    ``cost_ratio`` is the Cost ratio the engineer chose (the configured default unless changed).
    The Flag line and figures are those of that ratio, so the unit must store an Operating
    threshold and figures for it; a stale unit is refused rather than quoted under the wrong
    assumption.
    ``recorded_outcome`` is the Machine failure recorded for an unedited Example reading; it is
    shown only with a complete assessment.
    """
    if cost_ratio not in operating_point.thresholds:
        known = ", ".join(f"{r:g}" for r in sorted(operating_point.thresholds))
        raise ValueError(f"the unit has no threshold for cost ratio {cost_ratio:g}; it has {known}")
    model_line = f"Model: {model_name(operating_point.model)}."
    caveats = _caveats(cost_ratio)
    if assessment.status == "refused":
        return Presentation(
            state="not_assessed",
            headline="Not assessed",
            failure_risk=None,
            flag_line=None,
            flag_line_note=None,
            meaning=(),
            refusal=(
                *(f"{item.name}: {item.reason}" for item in assessment.missing),
                NOT_ASSESSED_NOTICE,
            ),
            outcome_line=None,
            model_line=model_line,
            caveats=caveats,
            model_card_title=model_card_title(operating_point.model),
        )
    if assessment.failure_risk is None or assessment.flag is None:
        raise ValueError("a complete assessment needs a failure risk and a flag")
    floor = operating_point.risk_floor
    return Presentation(
        state="inspect" if assessment.flag else "no_inspection_flagged",
        headline="Inspect" if assessment.flag else "No inspection flagged",
        failure_risk="Failure risk (rough estimate, not calibrated): "
        + risk_text(assessment.failure_risk, floor),
        flag_line=flag_line_text(operating_point.thresholds[cost_ratio], floor),
        flag_line_note=FLAG_LINE_NOTE,
        meaning=_meaning(operating_metrics, cost_ratio, assessment.flag),
        meaning_title=MEANING_TITLE,
        refusal=(),
        outcome_line=None if recorded_outcome is None else example_outcome_line(recorded_outcome),
        model_line=model_line,
        caveats=caveats,
        model_card_title=model_card_title(operating_point.model),
        range_note=_range_note(assessment),
        **_sensitivity(operating_point.model, operating_metrics, cost_ratio),
    )


def present_unavailable(unit: DisabledUnit, *, cost_ratio: float) -> Presentation:
    """The error state for a unit that failed its checks: no risk, Flag or threshold."""
    name = model_name(unit.model)
    # The reason can quote an arbitrary exception, so it is shown only if it passes the same scan.
    reason = unit.reason if not banned_wording_in(unit.reason) else "its files failed their checks"
    return Presentation(
        state="model_unavailable",
        headline="Model unavailable",
        failure_risk=None,
        flag_line=None,
        flag_line_note=None,
        meaning=(),
        refusal=(f"The {name} model could not be used: {reason}.", UNAVAILABLE_NOTICE),
        outcome_line=None,
        model_line=f"Model: {name}.",
        caveats=_caveats(cost_ratio),
        model_card_title=model_card_title(unit.model),
    )


def golden_banner(result: GoldenResult) -> str | None:
    """The failure banner for a failed golden check, or None when it passed."""
    if result.passed:
        return None
    # A difference can quote an arbitrary exception, so each is shown only if it passes the scan.
    shown = [
        line.rstrip(". ") if not banned_wording_in(line) else GOLDEN_WITHHELD
        for line in result.differences[:GOLDEN_SHOWN]
    ]
    rest = len(result.differences) - len(shown)
    detail = "; ".join(shown) + (f"; and {rest} more" if rest > 0 else "")
    return f"{GOLDEN_FAILED_NOTICE} {detail}."
