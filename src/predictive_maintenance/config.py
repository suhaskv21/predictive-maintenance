"""Run configuration from ``configs/default.toml``: the single source of seed and run settings."""

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from predictive_maintenance.data import schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"

# [plausibility] key -> the numeric input (dataset column) it bounds.
_PLAUSIBILITY_INPUTS = dict(
    zip(
        ("air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"),
        schema.NUMERIC_FEATURES,
        strict=True,
    )
)


@dataclass(frozen=True)
class SplitConfig:
    """Split and validation settings (ADR 0005)."""

    test_size: float
    cv_folds: int
    cv_repeats: int


@dataclass(frozen=True)
class ThresholdConfig:
    """Operating-threshold settings (ADR 0007).

    A cost ratio is an assumption, not a measurement: how many false alarms (one unnecessary
    inspection each) one missed failure (unplanned downtime) is worth. A ratio of 10 means a missed
    failure is treated as costing as much as 10 false alarms. Higher ratios flag more Readings.
    ``cost_ratios`` are the ratios shown side by side, ascending; ``default_cost_ratio`` is the one
    that sets the operating threshold, and must be one of them and larger than the smallest.
    """

    default_cost_ratio: float
    cost_ratios: tuple[float, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation settings: ``bootstrap_resamples`` for confidence intervals on test metrics."""

    bootstrap_resamples: int


@dataclass(frozen=True)
class MissingDataConfig:
    """Missing-data policy margins (ADR 0008), all judgement calls fixed before any number.

    A pattern of absent inputs is scorable only if, at the frozen Operating threshold, it keeps at
    least ``min_retained_saving`` of the complete-data saving over *flag nothing* (after subtracting
    one corrected SE), recall falls by at most ``max_recall_drop`` (absolute), and the Flag rate
    stays within ``flag_rate_ratio_min`` to ``flag_rate_ratio_max`` of the complete-data rate.
    A more complex strategy replaces a simpler one only by ``challenger_margin`` of retained saving.
    ``stress_speed_quantile`` sets "high speed" for the informative-missingness stress, and
    ``assumed_missing_rates`` are per-input rates (assumptions, the data has no nulls) for the
    data-loss check.
    """

    min_retained_saving: float
    max_recall_drop: float
    flag_rate_ratio_min: float
    flag_rate_ratio_max: float
    challenger_margin: float
    stress_speed_quantile: float
    assumed_missing_rates: tuple[float, ...]


@dataclass(frozen=True)
class PipelineHealthConfig:
    """Review triggers for the data-pipeline health tab (ADR 0008), defaults set by the owner.

    Rates are shares of all Readings in a window of ``review_window_weeks`` consecutive weeks, each
    with at least ``min_weekly_readings`` Readings. Imputation is worth reviewing when the
    Recoverable refusal rate reaches ``recoverable_refusal_rate`` and refusals are spread out
    (the ``top_sources`` most-refusing sources hold less than ``max_source_concentration``); it is
    not the lever when the unrecoverable rate reaches ``unrecoverable_refusal_rate``. Absence looks
    informative when one training-quartile band holds more than ``informative_band_share`` of
    refused Readings, or refused Readings fail more than ``informative_failure_rate_ratio`` times
    as often as complete ones.
    """

    review_window_weeks: int
    min_weekly_readings: int
    recoverable_refusal_rate: float
    unrecoverable_refusal_rate: float
    top_sources: int
    max_source_concentration: float
    informative_band_share: float
    informative_failure_rate_ratio: float


@dataclass(frozen=True)
class FailureModesConfig:
    """Rule for the per-mode predictability check (ADR 0009), fixed before any number.

    A Failure mode is predictable enough to justify a follow-up ADR only if its mean repeated-CV
    PR-AUC is at least ``min_pr_auc``, at least ``min_lift`` times the mode's prevalence, and
    exceeds the prevalence by more than one corrected standard error.
    """

    min_pr_auc: float
    min_lift: float


@dataclass(frozen=True)
class AppConfig:
    """App settings (ADR 0009).

    ``example_count`` is how many Example readings are offered, of which ``example_failures`` are
    recorded failures.
    """

    example_count: int
    example_failures: int


@dataclass(frozen=True)
class DeployConfig:
    """Deployment checks (ADR 0010, ADR 0012).

    ``accepted_thresholds`` maps a model name to its Operating threshold accepted in ADR 0007 at
    the default cost ratio; training fails if its re-derived threshold, rounded to 3 places,
    differs by more than ``threshold_tolerance``. ``site_budget_mb`` is the largest exported site
    the build accepts.
    """

    accepted_thresholds: Mapping[str, float]
    threshold_tolerance: float
    site_budget_mb: float


@dataclass(frozen=True)
class InputRange:
    """The physically possible values of one numeric input: ``low`` to ``high`` inclusive.

    ``low_inclusive`` is False for an input that must be strictly above ``low`` (rotational speed).
    """

    low: float
    high: float
    low_inclusive: bool = True

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the range."""
        above = value >= self.low if self.low_inclusive else value > self.low
        return above and value <= self.high


@dataclass(frozen=True)
class PlausibilityConfig:
    """Which input values are physically possible (ADR 0013); any other value is treated as absent.

    ``ranges`` maps each numeric input (by its dataset column name) to its valid range and
    ``types`` lists the valid product Types. Values inside these but outside the training data's
    range are still scored.
    """

    ranges: Mapping[str, InputRange]
    types: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    """Run configuration. All randomness in the project is seeded from ``seed``."""

    seed: int
    split: SplitConfig
    threshold: ThresholdConfig
    evaluation: EvaluationConfig
    missing_data: MissingDataConfig
    pipeline_health: PipelineHealthConfig
    failure_modes: FailureModesConfig
    app: AppConfig
    deploy: DeployConfig
    plausibility: PlausibilityConfig


def _positive(value: object, name: str, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{where}: {name!r} must be a positive number, got {value!r}")
    return float(value)


def _integer(table: dict[str, object], name: str, where: str, minimum: int | None = None) -> int:
    value = table.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or (minimum is not None and value < minimum)
    ):
        bound = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{where}: {name!r} must be an integer{bound}, got {value!r}")
    return value


def _threshold(raw: dict[str, object], path: Path) -> ThresholdConfig:
    table = raw.get("threshold")
    if not isinstance(table, dict):
        raise ValueError(f"{path}: a [threshold] table is required")
    where = f"{path} [threshold]"

    default = _positive(table.get("default_cost_ratio"), "default_cost_ratio", where)
    listed = table.get("cost_ratios")
    if not isinstance(listed, list) or not listed:
        raise ValueError(f"{where}: 'cost_ratios' must be a non-empty list, got {listed!r}")
    ratios = tuple(_positive(value, "cost_ratios", where) for value in listed)
    if list(ratios) != sorted(set(ratios)):
        raise ValueError(f"{where}: 'cost_ratios' must be ascending and distinct, got {listed!r}")
    if default not in ratios:
        raise ValueError(
            f"{where}: 'default_cost_ratio' must be one of 'cost_ratios', got {default}"
        )
    if default == ratios[0]:
        raise ValueError(
            f"{where}: 'default_cost_ratio' must exceed the smallest of 'cost_ratios' ({ratios[0]})"
        )
    return ThresholdConfig(default_cost_ratio=default, cost_ratios=ratios)


def _evaluation(raw: dict[str, object], path: Path) -> EvaluationConfig:
    table = raw.get("evaluation")
    if not isinstance(table, dict):
        raise ValueError(f"{path}: an [evaluation] table is required")
    return EvaluationConfig(
        bootstrap_resamples=_integer(
            table, "bootstrap_resamples", f"{path} [evaluation]", minimum=1
        )
    )


def _missing_data(raw: dict[str, object], path: Path) -> MissingDataConfig:
    table = raw.get("missing_data")
    if not isinstance(table, dict):
        raise ValueError(f"{path}: a [missing_data] table is required")
    where = f"{path} [missing_data]"

    def number(name: str, *, exclusive_high: bool = False) -> float:
        """A number in (0, 1] (or (0, 1) when ``exclusive_high``)."""
        value = table.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{where}: {name!r} must be a number in (0, 1], got {value!r}")
        if value <= 0 or value > 1 or (exclusive_high and value == 1):
            raise ValueError(f"{where}: {name!r} must be a number in (0, 1], got {value!r}")
        return float(value)

    ratio_min = number("flag_rate_ratio_min", exclusive_high=True)
    ratio_max = table.get("flag_rate_ratio_max")
    if isinstance(ratio_max, bool) or not isinstance(ratio_max, int | float) or ratio_max < 1:
        raise ValueError(f"{where}: 'flag_rate_ratio_max' must be a number >= 1, got {ratio_max!r}")
    rates = table.get("assumed_missing_rates")
    if (
        not isinstance(rates, list)
        or not rates
        or any(
            isinstance(r, bool) or not isinstance(r, int | float) or not 0 < r < 1 for r in rates
        )
        or list(rates) != sorted(set(rates))
    ):
        raise ValueError(
            f"{where}: 'assumed_missing_rates' must be ascending, distinct numbers in (0, 1), "
            f"got {rates!r}"
        )
    return MissingDataConfig(
        min_retained_saving=number("min_retained_saving"),
        max_recall_drop=number("max_recall_drop"),
        flag_rate_ratio_min=ratio_min,
        flag_rate_ratio_max=float(ratio_max),
        challenger_margin=number("challenger_margin"),
        stress_speed_quantile=number("stress_speed_quantile", exclusive_high=True),
        assumed_missing_rates=tuple(float(r) for r in rates),
    )


def _pipeline_health(raw: dict[str, object], path: Path) -> PipelineHealthConfig:
    table = raw.get("pipeline_health")
    if not isinstance(table, dict):
        raise ValueError(f"{path}: a [pipeline_health] table is required")
    where = f"{path} [pipeline_health]"

    def share(name: str) -> float:
        value = table.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value <= 1:
            raise ValueError(f"{where}: {name!r} must be a number in (0, 1], got {value!r}")
        return float(value)

    ratio = table.get("informative_failure_rate_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, int | float) or ratio <= 1:
        raise ValueError(
            f"{where}: 'informative_failure_rate_ratio' must be a number > 1, got {ratio!r}"
        )
    return PipelineHealthConfig(
        review_window_weeks=_integer(table, "review_window_weeks", where, minimum=1),
        min_weekly_readings=_integer(table, "min_weekly_readings", where, minimum=1),
        recoverable_refusal_rate=share("recoverable_refusal_rate"),
        unrecoverable_refusal_rate=share("unrecoverable_refusal_rate"),
        top_sources=_integer(table, "top_sources", where, minimum=1),
        max_source_concentration=share("max_source_concentration"),
        informative_band_share=share("informative_band_share"),
        informative_failure_rate_ratio=float(ratio),
    )


def _failure_modes(raw: dict[str, object], path: Path) -> FailureModesConfig:
    table = raw.get("failure_modes")
    if not isinstance(table, dict):
        raise ValueError(f"{path}: a [failure_modes] table is required")
    where = f"{path} [failure_modes]"
    pr_auc = table.get("min_pr_auc")
    if isinstance(pr_auc, bool) or not isinstance(pr_auc, int | float) or not 0 < pr_auc < 1:
        raise ValueError(f"{where}: 'min_pr_auc' must be a number in (0, 1), got {pr_auc!r}")
    lift = table.get("min_lift")
    if isinstance(lift, bool) or not isinstance(lift, int | float) or lift <= 1:
        raise ValueError(f"{where}: 'min_lift' must be a number > 1, got {lift!r}")
    return FailureModesConfig(min_pr_auc=float(pr_auc), min_lift=float(lift))


def _app(raw: dict[str, object], path: Path) -> AppConfig:
    table = raw.get("app")
    if not isinstance(table, dict):
        raise ValueError(f"{path}: an [app] table is required")
    where = f"{path} [app]"
    count = _integer(table, "example_count", where, minimum=1)
    failures = _integer(table, "example_failures", where, minimum=0)
    if failures > count:
        raise ValueError(
            f"{where}: 'example_failures' must not exceed 'example_count' ({count}), got {failures}"
        )
    return AppConfig(example_count=count, example_failures=failures)


def _deploy(raw: dict[str, object], path: Path) -> DeployConfig:
    table = raw.get("deploy")
    if not isinstance(table, dict):
        raise ValueError(f"{path}: a [deploy] table is required")
    where = f"{path} [deploy]"

    accepted = table.get("accepted_thresholds")
    if not isinstance(accepted, dict) or not accepted:
        raise ValueError(
            f"{where}: 'accepted_thresholds' must be a non-empty table of model = threshold, "
            f"got {accepted!r}"
        )
    thresholds: dict[str, float] = {}
    for model, value in accepted.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 1:
            raise ValueError(
                f"{where}: 'accepted_thresholds' {model!r} must be in (0, 1), got {value!r}"
            )
        thresholds[model] = float(value)
    return DeployConfig(
        accepted_thresholds=thresholds,
        threshold_tolerance=_positive(
            table.get("threshold_tolerance"), "threshold_tolerance", where
        ),
        site_budget_mb=_positive(table.get("site_budget_mb"), "site_budget_mb", where),
    )


def _bound(table: dict[str, object], name: str, where: str) -> float:
    value = table.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{where}: {name!r} must be a finite number, got {value!r}")
    return float(value)


def _input_range(table: dict[str, object], key: str, path: Path) -> InputRange:
    where = f"{path} [plausibility] {key}"
    entry = table.get(key)
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: a table with 'min' (or 'above') and 'max' is required")
    if ("min" in entry) == ("above" in entry):
        raise ValueError(f"{where}: give exactly one of 'min' (inclusive) or 'above' (exclusive)")
    low_name = "min" if "min" in entry else "above"
    low = _bound(entry, low_name, where)
    high = _bound(entry, "max", where)
    if low >= high:
        raise ValueError(f"{where}: {low_name!r} must be below 'max', got {low} and {high}")
    return InputRange(low, high, low_inclusive=low_name == "min")


def _plausibility(raw: dict[str, object], path: Path) -> PlausibilityConfig:
    table = raw.get("plausibility")
    if not isinstance(table, dict):
        raise ValueError(f"{path}: a [plausibility] table is required")
    where = f"{path} [plausibility]"
    types = table.get("types")
    if (
        not isinstance(types, list)
        or not types
        or len(set(types)) != len(types)
        or any(t not in schema.TYPE_LEVELS for t in types)
    ):
        raise ValueError(
            f"{where}: 'types' must be distinct values from {list(schema.TYPE_LEVELS)}, "
            f"got {types!r}"
        )
    return PlausibilityConfig(
        ranges={
            column: _input_range(table, key, path) for key, column in _PLAUSIBILITY_INPUTS.items()
        },
        types=tuple(types),
    )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load the run configuration from a TOML file.

    Raises:
        ValueError: if ``seed`` or the ``[split]``, ``[threshold]``, ``[evaluation]``,
            ``[missing_data]``, ``[pipeline_health]``, ``[failure_modes]``, ``[app]``,
            ``[deploy]`` or ``[plausibility]`` tables and their settings are absent or invalid.
    """
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    seed = _integer(raw, "seed", str(path))
    split = raw.get("split")
    if not isinstance(split, dict):
        raise ValueError(f"{path}: a [split] table is required")
    test_size = split.get("test_size")
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, int | float)
        or not 0 < test_size < 1
    ):
        raise ValueError(f"{path}: 'test_size' must be a number in (0, 1), got {test_size!r}")
    return Config(
        seed=seed,
        split=SplitConfig(
            test_size=float(test_size),
            cv_folds=_integer(split, "cv_folds", f"{path} [split]", minimum=2),
            cv_repeats=_integer(split, "cv_repeats", f"{path} [split]", minimum=1),
        ),
        threshold=_threshold(raw, path),
        evaluation=_evaluation(raw, path),
        missing_data=_missing_data(raw, path),
        pipeline_health=_pipeline_health(raw, path),
        failure_modes=_failure_modes(raw, path),
        app=_app(raw, path),
        deploy=_deploy(raw, path),
        plausibility=_plausibility(raw, path),
    )
