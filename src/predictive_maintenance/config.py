"""Run configuration from ``configs/default.toml``: the single source of seed and run settings."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"


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
class Config:
    """Run configuration. All randomness in the project is seeded from ``seed``."""

    seed: int
    split: SplitConfig
    threshold: ThresholdConfig
    evaluation: EvaluationConfig


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

    def positive(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise ValueError(f"{where}: {name!r} must be a positive number, got {value!r}")
        return float(value)

    default = positive(table.get("default_cost_ratio"), "default_cost_ratio")
    listed = table.get("cost_ratios")
    if not isinstance(listed, list) or not listed:
        raise ValueError(f"{where}: 'cost_ratios' must be a non-empty list, got {listed!r}")
    ratios = tuple(positive(value, "cost_ratios") for value in listed)
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


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load the run configuration from a TOML file.

    Raises:
        ValueError: if ``seed`` or the ``[split]``, ``[threshold]`` or ``[evaluation]`` tables and
            their settings are absent or invalid.
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
    )
