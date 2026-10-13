"""Run configuration from ``configs/default.toml`` (single source of seed and split settings)."""

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
class Config:
    """Run configuration. All randomness in the project is seeded from ``seed``."""

    seed: int
    split: SplitConfig


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


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load the run configuration from a TOML file.

    Raises:
        ValueError: if ``seed`` or the ``[split]`` table and its settings are absent or invalid.
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
    )
