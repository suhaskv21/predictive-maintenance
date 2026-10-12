"""Run configuration loaded from ``configs/default.toml`` (single source of the seed)."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"


@dataclass(frozen=True)
class Config:
    """Run configuration. All randomness in the project is seeded from ``seed``."""

    seed: int


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load the run configuration from a TOML file.

    Raises:
        ValueError: if ``seed`` is absent or not an integer.
    """
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{path}: 'seed' must be an integer, got {seed!r}")
    return Config(seed=seed)
