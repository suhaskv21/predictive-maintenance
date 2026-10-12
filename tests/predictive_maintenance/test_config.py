from pathlib import Path

import pytest

from predictive_maintenance.config import DEFAULT_CONFIG_PATH, Config, load_config


def test_default_config_file_provides_the_seed() -> None:
    assert DEFAULT_CONFIG_PATH.name == "default.toml"
    config = load_config()
    assert isinstance(config.seed, int)


def test_load_config_reads_the_seed_from_the_given_file(tmp_path: Path) -> None:
    path = tmp_path / "custom.toml"
    path.write_text("seed = 7\n")
    assert load_config(path) == Config(seed=7)


def test_load_config_rejects_a_file_without_a_seed(tmp_path: Path) -> None:
    path = tmp_path / "empty.toml"
    path.write_text("")
    with pytest.raises(ValueError, match="seed"):
        load_config(path)


def test_load_config_rejects_a_non_integer_seed(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('seed = "forty-two"\n')
    with pytest.raises(ValueError, match="seed"):
        load_config(path)
