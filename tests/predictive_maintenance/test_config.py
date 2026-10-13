from pathlib import Path

import pytest

from predictive_maintenance.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    EvaluationConfig,
    SplitConfig,
    ThresholdConfig,
    load_config,
)

VALID = """\
seed = 7

[split]
test_size = 0.25
cv_folds = 4
cv_repeats = 2

[threshold]
default_cost_ratio = 10
cost_ratios = [5, 10, 20]

[evaluation]
bootstrap_resamples = 500
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "custom.toml"
    path.write_text(text)
    return path


def test_default_config_file_provides_the_seed() -> None:
    assert DEFAULT_CONFIG_PATH.name == "default.toml"
    config = load_config()
    assert isinstance(config.seed, int)


def test_default_config_holds_the_split_decided_in_adr_0005() -> None:
    assert load_config().split == SplitConfig(test_size=0.3, cv_folds=5, cv_repeats=5)


def test_load_config_reads_seed_and_split_from_the_given_file(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, VALID))
    assert config == Config(
        seed=7,
        split=SplitConfig(test_size=0.25, cv_folds=4, cv_repeats=2),
        threshold=ThresholdConfig(default_cost_ratio=10.0, cost_ratios=(5.0, 10.0, 20.0)),
        evaluation=EvaluationConfig(bootstrap_resamples=500),
    )


def test_default_config_holds_the_threshold_settings_proposed_in_adr_0007() -> None:
    config = load_config()
    assert config.threshold == ThresholdConfig(
        default_cost_ratio=10.0, cost_ratios=(5.0, 10.0, 20.0)
    )
    assert config.evaluation == EvaluationConfig(bootstrap_resamples=2000)


@pytest.mark.parametrize("table", ["threshold", "evaluation"])
def test_load_config_rejects_a_file_without_the_table(tmp_path: Path, table: str) -> None:
    head, _, rest = VALID.partition(f"[{table}]")
    tail = rest.split("\n\n", 1)[1] if "\n\n" in rest else ""
    with pytest.raises(ValueError, match=rf"\[{table}\]"):
        load_config(write(tmp_path, head + tail))


@pytest.mark.parametrize(
    "bad_line",
    [
        "default_cost_ratio = 0",
        "default_cost_ratio = -1",
        "default_cost_ratio = true",
        'default_cost_ratio = "10"',
        "default_cost_ratio = 7",
        "default_cost_ratio = 5",
        "cost_ratios = []",
        "cost_ratios = [20, 10, 5]",
        "cost_ratios = [5, 5, 10]",
        "cost_ratios = [5, 0, 10]",
        'cost_ratios = "5, 10, 20"',
    ],
)
def test_load_config_rejects_invalid_threshold_settings(tmp_path: Path, bad_line: str) -> None:
    name = bad_line.split(" = ")[0]
    lines = [bad_line if line.startswith(name) else line for line in VALID.splitlines()]
    with pytest.raises(ValueError, match=name):
        load_config(write(tmp_path, "\n".join(lines)))


@pytest.mark.parametrize("name", ["default_cost_ratio", "cost_ratios"])
def test_load_config_rejects_a_missing_threshold_setting(tmp_path: Path, name: str) -> None:
    lines = [line for line in VALID.splitlines() if not line.startswith(name)]
    with pytest.raises(ValueError, match=name):
        load_config(write(tmp_path, "\n".join(lines)))


@pytest.mark.parametrize("bad", ["0", "1.5", "true", '"many"'])
def test_load_config_rejects_invalid_bootstrap_resamples(tmp_path: Path, bad: str) -> None:
    text = VALID.replace("bootstrap_resamples = 500", f"bootstrap_resamples = {bad}")
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        load_config(write(tmp_path, text))


def test_load_config_rejects_a_file_without_a_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed"):
        load_config(write(tmp_path, VALID.replace("seed = 7\n", "")))


def test_load_config_rejects_a_non_integer_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed"):
        load_config(write(tmp_path, VALID.replace("seed = 7", 'seed = "forty-two"')))


def test_load_config_rejects_a_file_without_a_split_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\[split\]"):
        load_config(write(tmp_path, "seed = 7\n"))


@pytest.mark.parametrize("name", ["test_size", "cv_folds", "cv_repeats"])
def test_load_config_rejects_a_missing_split_setting(tmp_path: Path, name: str) -> None:
    lines = [line for line in VALID.splitlines() if not line.startswith(name)]
    with pytest.raises(ValueError, match=name):
        load_config(write(tmp_path, "\n".join(lines)))


@pytest.mark.parametrize("bad", ["0", "1", "1.5", "-0.1", '"0.3"'])
def test_load_config_rejects_a_test_size_outside_zero_to_one(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="test_size"):
        load_config(write(tmp_path, VALID.replace("test_size = 0.25", f"test_size = {bad}")))


@pytest.mark.parametrize(
    ("setting", "bad"),
    [("cv_folds", "1"), ("cv_folds", "2.5"), ("cv_repeats", "0"), ("cv_repeats", "true")],
)
def test_load_config_rejects_invalid_cv_counts(tmp_path: Path, setting: str, bad: str) -> None:
    text = VALID.replace(f"{setting} = {4 if setting == 'cv_folds' else 2}", f"{setting} = {bad}")
    with pytest.raises(ValueError, match=setting):
        load_config(write(tmp_path, text))
