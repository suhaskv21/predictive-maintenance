import polars as pl
import pytest

from predictive_maintenance.data import schema
from predictive_maintenance.training.examples import select_examples

SEED = 7
COLUMNS = [*schema.FEATURE_COLUMNS, schema.TARGET]


def train_partition(n: int = 60, failures: int = 8) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "UDI": list(range(n)),
            schema.CATEGORICAL_FEATURE: ["L", "M", "H"] * (n // 3),
            "Air temperature [K]": [290.0 + i / 10 for i in range(n)],
            "Process temperature [K]": [300.0 + i / 10 for i in range(n)],
            "Rotational speed [rpm]": [1400 + i for i in range(n)],
            "Torque [Nm]": [30.0 + i / 5 for i in range(n)],
            "Tool wear [min]": list(range(n)),
            schema.TARGET: [1] * failures + [0] * (n - failures),
            **{mode: [0] * n for mode in schema.FAILURE_MODES},
        }
    )


def test_the_draw_has_the_requested_failures_and_the_rest_non_failures() -> None:
    chosen = select_examples(train_partition(), count=10, failures=3, seed=SEED)
    assert chosen.height == 10
    assert chosen[schema.TARGET].sum() == 3


def test_every_example_is_a_train_partition_reading() -> None:
    train = train_partition()
    chosen = select_examples(train, count=10, failures=3, seed=SEED)
    assert set(chosen["Tool wear [min]"]) <= set(train["Tool wear [min]"])
    joined = chosen.join(train.select([*COLUMNS]), on=COLUMNS, how="inner")
    assert joined.height == chosen.height


def test_an_example_carries_the_six_inputs_and_only_the_machine_failure_outcome() -> None:
    chosen = select_examples(train_partition(), count=10, failures=3, seed=SEED)
    assert chosen.columns == COLUMNS


def test_the_same_seed_gives_the_same_draw_and_another_seed_a_different_one() -> None:
    train = train_partition()
    first = select_examples(train, count=10, failures=3, seed=SEED)
    assert first.equals(select_examples(train, count=10, failures=3, seed=SEED))
    assert not first.equals(select_examples(train, count=10, failures=3, seed=SEED + 1))


def test_the_order_does_not_group_failures_first() -> None:
    chosen = select_examples(train_partition(), count=10, failures=3, seed=SEED)
    outcomes = chosen[schema.TARGET].to_list()
    assert outcomes != sorted(outcomes) and outcomes != sorted(outcomes, reverse=True)


def test_too_few_failures_in_the_train_partition_fails_loudly() -> None:
    with pytest.raises(ValueError, match="3 failures"):
        select_examples(train_partition(failures=2), count=10, failures=3, seed=SEED)


def test_too_few_non_failures_in_the_train_partition_fails_loudly() -> None:
    with pytest.raises(ValueError, match="non-failures"):
        select_examples(train_partition(n=9, failures=3), count=10, failures=3, seed=SEED)
