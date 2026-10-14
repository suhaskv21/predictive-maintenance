"""Choose the Example readings the app offers (ADR 0009, ADR 0012).

The draw takes the train partition only, so no Holdout Reading can be selected, and uses the seed
from the config. It is per outcome class: a plain draw at the data's failure prevalence would give
no failure at all. Only the six inputs and the ``Machine failure`` outcome are kept; the failure
mode columns never leave this function.
"""

import polars as pl

from predictive_maintenance.data import schema


def _draw(partition: pl.DataFrame, n: int, seed: int, what: str) -> pl.DataFrame:
    if partition.height < n:
        raise ValueError(
            f"the train partition has {partition.height} {what}, fewer than the {n} "
            f"{what} the Example readings need"
        )
    return partition.sample(n=n, seed=seed)


def select_examples(train: pl.DataFrame, *, count: int, failures: int, seed: int) -> pl.DataFrame:
    """``count`` Readings from ``train``, ``failures`` of them recorded failures, in mixed order.

    Returns the six inputs and ``Machine failure``. The order is shuffled so that a Reading's
    position does not tell its outcome.

    Raises:
        ValueError: if ``train`` has fewer failures than ``failures`` or fewer non-failures than
            ``count - failures``.
    """
    columns = [*schema.FEATURE_COLUMNS, schema.TARGET]
    failed = train.filter(pl.col(schema.TARGET) == 1)
    intact = train.filter(pl.col(schema.TARGET) == 0)
    chosen = pl.concat(
        [
            _draw(failed, failures, seed, "failures"),
            _draw(intact, count - failures, seed, "non-failures"),
        ]
    )
    return chosen.select(columns).sample(fraction=1.0, shuffle=True, seed=seed)
