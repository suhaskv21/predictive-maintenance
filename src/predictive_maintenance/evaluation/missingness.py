"""Missing-data policy analysis (ADR 0008): inject absent inputs, score, judge, choose a strategy.

Design, fixed before any number:

- **Complete-trained.** Each model is fitted once per CV fold on complete training Readings, so the
  ADR 0006/0007 models and thresholds do not move. A strategy only decides how absent inputs are
  filled at scoring time (``preprocessing.imputation``).
- **Absence is injected into validation rows only**, on 100 % of them, for every one of the 63
  non-empty *patterns* of absent inputs (``all_patterns``). The app scores one Reading at a time,
  so the question is "given these inputs are absent, how good is the Flag?". A random MCAR rate
  would dilute exactly that. Nothing here is random: the folds and the models carry the seed.
- **The Operating threshold is frozen** (ADR 0007) and applied unchanged to every pattern.
- A pattern is *scorable* for a strategy only if, paired with the complete Readings on identical
  folds, it keeps the retained saving, recall and Flag rate that ``MissingDataConfig`` demands,
  and every pattern with one input fewer absent is scorable too (monotonic closure).
- **Stress.** Torque absent only on validation Readings whose speed is at or above the training
  fold's ``stress_speed_quantile``, judged on those rows against the same rows complete. It can
  only veto the ``Torque`` pattern, never allow it.

Vocabulary follows ``GLOSSARY.md``. Nothing here sees the holdout: every function takes the train
partition and folds indexing it.
"""

import json
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score

from predictive_maintenance.config import MissingDataConfig
from predictive_maintenance.data import schema
from predictive_maintenance.evaluation.compare import Fold, SelectionStep, corrected_se
from predictive_maintenance.evaluation.threshold import (
    expected_cost,
    operating_metrics,
    pick_threshold,
)
from predictive_maintenance.preprocessing.imputation import (
    IMPUTING_STRATEGIES,
    STRATEGIES,
    InputImputer,
)
from predictive_maintenance.training.candidates import Candidate

Pattern = tuple[str, ...]

INPUTS = tuple(schema.FEATURE_COLUMNS)
TORQUE = "Torque [Nm]"
SPEED = "Rotational speed [rpm]"
TORQUE_PATTERN: Pattern = (TORQUE,)
SINGLE_SENSOR_PATTERNS: tuple[Pattern, ...] = tuple((c,) for c in schema.NUMERIC_FEATURES)
COMPLETE = "complete"
ALL_ROWS = "all"
STRESS = "stress"

RESULT_SCHEMA: dict[str, Any] = {
    "strategy": pl.String,
    "pattern": pl.String,
    "n_absent": pl.Int64,
    "scope": pl.String,
    "fold": pl.Int64,
    "rows": pl.Int64,
    "failures": pl.Int64,
    "recall": pl.Float64,
    "alerts_per_1000": pl.Float64,
    "cost_per_1000": pl.Float64,
    "saving": pl.Float64,
    "pr_auc": pl.Float64,
}


def all_patterns() -> list[Pattern]:
    """Every non-empty set of absent inputs (63), smallest first, inputs in canonical order."""
    return [c for size in range(1, len(INPUTS) + 1) for c in combinations(INPUTS, size)]


def pattern_name(pattern: Pattern) -> str:
    """Readable key for a pattern: its inputs joined by ``+``, or ``complete`` for none."""
    return "+".join(pattern) if pattern else COMPLETE


def inject_absence(
    x: pl.DataFrame, pattern: Pattern, rows: NDArray[np.bool_] | None = None
) -> pl.DataFrame:
    """Set the inputs of ``pattern`` to null, on all rows or only where ``rows`` is true."""
    unknown = set(pattern) - set(INPUTS)
    if unknown:
        raise ValueError(f"unknown inputs {sorted(unknown)}; known: {INPUTS}")
    if rows is None:
        return x.with_columns(pl.lit(None, dtype=x[c].dtype).alias(c) for c in pattern)
    if rows.shape != (x.height,):
        raise ValueError(f"rows must have shape ({x.height},), got {rows.shape}")
    mask = pl.Series(rows)
    return x.with_columns(
        pl.when(mask).then(pl.lit(None, dtype=x[c].dtype)).otherwise(pl.col(c)).alias(c)
        for c in pattern
    )


def _row(
    strategy: str,
    pattern: Pattern,
    scope: str,
    fold: int,
    y: NDArray[np.int_],
    scores: NDArray[np.float64],
    threshold: float,
    cost_ratio: float,
) -> tuple[Any, ...]:
    metrics = operating_metrics(y, scores, threshold, cost_ratio)
    pr_auc = float(average_precision_score(y, scores)) if y.any() else math.nan
    return (
        strategy,
        pattern_name(pattern),
        len(pattern),
        scope,
        fold,
        len(y),
        int(y.sum()),
        metrics.recall,
        metrics.alerts_per_1000,
        metrics.cost_per_1000,
        metrics.saving_vs_flag_nothing,
        pr_auc,
    )


def _evaluate_fold(
    candidate: Candidate,
    x: pl.DataFrame,
    y: NDArray[np.int_],
    fold: int,
    train_idx: NDArray[np.intp],
    test_idx: NDArray[np.intp],
    threshold: float,
    cost_ratio: float,
    strategies: Sequence[str],
    patterns: Sequence[Pattern],
    stress_quantile: float,
) -> list[tuple[Any, ...]]:
    # One model per fold: ``native`` is the tolerant-Type pipeline, whose fit on complete rows is
    # identical to the plain one. The imputer is swapped per strategy, so models are never refitted.
    fitted = candidate.make_pipeline(missing_strategy="native").fit(x[train_idx], y[train_idx])
    core = fitted[1:]
    imputers = {s: InputImputer(s).fit(x[train_idx]) for s in strategies}
    identity = InputImputer("native").fit(x[train_idx])
    x_val, y_val = x[test_idx], y[test_idx]

    def score(imputer: InputImputer, frame: pl.DataFrame) -> NDArray[np.float64]:
        probabilities: NDArray[np.float64] = core.predict_proba(imputer.transform(frame))[:, 1]
        return probabilities

    complete = score(identity, x_val)
    cutoff = float(np.quantile(x[train_idx][SPEED].to_numpy(), stress_quantile))
    high_speed = x_val[SPEED].to_numpy() >= cutoff
    rows = [
        _row(COMPLETE, (), ALL_ROWS, fold, y_val, complete, threshold, cost_ratio),
        _row(
            COMPLETE,
            (),
            STRESS,
            fold,
            y_val[high_speed],
            complete[high_speed],
            threshold,
            cost_ratio,
        ),
    ]
    for strategy, imputer in imputers.items():
        for pattern in patterns:
            scores = score(imputer, inject_absence(x_val, pattern))
            rows.append(
                _row(strategy, pattern, ALL_ROWS, fold, y_val, scores, threshold, cost_ratio)
            )
        stressed = score(imputer, inject_absence(x_val[np.flatnonzero(high_speed)], TORQUE_PATTERN))
        rows.append(
            _row(
                strategy,
                TORQUE_PATTERN,
                STRESS,
                fold,
                y_val[high_speed],
                stressed,
                threshold,
                cost_ratio,
            )
        )
    return rows


def evaluate_patterns(
    candidate: Candidate,
    train: pl.DataFrame,
    folds: Sequence[Fold],
    *,
    threshold: float,
    cost_ratio: float,
    stress_quantile: float,
    strategies: Sequence[str] = IMPUTING_STRATEGIES,
    patterns: Sequence[Pattern] | None = None,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """Operating-point metrics for every strategy and pattern on every fold (long format).

    Rows with ``strategy == "complete"`` are the complete-Reading baseline of the same fitted model
    (scope ``all`` on all validation rows, scope ``stress`` on the high-speed rows). Columns follow
    ``RESULT_SCHEMA``. ``threshold`` is the frozen Operating threshold of this model.

    Raises:
        ValueError: if ``folds`` is empty or out of range, or a strategy is not an imputing one.
    """
    if not folds or max(int(max(a.max(), b.max())) for a, b in folds) >= train.height:
        raise ValueError("folds must be non-empty and index rows inside the train partition")
    bad = set(strategies) - set(IMPUTING_STRATEGIES)
    if bad:
        raise ValueError(
            f"strategies must be imputing ones {IMPUTING_STRATEGIES}, got {sorted(bad)}"
        )
    chosen = list(all_patterns() if patterns is None else patterns)
    x = train.select(schema.FEATURE_COLUMNS)
    y = train[schema.TARGET].to_numpy()
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_fold)(
            candidate,
            x,
            y,
            i,
            a,
            b,
            threshold,
            cost_ratio,
            strategies,
            chosen,
            stress_quantile,
        )
        for i, (a, b) in enumerate(folds)
    )
    return pl.DataFrame(
        [row for fold_rows in results for row in fold_rows], schema=RESULT_SCHEMA, orient="row"
    )


def _paired(results: pl.DataFrame, scope: str) -> pl.DataFrame:
    """Each strategy row joined to the complete baseline of the same fold and scope."""
    baseline = results.filter((pl.col("strategy") == COMPLETE) & (pl.col("scope") == scope)).select(
        "fold",
        pl.col("recall").alias("recall_c"),
        pl.col("alerts_per_1000").alias("alerts_c"),
        pl.col("cost_per_1000").alias("cost_c"),
        pl.col("saving").alias("saving_c"),
        pl.col("pr_auc").alias("pr_auc_c"),
    )
    return (
        results.filter((pl.col("strategy") != COMPLETE) & (pl.col("scope") == scope))
        .join(baseline, on="fold")
        .with_columns(
            retained=pl.col("saving") / pl.col("saving_c"),
            recall_drop=pl.col("recall_c") - pl.col("recall"),
        )
        .sort("strategy", "pattern", "fold")
    )


def _judge(group: pl.DataFrame, cfg: MissingDataConfig, n_cv_folds: int) -> dict[str, Any]:
    retained = group["retained"].drop_nans().drop_nulls().to_numpy()
    recall_drop = group["recall_drop"].drop_nans().drop_nulls().to_numpy()
    usable = len(retained)
    mean_retained = float(retained.mean()) if usable else math.nan
    se = corrected_se(retained, n_cv_folds=n_cv_folds) if usable > 1 else math.nan
    mean_drop = float(recall_drop.mean()) if len(recall_drop) else math.nan
    alerts_c = float(group["alerts_c"].mean())  # type: ignore[arg-type]
    flag_ratio = float(group["alerts_per_1000"].mean()) / alerts_c if alerts_c else math.nan  # type: ignore[arg-type]
    passes_saving = usable > 1 and mean_retained - se >= cfg.min_retained_saving
    passes_recall = len(recall_drop) > 0 and mean_drop <= cfg.max_recall_drop
    passes_flag = cfg.flag_rate_ratio_min <= flag_ratio <= cfg.flag_rate_ratio_max
    return {
        "usable_folds": usable,
        "retained_saving": mean_retained,
        "retained_se": se,
        "recall_drop": mean_drop,
        "flag_rate_ratio": flag_ratio,
        "pr_auc_relative_drop": 1 - float(group["pr_auc"].mean()) / float(group["pr_auc_c"].mean()),  # type: ignore[arg-type]
        "cost_per_1000": float(group["cost_per_1000"].mean()),  # type: ignore[arg-type]
        "passes_saving": passes_saving,
        "passes_recall": passes_recall,
        "passes_flag": passes_flag,
        "passes": bool(passes_saving and passes_recall and passes_flag),
    }


VERDICT_SCHEMA: dict[str, Any] = {
    "strategy": pl.String,
    "pattern": pl.String,
    "n_absent": pl.Int64,
    "usable_folds": pl.Int64,
    "retained_saving": pl.Float64,
    "retained_se": pl.Float64,
    "recall_drop": pl.Float64,
    "flag_rate_ratio": pl.Float64,
    "pr_auc_relative_drop": pl.Float64,
    "cost_per_1000": pl.Float64,
    "passes_saving": pl.Boolean,
    "passes_recall": pl.Boolean,
    "passes_flag": pl.Boolean,
    "passes": pl.Boolean,
    "stress_ok": pl.Boolean,
    "scorable": pl.Boolean,
}


def pattern_verdicts(
    results: pl.DataFrame, cfg: MissingDataConfig, *, n_cv_folds: int
) -> pl.DataFrame:
    """Judge every strategy and pattern of one model; apply the stress veto and the closure.

    ``passes`` is the three tests (retained saving minus its Nadeau-Bengio corrected SE, recall
    drop, Flag-rate ratio). ``stress_ok`` is the same tests on the torque-absent high-speed rows,
    and only matters for the ``Torque`` pattern. ``scorable`` additionally requires every pattern
    with one input fewer absent to be scorable (smaller patterns are decided first). The ``Torque``
    stress also vetoes through the closure every pattern that contains torque.
    """
    stress_pairs = _paired(results, STRESS)
    stress = {
        strategy: bool(_judge(group, cfg, n_cv_folds)["passes"])
        for (strategy,), group in stress_pairs.group_by(["strategy"])
    }
    rows = []
    for key, group in _paired(results, ALL_ROWS).group_by(["strategy", "pattern", "n_absent"]):
        strategy, pattern, n_absent = key
        verdict = _judge(group, cfg, n_cv_folds)
        stress_ok = stress.get(strategy, True) if pattern == pattern_name(TORQUE_PATTERN) else True
        rows.append(
            {
                "strategy": strategy,
                "pattern": pattern,
                "n_absent": n_absent,
                **verdict,
                "stress_ok": stress_ok,
            }
        )
    table = pl.DataFrame(rows, schema={k: v for k, v in VERDICT_SCHEMA.items() if k != "scorable"})
    scorable: dict[tuple[str, str], bool] = {}
    for record in table.sort("n_absent", "pattern").to_dicts():
        inputs = tuple(record["pattern"].split("+"))
        smaller = [
            scorable.get((record["strategy"], pattern_name(sub)), False)
            for sub in combinations(inputs, len(inputs) - 1)
            if sub
        ]
        scorable[(record["strategy"], record["pattern"])] = bool(
            record["passes"] and record["stress_ok"] and all(smaller)
        )
    return table.with_columns(
        pl.struct("strategy", "pattern")
        .map_elements(lambda s: scorable[(s["strategy"], s["pattern"])], return_dtype=pl.Boolean)
        .alias("scorable")
    ).sort("strategy", "n_absent", "pattern")


def single_sensor_retained(results: pl.DataFrame, strategy: str) -> NDArray[np.float64]:
    """Per-fold retained saving of ``strategy``, averaged over the five single-sensor patterns."""
    names = [pattern_name(p) for p in SINGLE_SENSOR_PATTERNS]
    paired = _paired(results, ALL_ROWS).filter(
        (pl.col("strategy") == strategy) & pl.col("pattern").is_in(names)
    )
    if paired.is_empty():
        raise ValueError(f"no single-sensor results for strategy {strategy!r}")
    return (
        paired.group_by("fold").agg(pl.col("retained").mean()).sort("fold")["retained"].to_numpy()
    )


def choose_strategy(
    results: pl.DataFrame,
    verdicts: pl.DataFrame,
    cfg: MissingDataConfig,
    *,
    n_cv_folds: int,
) -> tuple[str, tuple[SelectionStep, ...]]:
    """Simplest-first walk over ``STRATEGIES`` for one model (the ADR 0006 rule, ADR 0008 margins).

    ``reject`` is the incumbent and retains no saving. A challenger replaces the incumbent only if
    it makes at least one pattern scorable and its mean paired per-fold gain in retained saving
    (averaged over the five single-sensor patterns) exceeds
    ``max(challenger_margin, corrected SE)``.
    """
    incumbent, steps = STRATEGIES[0], []
    for challenger in STRATEGIES[1:]:
        gain = single_sensor_retained(results, challenger)
        base = (
            np.zeros_like(gain)
            if incumbent == STRATEGIES[0]
            else single_sensor_retained(results, incumbent)
        )
        deltas = gain - base
        se = corrected_se(deltas, n_cv_folds=n_cv_folds)
        threshold = max(cfg.challenger_margin, se)
        has_scorable = bool(
            verdicts.filter((pl.col("strategy") == challenger) & pl.col("scorable")).height
        )
        replaced = has_scorable and float(deltas.mean()) > threshold
        steps.append(
            SelectionStep(challenger, incumbent, float(deltas.mean()), se, threshold, replaced)
        )
        if replaced:
            incumbent = challenger
    return incumbent, tuple(steps)


def scorable_patterns(verdicts: pl.DataFrame, strategy: str) -> list[Pattern]:
    """The patterns the policy scores for ``strategy`` (none for ``reject``)."""
    if strategy == "reject":
        return []
    rows = verdicts.filter((pl.col("strategy") == strategy) & pl.col("scorable"))
    return [tuple(p.split("+")) for p in rows["pattern"]]


def refused_share(rate: float, scorable: Collection[Pattern]) -> float:
    """Share of Readings the app refuses if each input is independently absent with ``rate``.

    Sums the probability of every pattern that is not scorable; the complete Reading is always
    scored. Independence (MCAR) is an assumption: the raw data has no nulls to measure it from.
    """
    allowed = {frozenset(p) for p in scorable}
    n = len(INPUTS)
    return sum(
        rate**k * (1 - rate) ** (n - k)
        for k in range(1, n + 1)
        for pattern in combinations(INPUTS, k)
        if frozenset(pattern) not in allowed
    )


def incomplete_share(rate: float) -> float:
    """Share of Readings with at least one input absent: what a complete-case training set loses."""
    return 1 - (1 - rate) ** len(INPUTS)


@dataclass(frozen=True)
class InformativeLoss:
    """What losing every Reading in the high-speed stress group takes from a training set."""

    rows_lost: float
    failures_lost: float
    failure_rate_lost: float
    failure_rate_overall: float


def informative_loss(train: pl.DataFrame, speed_quantile: float) -> InformativeLoss:
    """Share of Readings and of failures in the group where the torque sensor drops out.

    The group is the Readings at or above the ``speed_quantile`` of rotational speed. If missing
    torque is informative (it fails at high speed), dropping incomplete Readings removes failures
    in a different proportion than other Readings.
    """
    speed = train[SPEED].to_numpy()
    y = train[schema.TARGET].to_numpy().astype(bool)
    lost = speed >= np.quantile(speed, speed_quantile)
    return InformativeLoss(
        rows_lost=float(lost.mean()),
        failures_lost=float((lost & y).sum() / y.sum()),
        failure_rate_lost=float(y[lost].mean()),
        failure_rate_overall=float(y.mean()),
    )


def _score_patterns_fold(
    candidate: Candidate,
    x: pl.DataFrame,
    y: NDArray[np.int_],
    train_idx: NDArray[np.intp],
    test_idx: NDArray[np.intp],
    strategy: str,
    patterns: Sequence[Pattern],
) -> list[NDArray[np.float64]]:
    fitted = candidate.make_pipeline(missing_strategy="native").fit(x[train_idx], y[train_idx])
    core = fitted[1:]
    imputer = InputImputer(strategy).fit(x[train_idx])
    x_val = x[test_idx]
    out: list[NDArray[np.float64]] = []
    for pattern in patterns:
        scores: NDArray[np.float64] = core.predict_proba(
            imputer.transform(inject_absence(x_val, pattern))
        )[:, 1]
        out.append(scores)
    return out


def pattern_repeat_scores(
    candidate: Candidate,
    train: pl.DataFrame,
    folds: Sequence[Fold],
    *,
    n_splits: int,
    strategy: str,
    patterns: Sequence[Pattern],
    n_jobs: int = -1,
) -> dict[Pattern, NDArray[np.float64]]:
    """Pooled out-of-fold scores per CV repeat with ``pattern`` absent, shape ``(repeats, rows)``.

    Same layout as ``threshold.repeat_scores``, with absence injected into the validation rows only.

    Raises:
        ValueError: if ``folds`` is not a whole number of repeats of ``n_splits``.
    """
    if not folds or len(folds) % n_splits:
        raise ValueError(f"{len(folds)} folds are not a whole number of repeats of {n_splits}")
    x = train.select(schema.FEATURE_COLUMNS)
    y = train[schema.TARGET].to_numpy()
    results = Parallel(n_jobs=n_jobs)(
        delayed(_score_patterns_fold)(candidate, x, y, a, b, strategy, patterns) for a, b in folds
    )
    pooled = {p: np.full((len(folds) // n_splits, train.height), np.nan) for p in patterns}
    for i, ((_, b), fold_scores) in enumerate(zip(folds, results, strict=True)):
        for pattern, scores in zip(patterns, fold_scores, strict=True):
            pooled[pattern][i // n_splits, b] = scores
    return pooled


@dataclass(frozen=True)
class RecoverableCost:
    """Cost at the frozen threshold against a threshold re-picked on the injected scores."""

    frozen_cost: float
    repicked_threshold: float
    repicked_cost: float

    @property
    def recoverable(self) -> float:
        """Cost per Reading a pattern-specific threshold would recover (optimistic upper bound)."""
        return self.frozen_cost - self.repicked_cost


def recoverable_cost(
    y: NDArray[np.int_], matrix: NDArray[np.float64], *, frozen: float, cost_ratio: float
) -> RecoverableCost:
    """Mean over repeats of the expected cost at ``frozen`` and at the per-repeat best threshold.

    The re-picked threshold is chosen and scored on the same scores, so the recoverable cost is an
    optimistic upper bound. It is reported to size the question and is never shipped (ADR 0008).
    """
    labels = y.astype(bool)
    picks = [pick_threshold(labels, row, cost_ratio) for row in matrix]
    frozen_cost = float(
        np.mean([expected_cost(labels, row >= frozen, cost_ratio) for row in matrix])
    )
    repicked_cost = float(
        np.mean(
            [
                expected_cost(labels, row >= t, cost_ratio)
                for row, t in zip(matrix, picks, strict=True)
            ]
        )
    )
    return RecoverableCost(frozen_cost, float(np.median(picks)), repicked_cost)


@dataclass(frozen=True)
class PatternDegradation:
    """Measured cost of scoring with one pattern of inputs absent, on identical CV folds."""

    pattern: Pattern
    retained_saving: float
    retained_se: float
    recall_drop: float
    flag_rate_ratio: float


@dataclass(frozen=True)
class MissingDataPolicy:
    """What the app is told about Incomplete readings for one model (ADR 0008).

    Stored as ``<model>.missing.json`` next to the model's Pipeline artifact, separate from
    ``OperatingPoint``. ``scorable`` is empty for the ``reject`` strategy: every Incomplete reading
    is then a Refused reading. Any pattern not listed is refused.

    ``measured`` is a reference, never used to score: the patterns that ``measured_strategy`` was
    measured to score acceptably but that were not adopted. The pipeline-health monitor uses it to
    tell a *recoverable* refusal from one no imputation would fix.
    """

    model: str
    strategy: str
    cost_ratio: float
    scorable: tuple[PatternDegradation, ...]
    measured_strategy: str | None = None
    measured: tuple[PatternDegradation, ...] = ()

    def __post_init__(self) -> None:
        """Reject an unknown strategy, scorable patterns under ``reject``, an invalid reference."""
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if self.strategy == STRATEGIES[0] and self.scorable:
            raise ValueError("the reject strategy scores no Incomplete reading")
        if self.measured_strategy is not None and self.measured_strategy not in IMPUTING_STRATEGIES:
            raise ValueError(
                f"measured_strategy must be an imputing strategy, got {self.measured_strategy!r}"
            )
        if self.measured and self.measured_strategy is None:
            raise ValueError("measured patterns need a measured_strategy")

    def degradation(self, absent: Collection[str]) -> PatternDegradation | None:
        """The measured degradation if this set of absent inputs is scorable, else None."""
        wanted = frozenset(absent)
        return next((d for d in self.scorable if frozenset(d.pattern) == wanted), None)

    def recoverable(self, absent: Collection[str]) -> PatternDegradation | None:
        """The reference degradation if this refused pattern would score under ``measured``."""
        wanted = frozenset(absent)
        if not wanted or self.is_scorable(wanted):
            return None
        return next((d for d in self.measured if frozenset(d.pattern) == wanted), None)

    def is_scorable(self, absent: Collection[str]) -> bool:
        """Whether a Reading with exactly these inputs absent is scored rather than refused."""
        return not absent or self.degradation(absent) is not None

    def to_json(self) -> str:
        """Serialise to the sidecar JSON."""
        return json.dumps(
            {
                "model": self.model,
                "strategy": self.strategy,
                "cost_ratio": self.cost_ratio,
                "scorable": [_degradation_json(d) for d in self.scorable],
                "measured_strategy": self.measured_strategy,
                "measured": [_degradation_json(d) for d in self.measured],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "MissingDataPolicy":
        """Inverse of ``to_json``."""
        raw = json.loads(text)
        return cls(
            model=raw["model"],
            strategy=raw["strategy"],
            cost_ratio=float(raw["cost_ratio"]),
            scorable=tuple(_degradation_from_json(d) for d in raw["scorable"]),
            measured_strategy=raw.get("measured_strategy"),
            measured=tuple(_degradation_from_json(d) for d in raw.get("measured", [])),
        )


def _degradation_json(d: PatternDegradation) -> dict[str, Any]:
    return {
        "pattern": list(d.pattern),
        "retained_saving": d.retained_saving,
        "retained_se": d.retained_se,
        "recall_drop": d.recall_drop,
        "flag_rate_ratio": d.flag_rate_ratio,
    }


def _degradation_from_json(d: dict[str, Any]) -> PatternDegradation:
    return PatternDegradation(
        pattern=tuple(d["pattern"]),
        retained_saving=float(d["retained_saving"]),
        retained_se=float(d["retained_se"]),
        recall_drop=float(d["recall_drop"]),
        flag_rate_ratio=float(d["flag_rate_ratio"]),
    )


def _degradations(verdicts: pl.DataFrame, strategy: str) -> tuple[PatternDegradation, ...]:
    rows = (
        verdicts.filter((pl.col("strategy") == strategy) & pl.col("scorable"))
        .sort("n_absent", "pattern")
        .to_dicts()
    )
    return tuple(
        PatternDegradation(
            pattern=tuple(r["pattern"].split("+")),
            retained_saving=r["retained_saving"],
            retained_se=r["retained_se"],
            recall_drop=r["recall_drop"],
            flag_rate_ratio=r["flag_rate_ratio"],
        )
        for r in rows
    )


def build_policy(
    model: str,
    strategy: str,
    verdicts: pl.DataFrame,
    *,
    cost_ratio: float,
    measured_strategy: str | None = None,
) -> MissingDataPolicy:
    """The sidecar for ``model`` from its verdict table and the adopted ``strategy``.

    ``measured_strategy`` (an imputing one) adds its scorable patterns as the non-scoring reference
    that the pipeline-health monitor uses.
    """
    return MissingDataPolicy(
        model=model,
        strategy=strategy,
        cost_ratio=cost_ratio,
        scorable=() if strategy == STRATEGIES[0] else _degradations(verdicts, strategy),
        measured_strategy=measured_strategy,
        measured=() if measured_strategy is None else _degradations(verdicts, measured_strategy),
    )
