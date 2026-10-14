"""Score the Holdout once and check the committed report against a Pipeline (ADR 0007, ADR 0012).

``just holdout`` runs ``main``: it scores the Holdout for the default model with the frozen
Operating threshold of its unit and writes ``reports/holdout-metrics.json``. The report holds the
Pipeline's SHA-256, the date and each ADR 0007 rule 9 metric with a percentile 95 % interval from a
stratified bootstrap (resamples from ``configs/``, seed stream 11 of the config seed), in which the
threshold is never re-chosen.

The Holdout is for one look only. If a report exists the command refuses to rescore unless
``--rescore`` is passed, and a rescore needs a new ADR first. Builds never rescore: they call
``check_report``, which fails if the report's hash is not the Pipeline's.

If builds turn out not to be bit-identical, replace the hash comparison in ``check_report`` with a
fingerprint (pinned library versions, and scores on the fixed train Readings of the manifest equal
within 1e-9). That is a fallback for the check only, never a reason to rescore.

``apply_report`` puts the report's figures into the unit's operating-metrics, labelled ``holdout``
and with their intervals, and keeps the CV figures only under ``sensitivity``. Builds call it after
training, because training writes the CV figures afresh each time.
"""

import argparse
import json
import math
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import polars as pl

from predictive_maintenance.config import PROJECT_ROOT, Config, load_config
from predictive_maintenance.data import schema
from predictive_maintenance.data.ingest import load_dataset
from predictive_maintenance.evaluation.compare import derived_seeds, score_readings
from predictive_maintenance.evaluation.split import stratified_holdout
from predictive_maintenance.evaluation.threshold import (
    BOOTSTRAP_STREAM,
    CI_LEVEL,
    OperatingPoint,
    bootstrap_metric_cis,
)
from predictive_maintenance.training.train import DEFAULT_MODEL, MODELS_DIR
from predictive_maintenance.training.unit import file_sha256, ratio_key, unit_paths

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_NAME = "holdout-metrics.json"
HOLDOUT_LABEL = "holdout"
# The figures the Flag meaning quotes, as named in the report and in the operating-metrics file.
QUOTED_FIGURES = ("recall", "precision", "flags_per_1000", "false_alarms_per_caught")


class HoldoutAlreadyScoredError(RuntimeError):
    """The Holdout report exists, so scoring it again would be a second look."""


class HoldoutMismatchError(RuntimeError):
    """The committed Holdout report was written for a different Pipeline."""


def _none_if_nan(value: float) -> float | None:
    return None if math.isnan(value) else value


def score_holdout(
    readings: pl.DataFrame,
    config: Config,
    models_dir: Path,
    reports_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    rescore: bool = False,
) -> Path:
    """Score the Holdout of ``readings`` for the unit of ``model`` and write the report.

    The threshold and Cost ratio are those frozen in the unit's OperatingPoint. Returns the path of
    ``holdout-metrics.json`` under ``reports_dir``.

    Raises:
        HoldoutAlreadyScoredError: if the report exists and ``rescore`` is False. A rescore needs
            a new ADR.
    """
    report = reports_dir / REPORT_NAME
    if report.exists() and not rescore:
        raise HoldoutAlreadyScoredError(
            f"{report} exists: the Holdout has been scored. Rescoring needs a new ADR, then "
            "--rescore."
        )
    unit = unit_paths(models_dir, model)
    point = OperatingPoint.from_json(unit.operating_point.read_text())
    _, holdout = stratified_holdout(readings, test_size=config.split.test_size, seed=config.seed)
    scores = score_readings(joblib.load(unit.pipeline), holdout.select(schema.FEATURE_COLUMNS))
    resamples = config.evaluation.bootstrap_resamples
    intervals = bootstrap_metric_cis(
        holdout[schema.TARGET].to_numpy(),
        scores,
        threshold=point.threshold,
        cost_ratio=point.default_cost_ratio,
        n_resamples=resamples,
        seed=derived_seeds(config.seed, BOOTSTRAP_STREAM, 1)[0],
    )
    content: dict[str, Any] = {
        "model": model,
        "pipeline_sha256": file_sha256(unit.pipeline),
        "date": date.today().isoformat(),
        "threshold": point.threshold,
        "cost_ratio": point.default_cost_ratio,
        "holdout_readings": holdout.height,
        "holdout_failures": int(holdout[schema.TARGET].sum()),
        "resamples": resamples,
        "interval_level": CI_LEVEL,
        "seed_stream": BOOTSTRAP_STREAM,
        "metrics": {
            row["metric"]: {
                "estimate": _none_if_nan(row["estimate"]),
                "low": _none_if_nan(row["low"]),
                "high": _none_if_nan(row["high"]),
            }
            for row in intervals.iter_rows(named=True)
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(content, indent=2) + "\n")
    return report


def check_report(report: Path, pipeline: Path) -> None:
    """Fail unless ``report`` was written for the Pipeline file at ``pipeline``.

    Raises:
        FileNotFoundError: if the report does not exist.
        HoldoutMismatchError: if the report's SHA-256 is not the Pipeline's.
    """
    recorded = json.loads(report.read_text())["pipeline_sha256"]
    if recorded != file_sha256(pipeline):
        raise HoldoutMismatchError("holdout figures belong to a different model")


def apply_report(report: Path, models_dir: Path, *, model: str = DEFAULT_MODEL) -> None:
    """Make the unit's operating-metrics quote the Holdout figures of ``report``.

    The file then has ``source`` ``holdout``, the figures of the default Cost ratio with their
    intervals, and the training-time CV figures under ``sensitivity`` (labelled ``cv_train``). The
    manifest's hash of the file is updated; the Pipeline is not touched. Nothing is rescored, and
    applying twice changes nothing.

    Raises:
        HoldoutMismatchError: if the report was not written for the unit's Pipeline; no file is
            changed.
    """
    unit = unit_paths(models_dir, model)
    check_report(report, unit.pipeline)
    metrics = json.loads(unit.operating_metrics.read_text())
    if metrics["source"] == HOLDOUT_LABEL:
        return
    content = json.loads(report.read_text())
    figures = content["metrics"]
    applied = {
        "model": metrics["model"],
        "source": HOLDOUT_LABEL,
        "ratios": {
            ratio_key(content["cost_ratio"]): {
                **{name: figures[name]["estimate"] for name in QUOTED_FIGURES},
                "intervals": {
                    name: {"low": figures[name]["low"], "high": figures[name]["high"]}
                    for name in QUOTED_FIGURES
                },
            }
        },
        "holdout": {
            key: content[key]
            for key in ("date", "holdout_readings", "holdout_failures", "resamples")
        }
        | {"interval_level": content["interval_level"]},
        "sensitivity": {"source": metrics["source"], "ratios": metrics["ratios"]},
        "training_ranges": metrics["training_ranges"],
        "training_types": metrics["training_types"],
    }
    unit.operating_metrics.write_text(json.dumps(applied, indent=2) + "\n")
    manifest = json.loads(unit.manifest.read_text())
    manifest["files"][unit.operating_metrics.name] = file_sha256(unit.operating_metrics)
    unit.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    """Score the Holdout for the default model in ``models/`` and write it under ``reports/``.

    With ``--check`` it only compares the committed report with the Pipeline (for builds), and
    with ``--apply`` it also puts the report's figures into the unit's operating-metrics.
    """
    parser = argparse.ArgumentParser(description="Score the Holdout once (ADR 0012).")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="score again although a report exists; needs a new ADR first",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only check that the committed report belongs to the Pipeline in models/",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="check the report, then label the unit's operating-metrics with its figures",
    )
    args = parser.parse_args(argv)
    if args.check or args.apply:
        try:
            check_report(REPORTS_DIR / REPORT_NAME, unit_paths(MODELS_DIR, DEFAULT_MODEL).pipeline)
        except (HoldoutMismatchError, FileNotFoundError) as error:
            raise SystemExit(str(error)) from error
        print("holdout report matches the Pipeline")
        if args.apply:
            apply_report(REPORTS_DIR / REPORT_NAME, MODELS_DIR)
            print("operating-metrics now quote the holdout figures")
        return
    try:
        path = score_holdout(
            load_dataset(), load_config(), MODELS_DIR, REPORTS_DIR, rescore=args.rescore
        )
    except HoldoutAlreadyScoredError as error:
        raise SystemExit(str(error)) from error
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
