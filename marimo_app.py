import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from predictive_maintenance.config import load_config
    from predictive_maintenance.data import schema
    from predictive_maintenance.inference.artifact import DisabledUnit, load_unit
    from predictive_maintenance.inference.assessment import assess
    from predictive_maintenance.inference.examples import find_example, load_examples, recorded_outcome
    from predictive_maintenance.inference.golden import check_golden
    from predictive_maintenance.inference.presentation import (
        golden_banner,
        present,
        present_unavailable,
    )
    from predictive_maintenance.inference.site import config_file, models_dir

    return (
        DisabledUnit,
        assess,
        check_golden,
        config_file,
        find_example,
        golden_banner,
        load_config,
        load_examples,
        load_unit,
        mo,
        models_dir,
        present,
        present_unavailable,
        recorded_outcome,
        schema,
    )


@app.cell
def _():
    # The build workflow stamps the commit the site was built from (ADR 0012).
    BUILD_COMMIT = "local run"
    MODEL = "random_forest_unweighted"
    return BUILD_COMMIT, MODEL


@app.cell
def _(MODEL, mo, models_dir):
    # Local run: the folders beside the notebook. Published site: downloaded from its public/ folder.
    unit_dir = models_dir(mo.notebook_location(), MODEL)
    return (unit_dir,)


@app.cell
def _(config_file, mo):
    config_path = config_file(mo.notebook_location())
    return (config_path,)


@app.cell
def _(MODEL, config_path, load_config, load_unit, unit_dir):
    config = load_config(config_path)
    unit = load_unit(unit_dir, MODEL)
    return config, unit


@app.cell
def _(DisabledUnit, check_golden, golden_banner, unit):
    # Scores the manifest's embedded train Readings once at load; None unless the check fails.
    scoring_banner = None if isinstance(unit, DisabledUnit) else golden_banner(check_golden(unit))
    return (scoring_banner,)


@app.cell
def _(load_examples, mo, unit):
    # Training stored the Example readings (train partition only) in the unit; labels hide the outcome.
    examples = load_examples(unit)
    example_picker = mo.ui.dropdown(
        [example.label for example in examples], allow_select_none=True, label="Example reading"
    )
    return example_picker, examples


@app.cell
def _(example_picker, examples, find_example):
    example = find_example(examples, example_picker.value)
    return (example,)


@app.cell
def _(config, example, mo, schema):
    # Choosing an Example reading rebuilds the form with its values; without one, the defaults.
    _air, _process, _speed, _torque, _wear = schema.NUMERIC_FEATURES
    _defaults = {
        schema.CATEGORICAL_FEATURE: "M",
        _air: 300.0,
        _process: 310.0,
        _speed: 1500,
        _torque: 40.0,
        _wear: 100,
    }
    _start = example.reading if example else _defaults
    inputs = mo.ui.dictionary(
        {
            schema.CATEGORICAL_FEATURE: mo.ui.dropdown(
                list(config.plausibility.types),
                value=_start[schema.CATEGORICAL_FEATURE],
                allow_select_none=True,
                label=schema.CATEGORICAL_FEATURE,
            ),
            **{
                name: mo.ui.number(value=_start[name], label=name)
                for name in schema.NUMERIC_FEATURES
            },
        }
    )
    return (inputs,)


@app.cell
def _(inputs):
    reading = inputs.value
    return (reading,)


@app.cell
def _(
    DisabledUnit,
    assess,
    config,
    example,
    present,
    present_unavailable,
    reading,
    recorded_outcome,
    unit,
):
    _ratio = config.threshold.default_cost_ratio
    if isinstance(unit, DisabledUnit):
        presentation = present_unavailable(unit, cost_ratio=_ratio)
    else:
        presentation = present(
            assess(unit, reading, config.plausibility),
            unit.operating_point,
            unit.operating_metrics,
            cost_ratio=_ratio,
            recorded_outcome=recorded_outcome(example, reading),
        )
    return (presentation,)


@app.cell
def _(BUILD_COMMIT, example_picker, examples, inputs, mo, presentation, scoring_banner):
    _kinds = {
        "inspect": "warn",
        "no_inspection_flagged": "neutral",
        "not_assessed": "info",
        "model_unavailable": "danger",
    }

    def _lines(lines):
        return mo.vstack([mo.plain_text(line) for line in lines if line], gap=0.25)

    # One column, and long lines wrap, so the page keeps its width whatever is rendered.
    _wrap = mo.Html("<style>pre.text-xs{white-space:pre-wrap;overflow-wrap:anywhere}</style>")

    mo.vstack(
        [
            _wrap,
            mo.md("# Machine failure check"),
            *([mo.callout(mo.plain_text(scoring_banner), kind="danger")] if scoring_banner else []),
            *([example_picker] if examples else []),
            mo.md("Enter the six readings from one machine."),
            mo.vstack(list(inputs.elements.values()), gap=0.5),
            mo.callout(mo.md(f"## {presentation.headline}"), kind=_kinds[presentation.state]),
            _lines(
                [
                    presentation.failure_risk,
                    presentation.flag_line,
                    presentation.flag_line_note,
                    presentation.range_note,
                ]
            ),
            _lines(presentation.refusal),
            _lines([presentation.outcome_line]),
            *(
                [mo.accordion({presentation.meaning_title: _lines(presentation.meaning)})]
                if presentation.meaning
                else []
            ),
            *(
                [
                    mo.accordion(
                        {presentation.sensitivity_title: _lines(presentation.sensitivity_lines())}
                    )
                ]
                if presentation.sensitivity
                else []
            ),
            mo.md("---"),
            mo.accordion(
                {
                    presentation.model_card_title: _lines(
                        [presentation.model_line, *presentation.caveats]
                    )
                }
            ),
            mo.md(f"<small>Build commit: `{BUILD_COMMIT}`</small>"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
