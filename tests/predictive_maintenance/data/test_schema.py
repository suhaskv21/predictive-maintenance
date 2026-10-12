from predictive_maintenance.data import schema


def test_candidate_features_are_type_plus_five_sensors() -> None:
    assert schema.FEATURE_COLUMNS == (
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    )


def test_target_is_machine_failure() -> None:
    assert schema.TARGET == "Machine failure"


def test_failure_modes_are_outcome_fields() -> None:
    assert schema.FAILURE_MODES == ("TWF", "HDF", "PWF", "OSF", "RNF")


def test_features_never_include_target_failure_modes_or_identifiers() -> None:
    forbidden = {schema.TARGET, *schema.FAILURE_MODES, "UDI", "Product ID"}
    assert forbidden.isdisjoint(schema.FEATURE_COLUMNS)


def test_every_column_is_either_a_feature_or_has_a_declared_role() -> None:
    declared = {
        *schema.FEATURE_COLUMNS,
        schema.TARGET,
        *schema.FAILURE_MODES,
        *schema.IDENTIFIER_COLUMNS,
    }
    assert declared == set(schema.REQUIRED_COLUMNS)
