"""Column roles of the AI4I 2020 dataset (ADR 0004 proposes this feature set)."""

CATEGORICAL_FEATURE = "Type"
TYPE_LEVELS = ("L", "M", "H")
NUMERIC_FEATURES = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
FEATURE_COLUMNS = (CATEGORICAL_FEATURE, *NUMERIC_FEATURES)

TARGET = "Machine failure"
FAILURE_MODES = ("TWF", "HDF", "PWF", "OSF", "RNF")
IDENTIFIER_COLUMNS = ("UDI", "Product ID")

REQUIRED_COLUMNS = (*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, TARGET, *FAILURE_MODES)
