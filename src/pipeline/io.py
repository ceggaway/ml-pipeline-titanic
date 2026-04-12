"""
io.py — Config and data loading, schema validation, and data contracts.
"""

import logging
import yaml
import pandas as pd

logger = logging.getLogger("pipeline")

REQUIRED_COLUMNS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

COLUMN_DTYPES = {
    "Pclass":  "numeric",
    "Age":     "numeric",
    "SibSp":   "numeric",
    "Parch":   "numeric",
    "Fare":    "numeric",
    "Name":    "string",
    "Sex":     "string",
    "Ticket":  "string",
    "Embarked":"string",
}

ALLOWED_VALUES = {
    "Sex":      {"male", "female"},
    "Embarked": {"S", "C", "Q"},
    "Pclass":   {1, 2, 3},
}

VALUE_RANGES = {
    "Fare":  (0, None),    # must be non-negative
    "Age":   (0, 120),
    "Pclass":(1, 3),
    "SibSp": (0, 10),
    "Parch": (0, 10),
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def validate_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Validate input DataFrame against schema contracts.

    Returns
    -------
    errors   : list of blocking issues (will halt the pipeline)
    warnings : list of non-blocking issues (logged but pipeline continues)
    """
    errors   = []
    warnings = []

    # 1. Required columns present
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return errors, warnings  # no point checking further

    # 2. Dtype checks
    for col, expected in COLUMN_DTYPES.items():
        if col not in df.columns:
            continue
        non_null = df[col].dropna()
        if expected == "numeric" and not pd.api.types.is_numeric_dtype(non_null):
            errors.append(f"Column '{col}' expected numeric, got {df[col].dtype}")
        elif expected == "string" and pd.api.types.is_numeric_dtype(non_null):
            errors.append(f"Column '{col}' expected string, got {df[col].dtype}")

    # 3. Allowed category values
    for col, allowed in ALLOWED_VALUES.items():
        if col not in df.columns:
            continue
        non_null    = df[col].dropna()
        bad_vals    = set(non_null.unique()) - allowed
        if bad_vals:
            warnings.append(f"Column '{col}' contains unexpected values: {bad_vals} (allowed: {allowed})")

    # 4. Value ranges
    for col, (lo, hi) in VALUE_RANGES.items():
        if col not in df.columns:
            continue
        non_null = df[col].dropna()
        if lo is not None and (non_null < lo).any():
            errors.append(f"Column '{col}' has values below minimum {lo}: min={non_null.min()}")
        if hi is not None and (non_null > hi).any():
            warnings.append(f"Column '{col}' has values above expected maximum {hi}: max={non_null.max()}")

    # 5. Null rate summary
    high_null_cols = [
        col for col in REQUIRED_COLUMNS
        if col in df.columns and df[col].isnull().mean() > 0.5
    ]
    if high_null_cols:
        warnings.append(f"High null rate (>50%) in columns: {high_null_cols}")

    return errors, warnings


def log_validation_report(errors: list[str], warnings: list[str]) -> None:
    if warnings:
        for w in warnings:
            logger.warning(f"[Schema] {w}")
    if errors:
        for e in errors:
            logger.error(f"[Schema] {e}")
    if not errors and not warnings:
        logger.info("Schema validation passed — no issues found.")
    elif not errors:
        logger.info(f"Schema validation passed with {len(warnings)} warning(s).")
