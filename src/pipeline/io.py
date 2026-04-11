"""
io.py — Config and data loading.
"""

import yaml
import pandas as pd


REQUIRED_COLUMNS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return list of missing required columns. Empty list means schema is valid."""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]
