import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "pipeline"))

from utils import engineer_features


def make_sample_df():
    return pd.DataFrame({
        "SibSp": [1, 0, 2],
        "Parch": [0, 0, 1],
    })


def test_family_size():
    df = make_sample_df()
    result = engineer_features(df)
    assert list(result["FamilySize"]) == [2, 1, 4]


def test_is_alone():
    df = make_sample_df()
    result = engineer_features(df)
    assert list(result["IsAlone"]) == [0, 1, 0]


def test_engineer_features_does_not_mutate_input():
    df = make_sample_df()
    original_cols = list(df.columns)
    engineer_features(df)
    assert list(df.columns) == original_cols
