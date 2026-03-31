"""
Tests for src/pipeline/utils.py

Run with:
    pytest tests/ -v
    pytest tests/ --cov=src --cov-report=term-missing
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "pipeline"))
from utils import (
    extract_raw_features,
    group_titles,
    impute,
    engineer_features,
    drop_raw_columns,
    encode,
    scale,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

CONFIG = {
    "features": {
        "title_mapping":   {"Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss"},
        "title_keep":      ["Mr", "Mrs", "Miss", "Master"],
        "title_rare_label": "Rare",
        "age_bins":        [0, 16, 60, float("inf")],
        "age_labels":      ["Child", "Adult", "Senior"],
        "drop":            ["Name", "Ticket", "Cabin", "PassengerId", "SibSp", "Parch"],
    },
    "preprocessing": {
        "impute": {
            "Age":        "median",
            "Embarked":   "mode",
            "Cabin_deck": "Unknown",
        },
        "scale":  ["Age", "Fare", "FamilySize", "LogFare", "Pclass_x_Fare", "FarePerPerson"],
        "encode": ["Sex", "Embarked", "Name_title", "Cabin_deck", "AgeGroup"],
    },
    "training": {"target": "Survived"},
}


def make_raw_df(n: int = 5) -> pd.DataFrame:
    """Minimal raw Titanic-shaped DataFrame."""
    return pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived":    [1, 0, 1, 0, 1],
        "Pclass":      [1, 3, 2, 3, 1],
        "Name":        [
            "Smith, Mr. John",
            "Doe, Mrs. Jane",
            "Brown, Miss. Alice",
            "Jones, Master. Tom",
            "White, Dr. Henry",
        ],
        "Sex":         ["male", "female", "female", "male", "male"],
        "Age":         [30.0, np.nan, 22.0, 8.0, 45.0],
        "SibSp":       [1, 0, 0, 3, 0],
        "Parch":       [0, 1, 0, 1, 0],
        "Ticket":      ["A/5 21171", "PC 17599", "113803", "349909", "113783"],
        "Fare":        [7.25, 71.28, 53.10, 21.07, 211.34],
        "Cabin":       [np.nan, "C85", np.nan, np.nan, "B28"],
        "Embarked":    ["S", "C", "S", np.nan, "S"],
    })


# ── extract_raw_features ──────────────────────────────────────────────────────

def test_extract_name_title():
    df = make_raw_df()
    out = extract_raw_features(df)
    assert "Name_title" in out.columns
    assert out["Name_title"].tolist() == ["Mr", "Mrs", "Miss", "Master", "Dr"]


def test_extract_cabin_deck():
    df = make_raw_df()
    out = extract_raw_features(df)
    assert "Cabin_deck" in out.columns
    assert out["Cabin_deck"].tolist() == [np.nan, "C", np.nan, np.nan, "B"]


def test_extract_does_not_mutate_input():
    df = make_raw_df()
    original_cols = list(df.columns)
    extract_raw_features(df)
    assert list(df.columns) == original_cols


# ── group_titles ──────────────────────────────────────────────────────────────

def test_group_titles_maps_mlle():
    df = pd.DataFrame({"Name_title": ["Mlle"]})
    out = group_titles(df, CONFIG)
    assert out["Name_title"].iloc[0] == "Miss"


def test_group_titles_rare():
    df = pd.DataFrame({"Name_title": ["Dr", "Rev", "Mr"]})
    out = group_titles(df, CONFIG)
    assert out["Name_title"].tolist() == ["Rare", "Rare", "Mr"]


def test_group_titles_keeps_common():
    df = pd.DataFrame({"Name_title": ["Mr", "Mrs", "Miss", "Master"]})
    out = group_titles(df, CONFIG)
    assert out["Name_title"].tolist() == ["Mr", "Mrs", "Miss", "Master"]


# ── impute ────────────────────────────────────────────────────────────────────

def test_impute_age_median():
    df = pd.DataFrame({"Age": [10.0, np.nan, 30.0], "Embarked": ["S", "S", "S"], "Cabin_deck": ["A", "A", "A"]})
    out, stats = impute(df, CONFIG)
    assert out["Age"].isnull().sum() == 0
    assert stats["Age"] == 20.0  # median of [10, 30]


def test_impute_uses_train_stats_on_test():
    train = pd.DataFrame({"Age": [10.0, 30.0], "Embarked": ["S", "S"], "Cabin_deck": ["A", "A"]})
    test  = pd.DataFrame({"Age": [np.nan],      "Embarked": ["S"],      "Cabin_deck": ["A"]})
    _, stats = impute(train, CONFIG)
    out, _   = impute(test, CONFIG, stats)
    assert out["Age"].iloc[0] == stats["Age"]


def test_impute_cabin_deck_unknown():
    df = pd.DataFrame({"Age": [30.0], "Embarked": ["S"], "Cabin_deck": [np.nan]})
    out, _ = impute(df, CONFIG)
    assert out["Cabin_deck"].iloc[0] == "Unknown"


def test_impute_does_not_mutate_input():
    df = pd.DataFrame({"Age": [np.nan], "Embarked": ["S"], "Cabin_deck": ["A"]})
    impute(df, CONFIG)
    assert df["Age"].isnull().sum() == 1


# ── engineer_features ─────────────────────────────────────────────────────────

def test_family_size():
    df = pd.DataFrame({"SibSp": [1, 0, 2], "Parch": [0, 0, 1],
                        "Fare": [10.0, 20.0, 30.0], "Age": [30.0, 25.0, 10.0],
                        "Pclass": [1, 2, 3]})
    out = engineer_features(df, CONFIG)
    assert list(out["FamilySize"]) == [2, 1, 4]


def test_is_alone():
    df = pd.DataFrame({"SibSp": [1, 0, 2], "Parch": [0, 0, 1],
                        "Fare": [10.0, 20.0, 30.0], "Age": [30.0, 25.0, 10.0],
                        "Pclass": [1, 2, 3]})
    out = engineer_features(df, CONFIG)
    assert list(out["IsAlone"]) == [0, 1, 0]


def test_log_fare():
    df = pd.DataFrame({"SibSp": [0], "Parch": [0], "Fare": [0.0],
                        "Age": [30.0], "Pclass": [1]})
    out = engineer_features(df, CONFIG)
    assert out["LogFare"].iloc[0] == pytest.approx(np.log1p(0.0))


def test_age_group_child():
    df = pd.DataFrame({"SibSp": [0], "Parch": [0], "Fare": [10.0],
                        "Age": [10.0], "Pclass": [1]})
    out = engineer_features(df, CONFIG)
    assert str(out["AgeGroup"].iloc[0]) == "Child"


def test_fare_per_person_no_division_by_zero():
    df = pd.DataFrame({"SibSp": [0], "Parch": [0], "Fare": [100.0],
                        "Age": [30.0], "Pclass": [2]})
    out = engineer_features(df, CONFIG)
    # FamilySize = 1, so FarePerPerson = 100
    assert out["FarePerPerson"].iloc[0] == pytest.approx(100.0)


def test_engineer_does_not_mutate_input():
    df = pd.DataFrame({"SibSp": [1], "Parch": [0], "Fare": [10.0],
                        "Age": [30.0], "Pclass": [1]})
    original_cols = list(df.columns)
    engineer_features(df, CONFIG)
    assert list(df.columns) == original_cols


# ── drop_raw_columns ──────────────────────────────────────────────────────────

def test_drop_raw_columns():
    df = pd.DataFrame({"Name": ["x"], "Ticket": ["y"], "Cabin": ["z"],
                        "PassengerId": [1], "SibSp": [0], "Parch": [0], "Age": [30.0]})
    out = drop_raw_columns(df, CONFIG)
    assert "Name" not in out.columns
    assert "Age" in out.columns


def test_drop_raw_columns_ignores_missing():
    df = pd.DataFrame({"Age": [30.0]})
    out = drop_raw_columns(df, CONFIG)   # should not raise
    assert "Age" in out.columns


# ── encode ────────────────────────────────────────────────────────────────────

def test_encode_creates_dummies():
    X_train = pd.DataFrame({"Sex": ["male", "female"], "Embarked": ["S", "C"],
                             "Name_title": ["Mr", "Mrs"], "Cabin_deck": ["A", "Unknown"],
                             "AgeGroup": ["Adult", "Child"]})
    X_test  = X_train.copy()
    out_train, out_test = encode(X_train, X_test, CONFIG)
    assert "Sex" not in out_train.columns
    assert any("Sex" in c for c in out_train.columns)


def test_encode_aligns_test_to_train():
    X_train = pd.DataFrame({"Sex": ["male", "female"], "Embarked": ["S", "C"],
                             "Name_title": ["Mr", "Mrs"], "Cabin_deck": ["A", "B"],
                             "AgeGroup": ["Adult", "Child"]})
    X_test  = pd.DataFrame({"Sex": ["male"], "Embarked": ["S"],
                             "Name_title": ["Mr"], "Cabin_deck": ["Unknown"],
                             "AgeGroup": ["Senior"]})
    out_train, out_test = encode(X_train, X_test, CONFIG)
    assert list(out_train.columns) == list(out_test.columns)


# ── scale ─────────────────────────────────────────────────────────────────────

def test_scale_zero_mean():
    X_train = pd.DataFrame({"Age": [10.0, 20.0, 30.0], "Fare": [5.0, 10.0, 15.0],
                             "FamilySize": [1.0, 2.0, 3.0], "LogFare": [1.0, 2.0, 3.0],
                             "Pclass_x_Fare": [5.0, 20.0, 45.0], "FarePerPerson": [5.0, 5.0, 5.0]})
    X_test  = X_train.copy()
    out_train, _, _ = scale(X_train, X_test, CONFIG)
    assert out_train["Age"].mean() == pytest.approx(0.0, abs=1e-10)


def test_scale_uses_train_stats_on_test():
    X_train = pd.DataFrame({"Age": [10.0, 30.0], "Fare": [5.0, 15.0],
                             "FamilySize": [1.0, 3.0], "LogFare": [1.0, 3.0],
                             "Pclass_x_Fare": [5.0, 45.0], "FarePerPerson": [5.0, 5.0]})
    X_test  = pd.DataFrame({"Age": [20.0], "Fare": [10.0],
                             "FamilySize": [2.0], "LogFare": [2.0],
                             "Pclass_x_Fare": [20.0], "FarePerPerson": [5.0]})
    _, out_test, fitted_scaler = scale(X_train, X_test, CONFIG)
    # Test mean value (20) should scale to 0 since train mean is 20
    assert out_test["Age"].iloc[0] == pytest.approx(0.0, abs=1e-10)
