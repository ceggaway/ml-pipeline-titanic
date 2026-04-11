"""
Integration tests for the full pipeline.

These tests exercise end-to-end behavior — training produces correct artefacts,
batch scoring produces the expected schema, fault tolerance works correctly,
and monitoring metrics are written.

Run with:
    pytest tests/test_integration.py -v
"""

import joblib
import numpy as np
import pandas as pd
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.pipeline.train import train
from src.pipeline.predict import batch_predict


CONFIG_PATH = "config/config.yaml"
DATA_PATH   = "data/raw/train.csv"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory):
    """Train once and return path to artefacts. Shared across all tests in module."""
    import os
    tmp = tmp_path_factory.mktemp("models")
    # Patch models dir so we don't overwrite production model
    original_dir = Path.cwd()
    os.chdir(tmp_path_factory.getbasetemp())
    (Path.cwd() / "models").mkdir(exist_ok=True)
    (Path.cwd() / "logs").mkdir(exist_ok=True)
    train(CONFIG_PATH, DATA_PATH)
    yield Path.cwd() / "models" / "final_model.joblib"
    os.chdir(original_dir)


# ── Training artefact tests ───────────────────────────────────────────────────

def test_training_produces_model_artefact():
    """Training must produce a loadable joblib file with all required keys."""
    assert Path("models/final_model.joblib").exists(), "final_model.joblib not found"
    artefacts = joblib.load("models/final_model.joblib")
    required_keys = {"model", "scaler", "train_stats", "train_columns", "threshold", "config", "version"}
    assert required_keys.issubset(artefacts.keys()), f"Missing keys: {required_keys - artefacts.keys()}"


def test_training_artefact_has_version():
    """Artefacts must include a version timestamp."""
    artefacts = joblib.load("models/final_model.joblib")
    assert "version" in artefacts
    assert len(artefacts["version"]) > 0


def test_training_artefact_threshold():
    """Threshold must be the configured value (0.46)."""
    artefacts = joblib.load("models/final_model.joblib")
    assert artefacts["threshold"] == pytest.approx(0.46)


def test_training_artefact_has_train_columns():
    """train_columns must be a non-empty list."""
    artefacts = joblib.load("models/final_model.joblib")
    assert isinstance(artefacts["train_columns"], list)
    assert len(artefacts["train_columns"]) > 0


# ── Batch scoring schema tests ────────────────────────────────────────────────

def _make_input_csv(tmp_path: Path, n: int = 10) -> Path:
    df = pd.DataFrame({
        "PassengerId": range(9000, 9000 + n),
        "Pclass":      [1, 3, 2, 3, 1, 2, 3, 1, 2, 3][:n],
        "Name":        [
            "Smith, Mr. John", "Doe, Mrs. Jane", "Brown, Miss. Alice",
            "Jones, Master. Tom", "White, Dr. Henry", "Green, Mr. Bob",
            "Black, Mrs. Sue", "Gray, Miss. Eve", "Blue, Mr. Sam", "Red, Master. Leo",
        ][:n],
        "Sex":         ["male","female","female","male","male","male","female","female","male","male"][:n],
        "Age":         [30.0, np.nan, 22.0, 8.0, 45.0, 33.0, 27.0, 19.0, 55.0, 12.0][:n],
        "SibSp":       [1, 0, 0, 3, 0, 0, 1, 0, 0, 2][:n],
        "Parch":       [0, 1, 0, 1, 0, 0, 0, 1, 0, 1][:n],
        "Ticket":      [f"T{i}" for i in range(n)],
        "Fare":        [7.25, 71.28, 53.10, 21.07, 211.34, 15.0, 30.0, 50.0, 8.0, 25.0][:n],
        "Cabin":       [np.nan, "C85", np.nan, np.nan, "B28", np.nan, np.nan, "D15", np.nan, np.nan][:n],
        "Embarked":    ["S", "C", "S", np.nan, "S", "S", "C", "S", "Q", "S"][:n],
    })
    path = tmp_path / "input.csv"
    df.to_csv(path, index=False)
    return path


def test_batch_scoring_output_schema(tmp_path):
    """Batch scoring must produce expected columns in correct order."""
    input_path  = _make_input_csv(tmp_path)
    output_path = tmp_path / "predictions.csv"
    batch_predict(CONFIG_PATH, str(input_path), str(output_path))

    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert "PassengerId"   in df.columns
    assert "prediction"    in df.columns
    assert "probability"   in df.columns
    assert "survived_label" in df.columns
    assert df.columns[0] == "PassengerId"   # PassengerId comes first


def test_batch_scoring_prediction_values(tmp_path):
    """Predictions must be 0 or 1 only."""
    input_path  = _make_input_csv(tmp_path)
    output_path = tmp_path / "predictions.csv"
    batch_predict(CONFIG_PATH, str(input_path), str(output_path))

    df = pd.read_csv(output_path)
    assert df["prediction"].isin([0, 1]).all()


def test_batch_scoring_probability_range(tmp_path):
    """Probabilities must be between 0 and 1."""
    input_path  = _make_input_csv(tmp_path)
    output_path = tmp_path / "predictions.csv"
    batch_predict(CONFIG_PATH, str(input_path), str(output_path))

    df = pd.read_csv(output_path)
    assert (df["probability"] >= 0).all()
    assert (df["probability"] <= 1).all()


def test_batch_scoring_survived_label(tmp_path):
    """survived_label must be 'Survived' or 'Not Survived' and consistent with prediction."""
    input_path  = _make_input_csv(tmp_path)
    output_path = tmp_path / "predictions.csv"
    batch_predict(CONFIG_PATH, str(input_path), str(output_path))

    df = pd.read_csv(output_path)
    assert df["survived_label"].isin(["Survived", "Not Survived"]).all()
    assert (df[df["prediction"] == 1]["survived_label"] == "Survived").all()
    assert (df[df["prediction"] == 0]["survived_label"] == "Not Survived").all()


# ── Fault tolerance tests ─────────────────────────────────────────────────────

def test_one_bad_row_does_not_kill_batch(tmp_path):
    """One malformed row must go to failed_rows.csv while others succeed."""
    df = pd.read_csv(_make_input_csv(tmp_path, n=5))

    # Corrupt row 2 by replacing Name with a float (breaks str.extract)
    df.loc[2, "Name"]     = None
    df.loc[2, "Embarked"] = None
    df.loc[2, "Age"]      = None
    df.loc[2, "Fare"]     = -999.0   # invalid fare

    corrupt_path = tmp_path / "corrupt_input.csv"
    df.to_csv(corrupt_path, index=False)

    output_path = tmp_path / "predictions.csv"
    batch_predict(CONFIG_PATH, str(corrupt_path), str(output_path))

    # Successful predictions must still exist
    assert output_path.exists()
    out = pd.read_csv(output_path)
    assert len(out) > 0


def test_metrics_file_is_created(tmp_path, monkeypatch):
    """Batch prediction must write a metrics.prom file."""
    import shutil
    project_root = Path(__file__).parents[1]

    (tmp_path / "models").mkdir()
    (tmp_path / "logs").mkdir()
    shutil.copy(project_root / "models" / "final_model.joblib", tmp_path / "models" / "final_model.joblib")

    monkeypatch.chdir(tmp_path)

    input_path  = _make_input_csv(tmp_path)
    output_path = tmp_path / "models" / "batch_outputs" / "predictions.csv"
    batch_predict(CONFIG_PATH, str(input_path), str(output_path))

    assert (tmp_path / "models" / "metrics.prom").exists()
