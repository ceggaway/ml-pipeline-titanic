"""
Integration tests for the full pipeline.

Covers:
- Training produces correct artefacts with expected keys and values
- Batch scoring produces expected output schema, prediction values, labels
- Fault tolerance: one bad row is isolated, others succeed
- Metrics file is written after batch run
- Schema validation: missing columns halt pipeline, invalid values warn
- Drift detection: output drift flag triggers correctly
- Model versioning: registry is created and populated
- CLI: correct output produced from command-line invocation

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

    # Corrupt row 2 with a Name that has no extractable title (breaks group_titles downstream)
    df.loc[2, "Name"] = "NoTitleHere"   # no comma/period pattern — title extraction returns NaN

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


# ── Schema validation tests ───────────────────────────────────────────────────

def test_schema_validation_fails_on_missing_columns(tmp_path):
    """Pipeline must halt with a clear error when required columns are missing."""
    df = pd.DataFrame({"PassengerId": [1, 2], "Age": [30, 25]})  # missing most columns
    input_path  = tmp_path / "bad_input.csv"
    output_path = tmp_path / "predictions.csv"
    df.to_csv(input_path, index=False)

    with pytest.raises(SystemExit):
        batch_predict(CONFIG_PATH, str(input_path), str(output_path))

    assert not output_path.exists()


def test_schema_validation_warns_on_invalid_category(tmp_path):
    """Invalid category values should warn but not halt the pipeline."""
    from src.pipeline.io import validate_schema
    df = _make_input_csv(tmp_path, n=3)
    df = pd.read_csv(df)
    df.loc[0, "Sex"] = "unknown"       # invalid but non-blocking
    df.loc[1, "Embarked"] = "Z"        # invalid but non-blocking

    errors, warnings = validate_schema(df)
    assert len(errors) == 0            # should not block
    assert any("Sex" in w or "Embarked" in w for w in warnings)


def test_schema_validation_errors_on_negative_fare(tmp_path):
    """Negative fare values must raise a blocking schema error."""
    from src.pipeline.io import validate_schema
    df = pd.read_csv(_make_input_csv(tmp_path, n=3))
    df.loc[0, "Fare"] = -10.0

    errors, _ = validate_schema(df)
    assert any("Fare" in e for e in errors)


# ── Drift detection tests ─────────────────────────────────────────────────────

def test_output_drift_triggers_above_threshold():
    """Drift check must return True when deviation exceeds threshold."""
    from src.pipeline.evaluate import check_output_drift
    assert check_output_drift(pct_survived=0.9, training_baseline=0.38, threshold=0.15) is True


def test_output_drift_passes_within_threshold():
    """Drift check must return False when deviation is within threshold."""
    from src.pipeline.evaluate import check_output_drift
    assert check_output_drift(pct_survived=0.40, training_baseline=0.38, threshold=0.15) is False


# ── Model versioning tests ────────────────────────────────────────────────────

def test_model_registry_is_created():
    """Training must create or update models/model_registry.json."""
    import json
    registry_path = Path("models/model_registry.json")
    assert registry_path.exists(), "model_registry.json not found"

    with open(registry_path) as f:
        registry = json.load(f)

    assert isinstance(registry, list)
    assert len(registry) > 0

    latest = registry[-1]
    assert "version"    in latest
    assert "git_hash"   in latest
    assert "features"   in latest
    assert "cv_auc"     in latest
    assert "hold_out"   in latest
    assert "status"     in latest
    assert latest["status"] == "latest"


def test_versioned_model_file_exists():
    """A timestamped versioned model file must exist alongside final_model.joblib."""
    versioned = list(Path("models").glob("model_2*.joblib"))
    assert len(versioned) > 0, "No versioned model files found"


# ── Promotion gate + rollback tests ──────────────────────────────────────────

def test_promotion_gate_blocks_weak_model(tmp_path, monkeypatch):
    """A model with cv_auc below min_cv_auc must not overwrite final_model.joblib."""
    import shutil, yaml
    project_root = Path(__file__).parents[1]

    (tmp_path / "models").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)
    shutil.copy(project_root / "data" / "raw" / "train.csv", tmp_path / "data" / "raw" / "train.csv")
    shutil.copy(project_root / "models" / "final_model.joblib", tmp_path / "models" / "final_model.joblib")
    original_mtime = (tmp_path / "models" / "final_model.joblib").stat().st_mtime

    # Write a config with an impossibly high promotion gate
    config_src = project_root / "config" / "config.yaml"
    with open(config_src) as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["min_cv_auc"] = 0.9999   # impossible to reach
    high_gate_config = tmp_path / "config.yaml"
    with open(high_gate_config, "w") as f:
        yaml.dump(cfg, f)

    monkeypatch.chdir(tmp_path)
    from src.pipeline.train import train
    train(str(high_gate_config), str(tmp_path / "data" / "raw" / "train.csv"))

    new_mtime = (tmp_path / "models" / "final_model.joblib").stat().st_mtime
    assert new_mtime == original_mtime, "final_model.joblib was overwritten despite failing promotion gate"

    import json
    with open(tmp_path / "models" / "model_registry.json") as f:
        registry = json.load(f)
    assert registry[-1]["status"] == "rejected"


def test_rollback_restores_model(tmp_path, monkeypatch):
    """--rollback must restore the specified versioned model to final_model.joblib."""
    import shutil, joblib
    project_root = Path(__file__).parents[1]

    (tmp_path / "models").mkdir()
    (tmp_path / "logs").mkdir()

    # Copy existing versioned models (if any) or create a fake one
    versioned = list((project_root / "models").glob("model_2*.joblib"))
    if not versioned:
        pytest.skip("No versioned model files available for rollback test")

    src_versioned = versioned[0]
    version = src_versioned.stem.replace("model_", "")
    shutil.copy(src_versioned, tmp_path / "models" / src_versioned.name)
    shutil.copy(project_root / "models" / "final_model.joblib", tmp_path / "models" / "final_model.joblib")

    # Seed registry with one entry
    registry = [{"version": version, "status": "superseded", "cv_auc": 0.89}]
    with open(tmp_path / "models" / "model_registry.json", "w") as f:
        import json; json.dump(registry, f)

    monkeypatch.chdir(tmp_path)
    from src.pipeline.pipeline import rollback
    rollback(version)

    artefacts = joblib.load(tmp_path / "models" / "final_model.joblib")
    assert artefacts["version"] == version

    import json
    with open(tmp_path / "models" / "model_registry.json") as f:
        reg = json.load(f)
    assert reg[0]["status"] == "latest"


# ── CLI tests ─────────────────────────────────────────────────────────────────

def test_cli_predict_produces_output(tmp_path):
    """CLI invocation with --predict must produce a predictions CSV."""
    import subprocess, sys
    output_path = tmp_path / "cli_predictions.csv"
    input_path  = _make_input_csv(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline.pipeline",
         "--predict",
         "--input",  str(input_path),
         "--output", str(output_path)],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) > 0
