"""
pipeline.py — Training and batch prediction entry point.

Usage:
    # Train and save model
    python src/pipeline/pipeline.py --config config/config.yaml --data data/raw/train.csv

    # Batch predict
    python src/pipeline/pipeline.py --predict \
        --input data/raw/test.csv \
        --output models/batch_outputs/predictions_$(date +%Y%m%d).csv
"""

import argparse
import joblib
import yaml
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import preprocess_train, preprocess_inference


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def build_model(config: dict) -> CatBoostClassifier:
    params = config["model"]["params"]
    return CatBoostClassifier(**params)


def evaluate(y_test, y_pred, y_prob, threshold: float = 0.5) -> None:
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Hold-out ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Decision threshold: {threshold}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))


def train(config_path: str, data_path: str) -> None:
    config = load_config(config_path)

    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"  Rows: {len(df)} | Columns: {len(df.columns)}")

    print("\nPreprocessing...")
    X_train, X_test, y_train, y_test, train_stats, scaler = preprocess_train(df, config)
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {X_train.shape[1]}")

    print("\nApplying undersampling...")
    rus = RandomUnderSampler(
        random_state=config["imbalance"]["random_state"]
    )
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    print(f"  Resampled train: {len(X_train_res)} rows | Balance: {y_train_res.mean():.2%}")

    print("\nTraining CatBoost...")
    model = build_model(config)
    model.fit(X_train_res, y_train_res)

    threshold = config["model"].get("threshold", 0.5)
    y_prob    = model.predict_proba(X_test)[:, 1]
    y_pred    = (y_prob >= threshold).astype(int)
    evaluate(y_test, y_pred, y_prob, threshold)

    print("\nRunning stratified 5-fold CV (resampling inside each fold)...")
    skf = StratifiedKFold(
        n_splits=config["training"]["cv_folds"],
        shuffle=True,
        random_state=42,
    )
    cv_pipe = ImbPipeline([
        ("sampler", RandomUnderSampler(random_state=config["imbalance"]["random_state"])),
        ("clf",     build_model(config)),
    ])
    cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=skf, scoring="roc_auc")
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Per-fold:   {[round(s, 4) for s in cv_scores]}")

    # Save model artefacts — everything needed for inference
    model_dir  = Path("models")
    model_dir.mkdir(exist_ok=True)
    artefacts  = {
        "model":         model,
        "scaler":        scaler,
        "train_stats":   train_stats,
        "train_columns": X_train.columns.tolist(),
        "threshold":     threshold,
        "config":        config,
    }
    model_path = model_dir / "final_model.joblib"
    joblib.dump(artefacts, model_path)
    print(f"\nModel saved → {model_path}")


def batch_predict(config_path: str, input_path: str, output_path: str) -> None:
    print(f"Loading model artefacts...")
    artefacts    = joblib.load("models/final_model.joblib")
    model        = artefacts["model"]
    scaler       = artefacts["scaler"]
    train_stats  = artefacts["train_stats"]
    train_columns = artefacts["train_columns"]
    threshold    = artefacts["threshold"]
    config       = artefacts["config"]

    print(f"Loading input data from {input_path}...")
    df = load_data(input_path)
    print(f"  Rows: {len(df)}")

    print("Preprocessing for inference...")
    X = preprocess_inference(df, config, train_stats, scaler, train_columns)

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    output = pd.DataFrame({
        "prediction":  y_pred,
        "probability": y_prob.round(4),
        "survived_label": ["Survived" if p == 1 else "Not Survived" for p in y_pred],
    })

    # If test data has PassengerId, include it for reference
    if "PassengerId" in df.columns:
        output.insert(0, "PassengerId", df["PassengerId"].values)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"\nPredictions saved → {output_path}")
    print(f"  Total rows:    {len(output)}")
    print(f"  Survived:      {y_pred.sum()} ({y_pred.mean():.2%})")
    print(f"  Not Survived:  {(y_pred == 0).sum()} ({(y_pred == 0).mean():.2%})")
    print(f"  Avg prob:      {y_prob.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic ML Pipeline")
    parser.add_argument("--config",  default="config/config.yaml",              help="Path to config YAML")
    parser.add_argument("--data",    default="data/raw/train.csv",               help="Path to training data CSV")
    parser.add_argument("--predict", action="store_true",                        help="Run batch prediction instead of training")
    parser.add_argument("--input",   default="data/raw/test.csv",                help="Input CSV for batch prediction")
    parser.add_argument("--output",  default="models/batch_outputs/predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.predict:
        batch_predict(args.config, args.input, args.output)
    else:
        train(args.config, args.data)
