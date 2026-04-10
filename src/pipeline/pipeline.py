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
import logging
import yaml
import pandas as pd
from pathlib import Path
from prometheus_client import CollectorRegistry, Gauge, write_to_textfile
from catboost import CatBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import preprocess_train, preprocess_inference


def setup_logging() -> logging.Logger:
    """Set up logger that writes to both terminal and logs/pipeline.log."""
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # File handler — full history
        fh = logging.FileHandler("logs/pipeline.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Console handler — same output as before
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def build_model(config: dict) -> CatBoostClassifier:
    params = config["model"]["params"]
    return CatBoostClassifier(**params)


def evaluate(y_test, y_pred, y_prob, threshold: float = 0.5) -> None:
    logger = logging.getLogger("pipeline")
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Hold-out ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    logger.info(f"Decision threshold: {threshold}")
    report = classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"])
    for line in report.splitlines():
        logger.info(line)


def train(config_path: str, data_path: str) -> None:
    logger = setup_logging()
    logger.info("=== Training started ===")

    try:
        config = load_config(config_path)

        logger.info(f"Loading data from {data_path}...")
        df = load_data(data_path)
        logger.info(f"  Rows: {len(df)} | Columns: {len(df.columns)}")

        logger.info("Preprocessing...")
        X_train, X_test, y_train, y_test, train_stats, scaler = preprocess_train(df, config)
        logger.info(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {X_train.shape[1]}")

        logger.info("Applying undersampling...")
        rus = RandomUnderSampler(random_state=config["imbalance"]["random_state"])
        X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
        logger.info(f"  Resampled train: {len(X_train_res)} rows | Balance: {y_train_res.mean():.2%}")

        logger.info("Training CatBoost...")
        model = build_model(config)
        model.fit(X_train_res, y_train_res)

        threshold = config["model"].get("threshold", 0.5)
        y_prob    = model.predict_proba(X_test)[:, 1]
        y_pred    = (y_prob >= threshold).astype(int)
        evaluate(y_test, y_pred, y_prob, threshold)

        logger.info("Running stratified 5-fold CV (resampling inside each fold)...")
        skf = StratifiedKFold(n_splits=config["training"]["cv_folds"], shuffle=True, random_state=42)
        cv_pipe = ImbPipeline([
            ("sampler", RandomUnderSampler(random_state=config["imbalance"]["random_state"])),
            ("clf",     build_model(config)),
        ])
        cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=skf, scoring="roc_auc")
        logger.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        logger.info(f"  Per-fold:   {[round(s, 4) for s in cv_scores]}")

        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        artefacts = {
            "model":         model,
            "scaler":        scaler,
            "train_stats":   train_stats,
            "train_columns": X_train.columns.tolist(),
            "threshold":     threshold,
            "config":        config,
        }
        model_path = model_dir / "final_model.joblib"
        joblib.dump(artefacts, model_path)
        logger.info(f"Model saved → {model_path}")
        logger.info("=== Training completed successfully ===")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


def batch_predict(config_path: str, input_path: str, output_path: str) -> None:
    logger = setup_logging()
    logger.info("=== Batch prediction started ===")

    registry = CollectorRegistry()
    rows_scored   = Gauge("batch_total_rows",    "Rows scored in this batch",       registry=registry)
    rows_failed   = Gauge("batch_failed_rows",   "Rows that failed prediction",     registry=registry)
    pct_survived  = Gauge("batch_pct_survived",  "Fraction predicted survived",     registry=registry)
    batch_success = Gauge("batch_success",       "1 if batch completed, 0 if not",  registry=registry)

    try:
        logger.info("Loading model artefacts...")
        artefacts     = joblib.load("models/final_model.joblib")
        model         = artefacts["model"]
        scaler        = artefacts["scaler"]
        train_stats   = artefacts["train_stats"]
        train_columns = artefacts["train_columns"]
        threshold     = artefacts["threshold"]
        config        = artefacts["config"]

        logger.info(f"Loading input data from {input_path}...")
        df = load_data(input_path)
        logger.info(f"  Rows: {len(df)}")

        # ── Per-row fault tolerance ───────────────────────────────────────────
        results     = []
        failed_rows = []

        for idx, row in df.iterrows():
            try:
                row_df = pd.DataFrame([row])
                X_row  = preprocess_inference(row_df, config, train_stats, scaler, train_columns)
                prob   = float(model.predict_proba(X_row)[0, 1])
                pred   = int(prob >= threshold)
                entry  = {"prediction": pred, "probability": round(prob, 4),
                          "survived_label": "Survived" if pred == 1 else "Not Survived"}
                if "PassengerId" in df.columns:
                    entry["PassengerId"] = row["PassengerId"]
                results.append(entry)
            except Exception as e:
                logger.warning(f"Row {idx} failed: {e}")
                failed_rows.append({"original_index": idx, "error": str(e), **row.to_dict()})

        # ── Save predictions ──────────────────────────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if results:
            output = pd.DataFrame(results)
            if "PassengerId" in output.columns:
                cols = ["PassengerId"] + [c for c in output.columns if c != "PassengerId"]
                output = output[cols]
            output.to_csv(output_path, index=False)
            logger.info(f"Predictions saved → {output_path}")

        if failed_rows:
            failed_path = Path(output_path).parent / "failed_rows.csv"
            pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
            logger.warning(f"  {len(failed_rows)} rows failed → {failed_path}")

        y_pred = output["prediction"].values
        logger.info(f"  Total rows:    {len(output)}")
        logger.info(f"  Failed rows:   {len(failed_rows)}")
        logger.info(f"  Survived:      {y_pred.sum()} ({y_pred.mean():.2%})")
        logger.info(f"  Not Survived:  {(y_pred == 0).sum()} ({(y_pred == 0).mean():.2%})")
        logger.info(f"  Avg prob:      {output['probability'].mean():.4f}")

        rows_scored.set(len(results))
        rows_failed.set(len(failed_rows))
        pct_survived.set(float(y_pred.mean()))
        batch_success.set(1)
        logger.info("=== Batch prediction completed successfully ===")

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        batch_success.set(0)
        rows_scored.set(0)
        rows_failed.set(0)
        pct_survived.set(0)

    finally:
        write_to_textfile("models/metrics.prom", registry=registry)
        logger.info("Metrics written → models/metrics.prom")


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
