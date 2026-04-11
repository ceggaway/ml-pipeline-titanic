"""
train.py — Model training entry point.
"""

import sys
import joblib
import logging
from datetime import datetime
from pathlib import Path

from catboost import CatBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

from .io import load_config, load_data
from .evaluate import evaluate
from .monitoring import setup_logging
from .utils import preprocess_train

logger = logging.getLogger("pipeline")


def build_model(config: dict) -> CatBoostClassifier:
    return CatBoostClassifier(**config["model"]["params"])


def train(config_path: str, data_path: str) -> None:
    setup_logging()
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

        # Record training baseline survival rate for drift detection
        training_pct_survived = float(y_train.mean())

        logger.info("Running stratified 5-fold CV...")
        skf = StratifiedKFold(n_splits=config["training"]["cv_folds"], shuffle=True, random_state=42)
        cv_pipe = ImbPipeline([
            ("sampler", RandomUnderSampler(random_state=config["imbalance"]["random_state"])),
            ("clf",     build_model(config)),
        ])
        cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=skf, scoring="roc_auc")
        logger.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        logger.info(f"  Per-fold:   {[round(s, 4) for s in cv_scores]}")

        # Save artefacts — versioned by timestamp + latest pointer
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        version   = datetime.now().strftime("%Y%m%d_%H%M%S")

        artefacts = {
            "model":                  model,
            "scaler":                 scaler,
            "train_stats":            train_stats,
            "train_columns":          X_train.columns.tolist(),
            "threshold":              threshold,
            "config":                 config,
            "version":                version,
            "training_pct_survived":  training_pct_survived,
        }

        versioned_path = model_dir / f"model_{version}.joblib"
        latest_path    = model_dir / "final_model.joblib"
        joblib.dump(artefacts, versioned_path)
        joblib.dump(artefacts, latest_path)

        logger.info(f"Model saved → {versioned_path} (and → {latest_path})")
        logger.info(f"Model version: {version}")
        logger.info("=== Training completed successfully ===")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
