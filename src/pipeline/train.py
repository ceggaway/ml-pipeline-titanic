"""
train.py — Model training entry point.
"""

import sys
import json
import joblib
import logging
import subprocess
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

DRIFT_FEATURES = ["Age", "Fare", "Pclass", "SibSp", "Parch"]


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


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

        from sklearn.metrics import roc_auc_score, f1_score, recall_score
        hold_out_metrics = {
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "f1":        round(f1_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
        }

        # Record training baseline stats for drift detection
        training_pct_survived = float(y_train.mean())
        raw_train = df[df.index.isin(X_train.index)] if hasattr(df, 'index') else df
        feature_stats = {}
        for feat in DRIFT_FEATURES:
            if feat in df.columns:
                feature_stats[f"{feat}_mean"] = round(float(df[feat].dropna().mean()), 4)
                feature_stats[f"{feat}_std"]  = round(float(df[feat].dropna().std()), 4)

        logger.info("Running stratified 5-fold CV...")
        skf = StratifiedKFold(n_splits=config["training"]["cv_folds"], shuffle=True, random_state=42)
        cv_pipe = ImbPipeline([
            ("sampler", RandomUnderSampler(random_state=config["imbalance"]["random_state"])),
            ("clf",     build_model(config)),
        ])
        cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=skf, scoring="roc_auc")
        cv_auc = round(float(cv_scores.mean()), 4)
        logger.info(f"  CV ROC-AUC: {cv_auc:.4f} ± {cv_scores.std():.4f}")
        logger.info(f"  Per-fold:   {[round(s, 4) for s in cv_scores]}")

        # Save artefacts
        model_dir  = Path("models")
        model_dir.mkdir(exist_ok=True)
        version    = datetime.now().strftime("%Y%m%d_%H%M%S")
        git_hash   = _git_commit_hash()

        artefacts = {
            "model":                  model,
            "scaler":                 scaler,
            "train_stats":            train_stats,
            "train_columns":          X_train.columns.tolist(),
            "threshold":              threshold,
            "config":                 config,
            "version":                version,
            "git_hash":               git_hash,
            "training_pct_survived":  training_pct_survived,
            "feature_stats":          feature_stats,
        }

        versioned_path = model_dir / f"model_{version}.joblib"
        latest_path    = model_dir / "final_model.joblib"
        joblib.dump(artefacts, versioned_path)
        joblib.dump(artefacts, latest_path)

        # Update model registry
        registry_path = model_dir / "model_registry.json"
        registry = []
        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)

        registry.append({
            "version":       version,
            "git_hash":      git_hash,
            "trained_on":    data_path,
            "features":      X_train.columns.tolist(),
            "n_features":    X_train.shape[1],
            "threshold":     threshold,
            "cv_auc":        cv_auc,
            "hold_out":      hold_out_metrics,
            "artefact_path": str(versioned_path),
            "status":        "latest",
        })
        # Mark previous entries as superseded
        for entry in registry[:-1]:
            entry["status"] = "superseded"

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        logger.info(f"Model saved → {versioned_path} (and → {latest_path})")
        logger.info(f"Model registry updated → {registry_path}")
        logger.info(f"Version: {version} | Git: {git_hash}")
        logger.info("=== Training completed successfully ===")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
