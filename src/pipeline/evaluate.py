"""
evaluate.py — Model evaluation, drift detection (output + feature level).
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, classification_report

logger = logging.getLogger("pipeline")


def evaluate(y_test, y_pred, y_prob, threshold: float = 0.5) -> None:
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Hold-out ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    logger.info(f"Decision threshold: {threshold}")
    report = classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"])
    for line in report.splitlines():
        logger.info(line)


def check_output_drift(
    pct_survived: float,
    training_baseline: float,
    threshold: float = 0.15,
) -> bool:
    """Warn if batch survival rate drifts more than threshold from training baseline."""
    drift = abs(pct_survived - training_baseline)
    if drift > threshold:
        logger.warning(
            f"[Drift] Output shift detected: batch={pct_survived:.3f} "
            f"vs baseline={training_baseline:.3f} (drift={drift:.3f} > {threshold})"
        )
        return True
    logger.info(f"[Drift] Output drift check passed: drift={drift:.3f} (threshold={threshold})")
    return False


def check_feature_drift(
    batch_df: pd.DataFrame,
    training_stats: dict,
    features: list[str] = None,
    ks_threshold: float = 0.05,
) -> dict:
    """
    KS test for feature-level distribution shift.

    Parameters
    ----------
    batch_df       : raw input DataFrame for the current batch
    training_stats : dict with keys like 'Age_mean', 'Age_std', 'Fare_mean' etc.
                     generated during training and stored in artefacts
    features       : list of numeric features to check (defaults to key numerics)
    ks_threshold   : p-value threshold below which drift is flagged

    Returns
    -------
    drift_report : dict with per-feature drift results
    """
    if features is None:
        features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]

    drift_report = {}
    any_drift    = False

    for feat in features:
        if feat not in batch_df.columns:
            continue
        batch_vals = batch_df[feat].dropna().values

        mean_key = f"{feat}_mean"
        std_key  = f"{feat}_std"

        if mean_key not in training_stats or std_key not in training_stats:
            continue

        # Reconstruct a reference distribution from training mean/std
        # (approximation — in production you'd store the full training distribution)
        ref_mean = training_stats[mean_key]
        ref_std  = training_stats[std_key]
        ref_vals = np.random.normal(ref_mean, ref_std, size=len(batch_vals))

        ks_stat, p_value = stats.ks_2samp(batch_vals, ref_vals)
        drifted = p_value < ks_threshold

        drift_report[feat] = {
            "ks_statistic": round(ks_stat, 4),
            "p_value":      round(p_value, 4),
            "drifted":      bool(drifted),
            "batch_mean":   round(float(batch_vals.mean()), 4),
            "train_mean":   round(ref_mean, 4),
        }

        if drifted:
            any_drift = True
            logger.warning(
                f"[Drift] Feature '{feat}' drift detected: "
                f"KS={ks_stat:.4f}, p={p_value:.4f} "
                f"(batch_mean={batch_vals.mean():.2f} vs train_mean={ref_mean:.2f})"
            )
        else:
            logger.info(f"[Drift] Feature '{feat}' OK: KS={ks_stat:.4f}, p={p_value:.4f}")

    if not any_drift:
        logger.info("[Drift] Feature drift check passed — no significant drift detected.")

    return drift_report


def save_drift_report(
    output_drift: bool,
    feature_drift: dict,
    pct_survived: float,
    training_baseline: float,
    model_version: str,
    output_dir: str = "models/batch_outputs",
) -> None:
    """Save drift report as JSON artefact alongside predictions."""
    report = {
        "model_version":       model_version,
        "output_drift":        bool(output_drift),
        "batch_pct_survived":  round(pct_survived, 4),
        "training_baseline":   round(training_baseline, 4),
        "feature_drift":       feature_drift,
    }
    path = Path(output_dir) / "drift_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Drift report saved → {path}")
