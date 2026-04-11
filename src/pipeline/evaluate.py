"""
evaluate.py — Model evaluation reporting.
"""

import logging
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


def check_drift(pct_survived: float, training_baseline: float, threshold: float = 0.15) -> bool:
    """
    Warn if batch survival rate drifts more than threshold from training baseline.
    Returns True if drift detected.
    """
    drift = abs(pct_survived - training_baseline)
    if drift > threshold:
        logger.warning(
            f"Drift detected: batch pct_survived={pct_survived:.3f} vs "
            f"training baseline={training_baseline:.3f} (drift={drift:.3f} > {threshold})"
        )
        return True
    logger.info(f"Drift check passed: drift={drift:.3f} (threshold={threshold})")
    return False
