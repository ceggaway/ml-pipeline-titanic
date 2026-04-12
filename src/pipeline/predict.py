"""
predict.py — Batch prediction entry point.
"""

import sys
import joblib
import logging
import pandas as pd
from pathlib import Path

from .io import load_data, validate_schema, log_validation_report
from .evaluate import check_output_drift, check_feature_drift, save_drift_report
from .monitoring import setup_logging, write_metrics
from .utils import preprocess_inference

logger = logging.getLogger("pipeline")


def batch_predict(config_path: str, input_path: str, output_path: str) -> None:
    setup_logging()
    logger.info("=== Batch prediction started ===")

    try:
        logger.info("Loading model artefacts...")
        artefacts             = joblib.load("models/final_model.joblib")
        model                 = artefacts["model"]
        scaler                = artefacts["scaler"]
        train_stats           = artefacts["train_stats"]
        train_columns         = artefacts["train_columns"]
        threshold             = artefacts["threshold"]
        config                = artefacts["config"]
        version               = artefacts.get("version", "unknown")
        training_pct_survived = artefacts.get("training_pct_survived", None)
        feature_stats         = artefacts.get("feature_stats", {})
        logger.info(f"  Model version: {version}")

        logger.info(f"Loading input data from {input_path}...")
        df = load_data(input_path)
        logger.info(f"  Rows: {len(df)}")

        # ── Schema validation ─────────────────────────────────────────────────
        errors, warnings = validate_schema(df)
        log_validation_report(errors, warnings)
        if errors:
            logger.error("Schema validation failed — halting pipeline.")
            write_metrics(0, 0, 0.0, 0, version, drift_flag=0)
            sys.exit(1)

        # ── Per-row prediction with fault tolerance ───────────────────────────
        results     = []
        failed_rows = []

        for idx, row in df.iterrows():
            try:
                row_df = pd.DataFrame([row])
                X_row  = preprocess_inference(row_df, config, train_stats, scaler, train_columns)
                prob   = float(model.predict_proba(X_row)[0, 1])
                pred   = int(prob >= threshold)
                entry  = {
                    "prediction":     pred,
                    "probability":    round(prob, 4),
                    "survived_label": "Survived" if pred == 1 else "Not Survived",
                }
                if "PassengerId" in df.columns:
                    entry["PassengerId"] = row["PassengerId"]
                results.append(entry)
            except Exception as e:
                logger.warning(f"Row {idx} failed: {e}")
                failed_rows.append({"original_index": idx, "error": str(e), **row.to_dict()})

        # ── Save predictions ──────────────────────────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if not results:
            logger.error("All rows failed — no predictions to save.")
            write_metrics(0, len(failed_rows), 0.0, 0, version, drift_flag=0)
            sys.exit(1)

        output = pd.DataFrame(results)
        if "PassengerId" in output.columns:
            cols   = ["PassengerId"] + [c for c in output.columns if c != "PassengerId"]
            output = output[cols]
        output.to_csv(output_path, index=False)
        logger.info(f"Predictions saved → {output_path}")

        if failed_rows:
            failed_path = Path(output_path).parent / "failed_rows.csv"
            pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
            logger.warning(f"  {len(failed_rows)} rows failed → {failed_path}")

        y_pred       = output["prediction"].values
        pct_survived = float(y_pred.mean())

        logger.info(f"  Total rows:    {len(output)}")
        logger.info(f"  Failed rows:   {len(failed_rows)}")
        logger.info(f"  Survived:      {y_pred.sum()} ({pct_survived:.2%})")
        logger.info(f"  Not Survived:  {(y_pred == 0).sum()} ({(1 - pct_survived):.2%})")
        logger.info(f"  Avg prob:      {output['probability'].mean():.4f}")

        # ── Drift detection ───────────────────────────────────────────────────
        output_drift   = False
        feature_drift  = {}
        output_dir     = str(Path(output_path).parent)

        if training_pct_survived is not None:
            output_drift = check_output_drift(pct_survived, training_pct_survived)

        if feature_stats:
            feature_drift = check_feature_drift(df, feature_stats)

        drift_flag = int(output_drift or any(v.get("drifted") for v in feature_drift.values()))
        save_drift_report(output_drift, feature_drift, pct_survived,
                          training_pct_survived or 0.0, version, output_dir)

        write_metrics(len(results), len(failed_rows), pct_survived, 1, version, drift_flag)
        logger.info("Metrics written → models/metrics.prom")
        logger.info("=== Batch prediction completed successfully ===")

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        write_metrics(0, 0, 0.0, 0, "unknown", drift_flag=0)
        sys.exit(1)
