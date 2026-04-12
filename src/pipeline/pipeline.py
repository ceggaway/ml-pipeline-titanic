"""
pipeline.py — CLI entry point.

Usage:
    # Train and save model
    python -m src.pipeline.pipeline --config config/config.yaml --data data/raw/train.csv

    # Batch predict
    python -m src.pipeline.pipeline --predict \
        --input data/raw/daily_input.csv \
        --output models/batch_outputs/predictions_$(date +%Y%m%d).csv

    # Roll back to a previous model version
    python -m src.pipeline.pipeline --rollback 20260411_153240
"""

import sys
import json
import shutil
import logging
import argparse
from pathlib import Path

from .train import train
from .predict import batch_predict
from .monitoring import setup_logging

logger = logging.getLogger("pipeline")


def rollback(version: str) -> None:
    """Restore a versioned model snapshot to final_model.joblib."""
    setup_logging()
    model_dir      = Path("models")
    versioned_path = model_dir / f"model_{version}.joblib"
    latest_path    = model_dir / "final_model.joblib"
    registry_path  = model_dir / "model_registry.json"

    if not versioned_path.exists():
        logger.error(f"Rollback failed: {versioned_path} not found.")
        sys.exit(1)

    shutil.copy2(versioned_path, latest_path)
    logger.info(f"Rolled back: {versioned_path} → {latest_path}")

    # Update registry status
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
        for entry in registry:
            if entry["version"] == version:
                entry["status"] = "latest"
            elif entry["status"] == "latest":
                entry["status"] = "superseded"
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        logger.info(f"Registry updated — version {version} is now 'latest'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic ML Pipeline")
    parser.add_argument("--config",   default="config/config.yaml",                  help="Path to config YAML")
    parser.add_argument("--data",     default="data/raw/train.csv",                  help="Path to training data CSV")
    parser.add_argument("--predict",  action="store_true",                           help="Run batch prediction")
    parser.add_argument("--input",    default="data/raw/daily_input.csv",            help="Input CSV for batch prediction")
    parser.add_argument("--output",   default="models/batch_outputs/predictions.csv",help="Output CSV path")
    parser.add_argument("--rollback", metavar="VERSION",                             help="Roll back to a versioned model (e.g. 20260411_153240)")
    args = parser.parse_args()

    if args.rollback:
        rollback(args.rollback)
    elif args.predict:
        batch_predict(args.config, args.input, args.output)
    else:
        train(args.config, args.data)
