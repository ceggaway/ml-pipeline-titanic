"""
pipeline.py — CLI entry point.

Usage:
    # Train and save model
    python -m src.pipeline.pipeline --config config/config.yaml --data data/raw/train.csv

    # Batch predict
    python -m src.pipeline.pipeline --predict \
        --input data/raw/daily_input.csv \
        --output models/batch_outputs/predictions_$(date +%Y%m%d).csv
"""

import argparse
from .train import train
from .predict import batch_predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic ML Pipeline")
    parser.add_argument("--config",  default="config/config.yaml",                  help="Path to config YAML")
    parser.add_argument("--data",    default="data/raw/train.csv",                  help="Path to training data CSV")
    parser.add_argument("--predict", action="store_true",                           help="Run batch prediction")
    parser.add_argument("--input",   default="data/raw/daily_input.csv",            help="Input CSV for batch prediction")
    parser.add_argument("--output",  default="models/batch_outputs/predictions.csv",help="Output CSV path")
    args = parser.parse_args()

    if args.predict:
        batch_predict(args.config, args.input, args.output)
    else:
        train(args.config, args.data)
