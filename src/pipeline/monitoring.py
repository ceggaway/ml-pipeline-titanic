"""
monitoring.py — Logging setup and Prometheus metrics writing.
"""

import logging
import time
from pathlib import Path
from prometheus_client import CollectorRegistry, Gauge, Info, write_to_textfile


def setup_logging() -> logging.Logger:
    """Set up logger that writes to both terminal and logs/pipeline.log."""
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        fh = logging.FileHandler("logs/pipeline.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def write_metrics(
    total_rows: int,
    failed_rows: int,
    pct_survived: float,
    success: int,
    model_version: str = "unknown",
    drift_flag: int = 0,
    output_path: str = "models/metrics.prom",
) -> None:
    registry = CollectorRegistry()
    Gauge("batch_total_rows",    "Rows scored in this batch",      registry=registry).set(total_rows)
    Gauge("batch_failed_rows",   "Rows that failed prediction",    registry=registry).set(failed_rows)
    Gauge("batch_pct_survived",  "Fraction predicted survived",    registry=registry).set(pct_survived)
    Gauge("batch_success",       "1 if batch completed, 0 if not", registry=registry).set(success)
    Gauge("batch_drift_flag",    "1 if output drift detected",     registry=registry).set(drift_flag)
    Gauge("batch_timestamp",     "Unix timestamp of batch run",    registry=registry).set(int(time.time()))
    Info("batch_model",          "Model version used for scoring", registry=registry).info({"version": model_version})
    write_to_textfile(output_path, registry=registry)
