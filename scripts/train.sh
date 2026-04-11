#!/bin/bash
python -m src.pipeline.pipeline \
  --config config/config.yaml \
  --data data/raw/train.csv
