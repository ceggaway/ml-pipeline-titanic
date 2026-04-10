#!/bin/bash
INPUT=${1:-data/raw/daily_input.csv}
OUTPUT="models/batch_outputs/predictions_$(date +%Y%m%d_%H%M%S).csv"

python src/pipeline/pipeline.py \
  --predict \
  --config config/config.yaml \
  --input "$INPUT" \
  --output "$OUTPUT"
