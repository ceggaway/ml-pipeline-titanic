# Commands Reference

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
# Via script
bash scripts/train.sh

# Via Python module directly
python -m src.pipeline.pipeline --config config/config.yaml --data data/raw/train.csv
```

## Batch Prediction
```bash
# Via script (default input: data/raw/daily_input.csv)
bash scripts/batch_predict.sh

# Custom input
bash scripts/batch_predict.sh data/raw/test.csv

# Via Python module directly with timestamp
python -m src.pipeline.pipeline --predict \
    --input data/raw/daily_input.csv \
    --output models/batch_outputs/predictions_$(date +%Y%m%d).csv
```

## Generate Synthetic Data
```bash
python scripts/generate_sample_data.py              # 100 rows → data/raw/daily_input.csv
python scripts/generate_sample_data.py --rows 200   # custom row count
```

## Tests
```bash
# Unit tests (utils.py)
pytest tests/test_pipeline.py -v

# Integration tests (end-to-end pipeline behaviour)
pytest tests/test_integration.py -v

# All tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Git
```bash
git add <files>
git commit -m "message"
git push origin main
```

## Monitoring — Start
```bash
brew services start node_exporter  # exposes metrics at http://localhost:9100
brew services start prometheus      # http://localhost:9090
brew services start grafana         # http://localhost:3000 (login: admin / admin)
```

## Monitoring — Stop
```bash
brew services stop node_exporter
brew services stop prometheus
brew services stop grafana
```

## Monitoring — Check status
```bash
brew services list
```

## Daily Batch (GitHub Actions) — Disable
```bash
# Option 1: via GitHub UI
# Repo → Actions → Daily Batch → ... → Disable workflow

# Option 2: delete the workflow file
rm .github/workflows/daily_batch.yml
git add .github/workflows/daily_batch.yml
git commit -m "remove daily batch workflow"
git push origin main
```
