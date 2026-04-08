# Commands Reference

## Setup
```bash
pip install -r requirements.txt
pip install prometheus-client   # for monitoring
```

## Training
```bash
# Via script
bash scripts/train.sh

# Via Python directly
python src/pipeline/pipeline.py --config config/config.yaml --data data/raw/train.csv
```

## Batch Prediction
```bash
python src/pipeline/pipeline.py --predict \
    --input data/raw/test.csv \
    --output models/batch_outputs/predictions.csv

# With timestamp in output filename
python src/pipeline/pipeline.py --predict \
    --input data/raw/test.csv \
    --output models/batch_outputs/predictions_$(date +%Y%m%d).csv
```

## Tests
```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
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
# Option 1: via GitHub UI
# Repo → Actions → Daily Batch Predict → ... → Disable workflow

# Option 2: delete the workflow file
```bash
rm .github/workflows/daily_batch.yml
git add .github/workflows/daily_batch.yml
git commit -m "remove daily batch workflow"
git push origin main
```
