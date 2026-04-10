# End-to-End ML Pipeline

A production-style machine learning pipeline built for learning — covering the full lifecycle from raw data to a monitored, daily batch-scoring model. Built entirely with free, open-source tools.

> **Dataset:** Titanic survival prediction
> **Goal:** Learn the full ML workflow — experimentation → training → batch scoring → monitoring

---

## Project Structure

```
end_to_end_ml_pipeline/
│
├── data/
│   └── raw/
│       ├── train.csv               # Training data
│       ├── test.csv                # Static test data
│       └── daily_input.csv         # Generated daily by scripts/generate_sample_data.py
│
├── experiments/                    # One folder per experiment
│   ├── exp_001_baseline/
│   ├── exp_002_imbalanced/
│   ├── exp_003_feature_engineering/
│   ├── exp_004_model_selection/
│   ├── exp_005_hyperparameter_tuning/
│   ├── exp_006_model_finalisation/
│   └── results.md                  # Master tracker — all experiment outcomes
│
├── config/
│   └── config.yaml                 # All pipeline settings + experiment lineage
│
├── src/
│   └── pipeline/
│       ├── pipeline.py             # Train + batch predict entry point
│       └── utils.py                # Preprocessing functions
│
├── models/
│   ├── final_model.joblib          # Saved model artefacts (model, scaler, stats, config)
│   ├── metrics.prom                # Prometheus metrics written after each batch run
│   └── batch_outputs/             # Prediction CSVs from each batch run
│
├── scripts/
│   ├── train.sh                    # Run training
│   └── generate_sample_data.py    # Generate synthetic daily input data
│
├── logs/
│   └── pipeline.log                # Timestamped log of every pipeline run
│
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Run tests on every push
│       ├── cd.yml                  # Train + predict on merge to main
│       └── daily_batch.yml         # Cron: generate data → train → predict → commit
│
├── tests/
│   └── test_pipeline.py            # 22 unit tests for utils.py
│
├── COMMANDS.md                     # Quick reference for all commands
├── requirements.txt
└── .gitignore
```

---

## Final Model

| Setting | Value |
|---|---|
| Model | CatBoostClassifier |
| Imbalance handling | RandomUnderSampler |
| Features | 17 (after dropping 8 near-zero importance) |
| CV ROC-AUC | 0.8966 |
| Hold-out AUC | 0.8482 |
| Hold-out Recall | 0.841 |
| Hold-out Precision | 0.73 |
| Hold-out F1 | 0.784 |
| Hold-out Accuracy | 0.82 |
| Decision threshold | 0.46 |
| Calibration | Not applied (negligible improvement) |

---

## Experiment Lineage

| Experiment | Change | Key Result |
|---|---|---|
| exp_001 | Baseline Random Forest | CV AUC 0.8741, Recall 0.68 |
| exp_002 | Imbalance handling | RandomUnderSampler → Recall 0.77 |
| exp_003 | Feature engineering | LogFare, AgeGroup, FamilySize, IsAlone, Pclass×Fare | F1 0.73, Recall 0.81 |
| exp_004 | Model selection | CatBoost wins, CV AUC 0.8909. Ensembling ruled out |
| exp_005 | Hyperparameter tuning | Defaults near-optimal, tuning degraded results |
| exp_006 | Finalisation | 17 features, threshold 0.46, CV AUC 0.8966 |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/ceggaway/end_to_end_ml.git
cd end_to_end_ml
pip install -r requirements.txt

# 2. Train the model
bash scripts/train.sh

# 3. Generate synthetic data and run batch prediction
python scripts/generate_sample_data.py
python src/pipeline/pipeline.py --predict \
    --input data/raw/daily_input.csv \
    --output models/batch_outputs/predictions.csv

# 4. Run tests
pytest tests/ -v
```

---

## Batch Prediction

Each batch run:
- Processes rows individually — if one row fails, the rest complete normally
- Saves predictions to `models/batch_outputs/predictions_<timestamp>.csv`
- Saves any failed rows to `models/batch_outputs/failed_rows.csv`
- Writes metrics to `models/metrics.prom` for Prometheus to scrape
- Logs everything to `logs/pipeline.log`

---

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Every push | Runs `pytest tests/` |
| `cd.yml` | Merge to main | Train → predict → verify output |
| `daily_batch.yml` | 1am UTC daily | Generate data → train → predict → commit output |

---

## Monitoring

Prometheus + Grafana running locally.

```bash
brew services start node_exporter   # http://localhost:9100
brew services start prometheus       # http://localhost:9090
brew services start grafana          # http://localhost:3000
```

Metrics written after each batch run:
- `batch_pct_survived` — fraction predicted survived
- `batch_total_rows` — rows scored
- `batch_failed_rows` — rows that failed
- `batch_success` — 1 if batch completed, 0 if it crashed

---

## Tech Stack

| Layer | Tool |
|---|---|
| Experimentation | VSCode + Jupyter |
| Model | CatBoost |
| Imbalance | imbalanced-learn |
| Preprocessing | scikit-learn |
| Batch scoring | Python CLI |
| Scheduling | GitHub Actions (cron) |
| CI/CD | GitHub Actions |
| Metrics | Prometheus + node_exporter |
| Dashboards | Grafana |
| Logging | Python `logging` module |

---

## Learning Goals

- [x] Understand the full ML lifecycle end-to-end
- [x] Write clean, modular Python (not just notebooks)
- [x] Avoid data leakage in preprocessing and CV
- [x] Automate batch scoring with GitHub Actions
- [x] Monitor model outputs with Prometheus + Grafana
- [x] Practice software engineering: tests, CI/CD, logging, error handling
