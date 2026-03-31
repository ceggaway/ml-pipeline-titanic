# 🧠 End-to-End ML Pipeline

A production-style machine learning pipeline built for learning — covering the full lifecycle from raw data to a monitored, batch-scoring model. Built entirely with free, open-source tools.

> **Current dataset:** Titanic survival prediction  
> **Goal:** Learn the full ML workflow — experimentation → training → batch scoring → monitoring

---

## 📁 Project Structure

```
end_to_end_ml_pipeline/
│
├── data/
│   └── raw/                        # Original, untouched source data
│
├── experiments/                    # Notebook-based experimentation
│   ├── exp_001_baseline/
│   │   ├── notebook.ipynb
│   │   └── notes.md
│   └── experiments_log.md          # Running decision journal — one summary per exp
│
├── config/
│   └── config.yaml                 # Final model + pipeline settings
│
├── src/
│   └── pipeline/
│       ├── pipeline.py             # Main pipeline: train + batch predict
│       └── utils.py                # Helper functions
│
├── models/
│   ├── final_model.joblib          # Saved trained model
│   └── batch_outputs/              # Prediction CSVs from batch runs
│
├── scripts/
│   ├── train.sh                    # Run training pipeline
│   └── batch_predict.sh            # Score a dataset and save predictions
│
├── monitoring/
│   ├── prometheus.yml              # Batch-job metrics scrape config
│   └── grafana/                    # Dashboard configs
│
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Run tests on push / pull request
│       ├── cd.yml                  # Release updated batch pipeline on merge to main
│       └── daily_batch.yml         # Cron job — score data and update monitoring daily
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔄 Workflow

Three phases with deliberate gates between each. The gates force you to be explicit about when something is ready to move forward.

---

### Phase 1 — Experiment 🔬

**Where:** `experiments/exp_00N/`  
**Tools:** VSCode + Jupyter

Each experiment gets its own folder with exactly two files:

- `notebook.ipynb` — all your code, charts, and trying things out
- `notes.md` — what you tried, what you observed, what you decided

Evaluate results manually by reading your metrics and charts directly in the notebook. When an experiment is done, summarise the outcome in `experiments_log.md`.

**You move to Phase 2 when** you are confident in your feature set and model choice.

---

### Phase 2 — Finalise ✅

**Where:** `config/config.yaml`, `src/pipeline/pipeline.py`, `models/`

1. Graduate your conclusions into `config/config.yaml` — features and hyperparameters locked here, not hardcoded in the script
2. Refactor your notebook logic into clean functions in `pipeline.py`
3. Run `bash scripts/train.sh` — the canonical training run
4. Save the fitted pipeline (scaler + model) to `models/final_model.joblib`

**You move to Phase 3 when** your tests pass and you are happy with evaluation results.

---

### Phase 3 — Batch Score & Monitor 📊

No real-time API. The model scores data on a schedule and you monitor the outputs over time.

**Manual batch run:**
```bash
bash scripts/batch_predict.sh data/raw/titanic.csv
# Output → models/batch_outputs/predictions_<timestamp>.csv
```

**Automated daily run** via GitHub Actions (`daily_batch.yml`):
- Triggers on a cron schedule every day at 1am
- Runs `batch_predict.sh` on new data automatically
- Commits output CSV to `models/batch_outputs/`
- Prometheus scrapes metrics, Grafana displays the dashboard

---

## ⚠️ Data Processing — Avoiding Leakage

A common beginner mistake is scaling or imputing before the train/test split. This leaks test information into training and gives falsely optimistic metrics.

**The correct order:**

```
Raw data  (data/raw/)
    ↓
Structural cleaning + feature engineering
    drop bad columns, fix dtypes,
    add FamilySize, IsAlone, etc.
    Safe to do on the full dataset.
    ↓
Train / Test split
    ↓               ↓
  Train           Test
    ↓
Fit imputer + scaler on train only
    ↓               ↓
Transform train   Transform test
using train stats  using same train stats — no leakage
```

Scaling and imputing happen **inside** a sklearn `Pipeline` so the stats are always learned from training data alone:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

preprocessing = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # learns from X_train only
    ("scaler", StandardScaler()),                    # learns from X_train only
])

X_train_ready = preprocessing.fit_transform(X_train)  # learns + applies
X_test_ready  = preprocessing.transform(X_test)        # applies same stats, no leakage
```

The fitted pipeline is saved to `models/final_model.joblib` so at batch scoring time, new data gets the exact same transforms automatically.

---

## 🗂️ config/config.yaml

Where experiment conclusions graduate to. `pipeline.py` reads from here so there are no hardcoded values buried in code.

```yaml
features:
  numeric:
    - Age        # impute: median
    - Fare       # log-transform before scaling
    - SibSp
    - Parch
  categorical:
    - Pclass     # ordinal
    - Sex        # one-hot
    - Embarked   # one-hot, impute: mode
  engineered:
    - FamilySize # SibSp + Parch + 1
    - IsAlone    # 1 if FamilySize == 1
  drop:
    - Name
    - Ticket
    - Cabin
    - PassengerId

model:
  type: RandomForestClassifier
  n_estimators: 200
  max_depth: 8
  min_samples_split: 5
  random_state: 42

training:
  test_size: 0.2
  cv_folds: 5
  scoring: roc_auc
  target: Survived
```

---

## 🗒️ experiments_log.md — How to Use It

This file tracks your **reasoning** — why you made each decision. You will thank yourself for this when you revisit the project later.

```markdown
## exp_002 — Feature engineering
**Date:** 2024-01-20
**Hypothesis:** FamilySize and IsAlone will improve AUC over baseline

**Result:** AUC 0.831 → 0.849 (+0.018)
**Key finding:** IsAlone was the stronger signal; raw SibSp/Parch less useful alone

**Decision:** Keep FamilySize + IsAlone. Drop raw SibSp/Parch.
**Status:** GRADUATED → config.yaml updated
```

---

## 🔁 CI/CD & Automation

Three GitHub Actions workflows:

**ci.yml — runs on every push**
- Installs dependencies
- Runs `pytest tests/`
- Fails the push if any test breaks
- Keeps your main branch always in a working state

**cd.yml — runs on merge to main**
- Installs dependencies and verifies the pipeline runs end-to-end
- Ensures the main branch is always deployable

**daily_batch.yml — runs on a cron schedule**
```yaml
on:
  schedule:
    - cron: "0 1 * * *"   # every day at 1am
```
- Pulls latest data
- Runs `batch_predict.sh` automatically
- Commits output CSV to `models/batch_outputs/`
- Monitoring dashboards update with the new predictions

---

## 📊 Monitoring

Prometheus scrapes metrics exposed by the batch script (prediction counts, score distributions, anomalies). Grafana reads from Prometheus and displays them as a dashboard you can check over time.

Run both locally:

```bash
# Prometheus
prometheus --config.file=monitoring/prometheus.yml

# Grafana (default port 3000)
grafana-server --homepath /usr/share/grafana
```

```
Grafana:    http://localhost:3000
Prometheus: http://localhost:9090
```

---

## ⚙️ Scripts

**scripts/train.sh**
```bash
#!/bin/bash
python src/pipeline/pipeline.py \
  --config config/config.yaml \
  --data data/raw/titanic.csv
```

**scripts/batch_predict.sh**
```bash
#!/bin/bash
python src/pipeline/pipeline.py \
  --predict \
  --input "$1" \
  --output models/batch_outputs/predictions_$(date +%Y%m%d_%H%M%S).csv
```

---

## 🧪 Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

Also runs automatically on every push via `ci.yml`.

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourname/end_to_end_ml_pipeline.git
cd end_to_end_ml_pipeline
pip install -r requirements.txt

# 2. Place raw data
cp titanic.csv data/raw/

# 3. Run training
bash scripts/train.sh

# 4. Run a batch prediction
bash scripts/batch_predict.sh data/raw/titanic.csv

# 5. Start monitoring
prometheus --config.file=monitoring/prometheus.yml
grafana-server --homepath /usr/share/grafana
```

---

## 📦 Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Development | VSCode + Jupyter | IDE and experiment notebooks |
| Pipeline | sklearn Pipeline | Leak-proof preprocessing + model |
| Batch scoring | Python scripts | Offline scoring, CSV output |
| Scheduling | GitHub Actions (cron) | Daily automated batch runs |
| CI/CD | GitHub Actions | Auto-test on every push |
| Metrics collection | Prometheus | Prediction counts, drift signals |
| Dashboards | Grafana | Visualise outputs over time |

All tools are free and open-source.

---

## 📌 Project Status

| Milestone | Status |
|---|---|
| EDA & baseline model | ✅ Complete |
| Feature engineering experiments | 🔄 In progress |
| config.yaml + pipeline.py | ⬜ Planned |
| Batch scoring script | ⬜ Planned |
| GitHub Actions CI/CD + daily batch | ⬜ Planned |
| Prometheus + Grafana monitoring | ⬜ Planned |

---

## 📖 Learning Goals

- [ ] Understand the full ML lifecycle end-to-end
- [ ] Write clean, modular Python (not just notebooks)
- [ ] Avoid data leakage with sklearn Pipelines
- [ ] Automate batch scoring with GitHub Actions
- [ ] Monitor model outputs over time with Prometheus + Grafana
- [ ] Practice software engineering habits: tests, CI/CD
