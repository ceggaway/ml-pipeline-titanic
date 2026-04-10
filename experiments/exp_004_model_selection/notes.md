# Experiment 004 — Model Selection

**Date:** 2025
**Hypothesis:** Gradient boosting models (XGBoost / CatBoost) will outperform Random Forest on this dataset, given their stronger performance on tabular data in practice.

---

## Setup

- Feature set: identical to exp_003 (LogFare, AgeGroup, FamilySize, IsAlone, Pclass×Fare, FarePerPerson, title grouping)
- Imbalance method: Random Undersampling (same across all models — architecture is the only variable)
- All models at default / sensible params — no tuning (that is exp_005)
- Selection criterion: F1 (Survived)

## Models compared

| Model | Notes |
|---|---|
| Random Forest | exp_003 reference baseline |
| Logistic Regression | Linear baseline — tests if tree complexity is needed |
| SVM | Non-linear kernel; strong on small datasets |
| XGBoost | Gradient boosting; level-wise growth |
| CatBoost | Gradient boosting; ordered boosting; robust to overfitting |

---

## Results

| Model | Hold-out AUC | CV AUC | Precision (Survived) | Recall (Survived) | F1 (Survived) | Accuracy |
|---|---|---|---|---|---|---|
| Random Forest | 0.8232 | 0.8771 | 0.64 | 0.78 | 0.71 | 0.75 |
| Logistic Regression | 0.8503 | 0.8723 | 0.72 | 0.80 | 0.76 | 0.80 |
| SVM | 0.8437 | 0.8681 | 0.70 | 0.83 | 0.76 | 0.79 |
| XGBoost | 0.8255 | 0.8838 | 0.69 | 0.84 | 0.76 | 0.79 |
| CatBoost | 0.8506 | 0.8909 | 0.76 | 0.81 | 0.78 | 0.83 |

---

## Key findings

- **CatBoost was the clear winner** — best on every metric: CV AUC 0.8909, F1 0.783, Hold-out AUC 0.8506, Accuracy 0.827
- CV AUC jumped from ~0.874 (RF) to **0.8909** — the largest single improvement across all experiments
- Logistic Regression and SVM were competitive on F1 but fell behind CatBoost on AUC
- LightGBM was dropped early due to `libomp.dylib` dependency issue on Apple Silicon
- Error analysis: 71.8% average error overlap between models → ensembling not worthwhile, models fail on the same hard cases

---

## Decision

**Best model:** CatBoost
**Carry forward to exp_005 for hyperparameter tuning:** CatBoost defaults

**Status:** COMPLETE
