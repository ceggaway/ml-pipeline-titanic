# Experiment 004 — Model Selection

**Date:** TBD
**Hypothesis:** Gradient boosting models (LightGBM / XGBoost / CatBoost) will outperform Random Forest on this dataset, given their stronger performance on tabular data in practice.

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
| SVM | Non-linear kernel; strong on small datasets; slow on larger |
| XGBoost | Gradient boosting; level-wise growth; different regularisation |
| CatBoost | Gradient boosting; ordered boosting; robust to overfitting |

---

## Results

*(Fill in after running)*

| Model | Hold-out AUC | CV AUC | Precision (Survived) | Recall (Survived) | F1 (Survived) | Accuracy |
|---|---|---|---|---|---|---|
| Random Forest | | | | | | |
| Logistic Regression | | | | | | |
| SVM | | | | | | |
| LightGBM | | | | | | |
| XGBoost | | | | | | |
| CatBoost | | | | | | |

---

## Key findings

*(Fill in after running)*

---

## Decision

**Best model:** TBD
**Carry forward to exp_005 for hyperparameter tuning:** TBD

**Status:** IN PROGRESS
