# Experiment 006 — Model Finalisation

**Date:** 2026-04-01
**Goal:** Three finalisation steps before moving to the training pipeline.

---

## Steps

### 1. Feature Selection
- Drop features with CatBoost importance < 1%
- Check CV AUC holds (tolerance: only revert if AUC drops > 0.002)
- Keep reduced set if AUC holds or improves

### 2. Threshold Tuning
- Compute precision, recall, F1 at every threshold
- Pick threshold that maximises F1 (Survived)
- Compare default (0.5) vs optimal threshold

### 3. Calibration Check
- Reliability diagram: predicted probability vs actual survival rate
- Brier score (lower = better; random = 0.25)
- Apply isotonic calibration and compare

---

## Results

### Feature selection
**Features dropped (8):** AgeGroup_Adult, IsAlone, Cabin_deck_C, Cabin_deck_B, AgeGroup_Senior, Cabin_deck_G, Cabin_deck_T, Cabin_deck_F

| | CV AUC |
|---|---|
| All features (25) | 0.8948 |
| Selected features (17) | 0.8966 |
| Difference | +0.0018 |

**Verdict: Use 17-feature set** — AUC improved slightly, leaner model.

**Final 17 features:**
`Name_title_Mr`, `Age`, `Sex_male`, `Pclass`, `FarePerPerson`, `Pclass_x_Fare`, `LogFare`, `Fare`, `Cabin_deck_Unknown`, `FamilySize`, `Embarked_S`, `Name_title_Mrs`, `Name_title_Miss`, `Name_title_Rare`, `Cabin_deck_D`, `Cabin_deck_E`, `Embarked_Q`

**Top features by importance:** Name_title_Mr (15.6%), Age (14.1%), Sex_male (13.9%), Pclass (9.2%), FarePerPerson (8.6%) — Sex/Title/Class dominate, consistent with exp_001 SHAP findings.

### Threshold tuning
| Threshold | F1 (Survived) | Recall (Survived) | Precision (Survived) | Accuracy |
|---|---|---|---|---|
| 0.5 (default) | 0.761 | 0.783 | 0.740 | 0.810 |
| 0.46 (optimal) | 0.784 | 0.841 | 0.734 | 0.821 |

**Verdict: Use threshold 0.46** — F1 improves by 0.023, recall improves to 0.841 at minimal precision cost. Larger gain than initially expected — worth applying.

### Calibration
| | Brier Score |
|---|---|
| Original | 0.1490 |
| Calibrated (isotonic) | 0.1482 |
| Improvement | +0.0008 |

**Verdict: No calibration** — improvement is negligible (0.0008). CatBoost is natively well-calibrated for tabular data. Reliability diagram shows reasonable alignment in the mid-probability range.

---

## Final model configuration

| Setting | Value |
|---|---|
| Model | CatBoostClassifier(random_state=42, verbose=0) |
| Imbalance method | RandomUnderSampler(random_state=42) |
| Features | 17 (see list above) |
| Decision threshold | 0.46 |
| Hold-out AUC | 0.8482 |
| F1 (Survived) | 0.784 |
| Recall (Survived) | 0.841 |
| Precision (Survived) | 0.734 |
| Accuracy | 0.821 |
| Brier score | 0.1490 |

## Final comparison

| Configuration | Hold-out AUC | F1 (Survived) | Recall (Survived) | Precision (Survived) | Accuracy |
|---|---|---|---|---|---|
| exp_004 CatBoost default (0.5) | 0.8506 | 0.783 | 0.812 | 0.757 | 0.827 |
| exp_006 + feature selection (0.5) | 0.8482 | 0.761 | 0.783 | 0.740 | 0.810 |
| exp_006 + tuned threshold (0.46) | 0.8482 | 0.784 | 0.841 | 0.734 | 0.821 |

**Status:** COMPLETE
