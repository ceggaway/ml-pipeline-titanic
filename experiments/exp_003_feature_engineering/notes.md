# Experiment 003 — Feature Engineering

**Date:** 2026-03-31
**Hypothesis:** Engineered features (log transforms, family aggregates, interaction terms, title grouping) on top of the exp_002 best method (Random Undersampling) will improve F1 on the Survived class.

---

## What I tried

Same base preprocessing as exp_001/002, plus:

| Feature | Description | Motivation |
|---|---|---|
| `LogFare` | log1p(Fare) | Fare is heavily right-skewed (max 512 vs median 14); log compresses the tail |
| `AgeGroup` | Child (<16) / Adult (16–60) / Senior (>60) | Encodes "women and children first" directly as a categorical signal |
| `FamilySize` | SibSp + Parch + 1 | Combines two weak features; small families survived more than solo or large groups |
| `IsAlone` | 1 if FamilySize == 1 | Binary flag; solo passengers had notably lower survival |
| `Pclass_x_Fare` | Pclass × Fare | Interaction: captures 'expensive relative to class' |
| `FarePerPerson` | Fare / FamilySize | Normalises shared-ticket fares so they're comparable across group sizes |
| Title grouping | Mr/Mrs/Miss/Master kept; all others → 'Rare' | Reduces 17 sparse dummies to 5 clean categories |
| `Ticket_is_numeric` | **Dropped** | Weakest SHAP signal in exp_001; not worth the noise |

**Imbalance method:** Random Undersampling (chosen from exp_002), applied inside CV folds via ImbPipeline.

---

## Results

| Metric | Value |
|---|---|
| Hold-out ROC-AUC | 0.8216 |
| CV ROC-AUC (5-fold) | 0.8727 ± 0.0182 |
| Accuracy | 0.77 |
| Precision (Survived) | 0.66 |
| Recall (Survived) | 0.81 |
| F1 (Survived) | 0.73 |

**Classification report (hold-out):**
- Not Survived: precision 0.86, recall 0.74, F1 0.79
- Survived: precision 0.66, recall 0.81, F1 0.73

---

## Comparison vs prior experiments

| Experiment | CV AUC | Hold-out AUC | Precision (Survived) | Recall (Survived) | F1 (Survived) | Accuracy |
|---|---|---|---|---|---|---|
| exp_001 Baseline | 0.8741 | 0.8282 | 0.75 | 0.68 | 0.71 | 0.79 |
| exp_002 Undersample | 0.8724 | 0.8237 | 0.63 | 0.77 | 0.69 | 0.74 |
| exp_003 FE + Undersample | 0.8727 | 0.8216 | 0.66 | 0.81 | 0.73 | 0.77 |

---

## Key findings

- **Recall on Survived improved significantly**: 0.68 (exp_001) → 0.81 (exp_003) — the feature engineering + undersampling combination is better at catching survivors
- **F1 on Survived improved**: 0.71 (exp_001) → 0.73 (exp_003), and better than exp_002 alone (0.69)
- **Trade-off**: Not Survived recall dropped (0.85 → 0.74) — the model is more aggressive about predicting survival
- **Hold-out AUC is stable**: 0.828 → 0.822 — marginal drop, within noise. CV AUC (more reliable) is essentially unchanged (0.8741 → 0.8727)
- Feature engineering recovered the F1 lost by undersampling in exp_002 and pushed it above the exp_001 baseline

## SHAP findings

*(Fill in after inspecting SHAP plots)*

---

## Decisions

- Feature engineering is worthwhile — F1 on Survived improved without significant AUC loss
- Next steps: hyperparameter tuning (exp_004) — the model is still using default RF params throughout all experiments
- Consider whether `Pclass_x_Fare` and `FarePerPerson` are contributing meaningfully (check SHAP bar plot)

**Status:** COMPLETE
