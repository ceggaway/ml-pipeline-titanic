# Experiment Results — Titanic ML Pipeline

Master tracker for all experiment outcomes. One row per experiment variant.

---

## Summary table

| Experiment | Focus | CV AUC | Hold-out AUC | Precision (Survived) | Recall (Survived) | F1 (Survived) | Accuracy |
|---|---|---|---|---|---|---|---|
| exp_001 Baseline | Baseline RF, no balancing | 0.8741 | 0.8282 | 0.75 | 0.68 | 0.71 | 0.79 |
| exp_002 Baseline (no balancing) | Imbalance comparison | 0.8773 | 0.8282 | 0.75 | 0.68 | 0.71 | 0.79 |
| exp_002 class_weight=balanced | Imbalance comparison | 0.8782 | 0.8244 | 0.77 | 0.71 | 0.74 | 0.80 |
| exp_002 Random Oversampling | Imbalance comparison | 0.8741 | 0.8119 | 0.66 | 0.71 | 0.69 | 0.75 |
| exp_002 Random Undersampling | Imbalance comparison | 0.8724 | 0.8237 | 0.63 | 0.77 | 0.69 | 0.74 |
| exp_002 SMOTE | Imbalance comparison | 0.8731 | 0.8230 | 0.69 | 0.74 | 0.71 | 0.77 |
| exp_003 FE + Undersampling | Feature engineering | 0.8727 | 0.8216 | 0.66 | 0.81 | 0.73 | 0.77 |
| exp_004 Random Forest | Model selection | 0.8771 | 0.8232 | 0.64 | 0.78 | 0.71 | 0.75 |
| exp_004 Logistic Regression | Model selection | 0.8723 | 0.8503 | 0.72 | 0.80 | 0.76 | 0.80 |
| exp_004 SVM | Model selection | 0.8681 | 0.8437 | 0.70 | 0.83 | 0.76 | 0.79 |
| exp_004 XGBoost | Model selection | 0.8838 | 0.8255 | 0.69 | 0.84 | 0.76 | 0.79 |
| exp_004 CatBoost | Model selection | 0.8909 | 0.8506 | 0.76 | 0.81 | 0.78 | 0.83 |
| exp_005 RandomizedSearch | Hyperparameter tuning | 0.7902* | 0.8414 | 0.71 | 0.80 | 0.75 | 0.80 |
| exp_005 GridSearch | Hyperparameter tuning | 0.7916* | 0.8428 | 0.71 | 0.80 | 0.75 | 0.79 |
| exp_006 Feature selection (0.5) | Model finalisation | 0.8966 | 0.8482 | 0.74 | 0.78 | 0.76 | 0.81 |
| **exp_006 Tuned threshold (0.46)** | **Final model** | **0.8966** | **0.8482** | **0.73** | **0.84** | **0.78** | **0.82** |

*exp_005 CV score is CV F1 (not AUC) — search was optimised for F1

---

## Key takeaways by experiment

### exp_001 — Baseline
- Established baseline: CV AUC **0.8741**, Survived recall **0.68**
- Sex/Title dominate SHAP. Pclass and Fare next.
- Recall on Survived (0.68) identified as main weakness → addressed in exp_002

### exp_002 — Imbalance Handling
- All methods maintain CV AUC ~0.874 — balancing does not hurt overall discrimination
- `class_weight=balanced` gave best F1 (0.737) and best accuracy (0.80)
- Random Undersampling gave best recall (0.768) at cost of precision and accuracy
- **Chosen method:** Random Undersampling — selected to prioritise recall/F1 balance for downstream experiments

### exp_003 — Feature Engineering (+ Undersampling)
- Recall on Survived improved to **0.81** — best across all RF experiments
- F1 on Survived improved to **0.73** — above both exp_001 (0.71) and exp_002 undersample (0.69)
- Hold-out AUC stable (0.822); CV AUC essentially unchanged (0.8727)
- Feature engineering recovered the F1 lost by undersampling and pushed it above baseline

### exp_004 — Model Selection
- **CatBoost was the clear winner** — best on every metric: CV AUC 0.8909, F1 0.783, AUC 0.8506, Accuracy 0.827
- CV AUC jumped from ~0.874 (RF) to **0.8909** — largest single improvement across all experiments
- Error analysis: 71.8% average error overlap → models fail on same passengers → ensembling not worthwhile
- Hard cases: older, lower-fare, solo passengers — likely a data ceiling, not a fixable feature gap

### exp_005 — Hyperparameter Tuning (CatBoost)
- **Tuning made results worse** — all metrics dropped vs defaults (AUC 0.8506 → 0.8428, F1 0.783 → 0.748)
- Root cause: CatBoost defaults are adaptive for small tabular datasets; forced params over-regularised on 546 rows
- Search optimised on balanced CV folds — params didn't transfer to imbalanced hold-out evaluation
- **Decision: use CatBoost defaults from exp_004**

### exp_006 — Model Finalisation
- **Feature selection:** dropped 8 near-zero features (25 → 17); CV AUC improved slightly (+0.0018)
- **Threshold tuning:** optimal threshold 0.46 (vs default 0.5); F1 improved 0.761 → 0.784, recall 0.783 → 0.841
- **Calibration:** Brier score improvement negligible (+0.0008) — no calibration applied
- **Final model:** CatBoost defaults, 17 features, threshold 0.46

---

## Final model configuration

| Setting | Value |
|---|---|
| Model | CatBoostClassifier(random_state=42, verbose=0) |
| Imbalance method | RandomUnderSampler(random_state=42) |
| Features (17) | Name_title_Mr, Age, Sex_male, Pclass, FarePerPerson, Pclass_x_Fare, LogFare, Fare, Cabin_deck_Unknown, FamilySize, Embarked_S, Name_title_Mrs, Name_title_Miss, Name_title_Rare, Cabin_deck_D, Cabin_deck_E, Embarked_Q |
| Decision threshold | 0.46 |
| CV AUC | 0.8966 |
| Hold-out AUC | 0.8482 |
| F1 (Survived) | 0.784 |
| Recall (Survived) | 0.841 |
| Brier score | 0.1490 |

---

## Metric guide

| Metric | What it measures | When to prioritise |
|---|---|---|
| CV AUC | Ranking ability across thresholds; most stable estimate | Overall model quality |
| Hold-out AUC | Same but on one fixed split; higher variance | Sanity check vs CV |
| Recall (Survived) | Of all actual survivors, how many did the model catch | When missing a survivor is costly |
| Precision (Survived) | Of predicted survivors, how many actually survived | When false alarms are costly |
| F1 (Survived) | Harmonic mean of precision and recall | When you want to balance both |
| Accuracy | Overall correct predictions | Less useful with class imbalance |
