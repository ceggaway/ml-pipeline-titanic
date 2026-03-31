# Experiment 001 — Baseline Random Forest

**Date:** 2026-03-31
**Hypothesis:** A Random Forest on basic cleaned features (with title, deck, and ticket format extracted) will give us a solid baseline AUC to beat.

---

## What I tried

- Dropped only `PassengerId`; kept `Name`, `Ticket`, `Cabin` and extracted helper features from them
- **Name_title** — extracted title (Mr, Mrs, Miss, Master, etc.)
- **Cabin_deck** — extracted deck letter (A–T); 77% missing → filled as 'Unknown'
- **Ticket_is_numeric** — bool flag: pure numeric vs prefixed ticket
- Imputed `Age` with train median (28.5), `Embarked` with train mode ('S')
- One-hot encoded: `Sex`, `Embarked`, `Name_title`, `Cabin_deck`
- StandardScaler on numerical: `Age`, `Fare`, `SibSp`, `Parch`
- RandomForestClassifier(n_estimators=100, random_state=42) — default hyperparameters
- Evaluation: hold-out test set + Stratified 5-fold CV

---

## Data summary

- 891 rows, 12 columns
- Class imbalance: 62% Not Survived / 38% Survived — moderate, stratified split used
- Nulls: Age 19.9%, Cabin 77.1%, Embarked 0.2%
- No duplicate rows

---

## EDA findings

### Numerical
| Feature | Corr with Survived | Observation |
|---|---|---|
| Fare | +0.257 | Strongest numerical signal — higher fare → higher survival |
| Parch | +0.082 | Slight positive — travelling with parents/children slightly helps |
| Age | −0.077 | Slight negative — older passengers less likely to survive |
| SibSp | −0.035 | Weakest — minimal linear relationship |

Fare is right-skewed (max 512 vs median ~14) — worth log-transforming in future experiments.

### Categorical
- **Sex** — strongest categorical signal. Females survived at ~74%, males at ~19%
- **Pclass** — clear gradient: 1st class ~63%, 2nd ~47%, 3rd ~24%
- **Embarked** — Cherbourg (C) passengers survived most (~55%), likely due to class mix
- **Name_title** — Miss/Mrs/Master had high survival; Mr was lowest (~16%). Title encodes gender + age group
- **Cabin_deck** — D and E decks had highest survival (~75%); Unknown (77% of data) lower; A and T lowest
- **Ticket_is_numeric** — prefixed tickets showed slightly different survival to numeric-only; weak but present signal

---

## Results

| Metric | Value |
|---|---|
| Hold-out ROC-AUC | 0.8282 |
| Hold-out Accuracy | 79% |
| Stratified 5-fold CV AUC | 0.8741 ± 0.0210 |
| Per-fold scores | 0.8605, 0.9075, 0.8447, 0.8803, 0.8775 |

**Classification report (hold-out):**
- Not Survived: precision 0.81, recall 0.85, F1 0.83
- Survived: precision 0.75, recall 0.68, F1 0.71

The model is better at identifying non-survivors than survivors — recall on the Survived class (0.68) is the weakest number. Worth improving in later experiments.

The gap between hold-out AUC (0.828) and CV AUC (0.874) suggests some variance — the hold-out split may have been slightly harder, or the model is mildly overfit. CV is more reliable.

---

## SHAP feature importance

From the beeswarm and bar plots:
- **Sex_male** — dominant feature. Being male strongly pushes prediction toward Not Survived (negative SHAP)
- **Name_title_Mr** — correlated with Sex_male; similarly large negative impact
- **Pclass** — higher class number (3rd) pushes toward Not Survived
- **Fare** — high fare (red) pushes toward Survived
- **Name_title_Miss / Mrs / Master** — positive SHAP; these titles indicate women and children who were prioritised
- **Cabin_deck_Unknown** — large portion of passengers; being unknown slightly negative
- **Age** — older passengers (red) tend toward Not Survived; younger slightly positive
- **Ticket_is_numeric**, **Embarked** — smaller but present contributions

Key insight: Sex and its proxy (title) dominate. Pclass and Fare are the next most informative signals.

---

## Decisions

- **Sex/Title** — keep both; they encode overlapping but complementary signal
- **Cabin_deck** — worth keeping despite 77% missing; the 'Unknown' category itself is informative
- **Fare** — consider log-transform in exp_002; raw distribution is heavily skewed
- **Name_title** — consider grouping rare titles (Don, Jonkheer, Lady, Sir, etc.) into 'Rare' bucket to reduce noise
- **SibSp/Parch** — consider combining into FamilySize + IsAlone as in config.yaml
- **Recall on Survived (0.68)** — investigate class weighting or threshold tuning in exp_002

**Status:** COMPLETE — baseline established. AUC 0.874 (CV). Moving to exp_002 for feature engineering.
