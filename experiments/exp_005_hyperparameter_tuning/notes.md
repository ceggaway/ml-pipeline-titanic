# Experiment 005 — Hyperparameter Tuning (CatBoost)

**Date:** 2026-04-01
**Goal:** Tune CatBoost (exp_004 winner) using two-stage search to improve on default params.

---

## Baseline (exp_004 CatBoost defaults)

| Metric | Value |
|---|---|
| CV AUC | 0.8909 |
| Hold-out AUC | 0.8506 |
| F1 (Survived) | 0.783 |
| Recall (Survived) | 0.812 |
| Accuracy | 0.827 |

---

## Search strategy

### Stage 1 — RandomizedSearchCV
- 50 random combinations from wide parameter space (250 fits total)
- Parameters searched: `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`, `border_count`
- Scoring: F1 (Survived)

### Stage 2 — GridSearchCV
- Narrow grid built automatically around stage 1 best values (±1 step per param)
- 81 combinations (405 fits total)
- Same scoring: F1 (Survived)

---

## Results

### Stage 1 — RandomizedSearch best params
| Parameter | Value |
|---|---|
| iterations | 200 |
| depth | 5 |
| learning_rate | 0.05 |
| l2_leaf_reg | 10 |
| border_count | 32 |

### Stage 2 — GridSearch best params
| Parameter | Value |
|---|---|
| iterations | 200 |
| depth | 5 |
| learning_rate | 0.05 |
| l2_leaf_reg | 9 |
| border_count | 32 |

### Tuning progression
| Stage | CV F1 | Hold-out AUC | F1 (Survived) | Recall (Survived) | Accuracy |
|---|---|---|---|---|---|
| Default (exp_004) | - | 0.8506 | 0.783 | 0.812 | 0.827 |
| RandomizedSearch | 0.7902 | 0.8414 | 0.753 | 0.797 | 0.799 |
| GridSearch (final) | 0.7916 | 0.8428 | 0.748 | 0.797 | 0.793 |

---

## Key findings

- **Tuning made results worse across all metrics** — AUC dropped from 0.8506 → 0.8428, F1 from 0.783 → 0.748
- **Root cause:** CatBoost's defaults are adaptive for small tabular datasets. Forcing `learning_rate=0.05` with `l2_leaf_reg=9-10` over-regularised the model on only 546 resampled training rows
- **The search optimised CV F1 on balanced (50/50) folds** — params tuned on balanced data don't always transfer to the imbalanced hold-out evaluation
- **Conclusion:** CatBoost defaults are near-optimal for this dataset size. Hyperparameter tuning confirmed, not improved, the default configuration

## Decision

**Use exp_004 default CatBoost — do not apply tuned params.**

**Status:** COMPLETE
