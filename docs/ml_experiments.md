# ML Surrogate Model Experiments Log

This document records all machine learning surrogate training experiments, model comparison results, prediction error analyses, and feature engineering evaluations for the Hybrid ML-GA research pipeline.

---

## Experiment ML-1: Three-Model Surrogate Comparison on cap41 Full Enumeration Corpus

**Date**: 2026-05-25
**Phase**: Phase 4 — Surrogate Model Training & Evaluation
**Dataset**: `data/processed/cflp_dataset.npz` (full enumeration of `cap41.txt`)
**Feature Mode**: Full (raw binary + 4 engineered scalar features)
**Train/Test Split**: 80% / 20% (2,013 training samples / 504 test samples)

### Training Corpus Characteristics
| Property | Value |
| :--- | :---: |
| Source Problem | `cap41.txt` (m=16 facilities, n=50 customers) |
| Generation Method | Full combinatorial enumeration (all 2,517 feasible configurations) |
| Total Samples | 2,517 |
| Feasibility Filter | $\sum s_i y_i \ge \sum d_j$ (minimum 12 open facilities required) |
| Feature Dimensions | 20 (16 binary + 4 engineered scalar aggregates) |
| Target Variable | Total objective cost $Z = Z_{transport} + \sum f_i y_i$ |

### Surrogate Model Comparison Results

| Model Type | R² Score | MAPE (%) | MAE ($) | RMSE ($) | Speedup vs. LP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (n=200, depth=15) | 0.9363 | 1.1705% | $58,703,412 | $81,650,462 | **50.4x** |
| **Gradient Boosting** (n=300, lr=0.05) | 0.9880 | 0.2359% | $12,507,936 | $35,479,781 | **835.7x** |
| **XGBoost** (n=300, lr=0.05) | **0.9922** | 0.2758% | $14,308,704 | $28,660,483 | **2,810.4x** |

### Scientific Observations

#### 1. XGBoost Achieves Best Overall Performance
XGBoost produced the highest R² (0.9922) and lowest RMSE ($28.7M), beating Gradient Boosting by a ~5% margin. Its MAPE of 0.2758% means the average prediction error is ~$12.1M on a ~$4.4B optimal cost — less than 0.28% of the true value. This is well within acceptable tolerance for GA fitness ranking.

#### 2. Gradient Boosting Achieves Lowest MAE
Gradient Boosting achieved the lowest MAE ($12.5M vs. XGBoost's $14.3M), suggesting slightly better average prediction behavior on this corpus despite a marginally lower R². The difference is minor and both models are viable.

#### 3. Random Forest Is Weakest in Accuracy
Random Forest underperformed significantly compared to the two boosting methods (R² = 0.9363 vs. 0.9880/0.9922, MAPE = 1.17% vs. 0.24%/0.28%). However, Random Forest remains essential for **uncertainty quantification** — its inter-tree variance provides the only native confidence score, enabling the confidence-aware evaluation strategy in `hybrid_ga.py`.

> [!NOTE]
> The relatively lower RF performance is expected: all 2,517 training samples come from the same problem instance (cap41), so the cost landscape has a very specific structure that boosting methods learn to fit more precisely with sequential residual correction. RF's bagging approach is less suited to fitting tight, structured landscapes.

#### 4. Speedup Analysis — A Critical Research Finding

| Model | Latency (ms/eval) | LP Baseline (ms/eval) | Speedup Factor |
| :--- | :---: | :---: | :---: |
| Random Forest | 0.24 ms | 12.3 ms | **50.4x** |
| Gradient Boosting | 0.015 ms | 12.3 ms | **835.7x** |
| XGBoost | 0.0044 ms | 12.3 ms | **2,810.4x** |

XGBoost provides a **2,810x speedup** over the LP solver — transforming a 61-second GA run into an approximately **1.3-second run** while maintaining 99.22% variance explained. This acceleration enables:
- Running 10x larger populations in the same wall-clock time
- Testing 10x more hyperparameter configurations per hour
- Scaling to larger instances (m=50+) that are impractical with LP-based fitness

#### 5. Feature Engineering Impact

The engineered scalar features (active_count, total_capacity, slack_ratio, avg_fixed_cost) were used in this experiment. A follow-up experiment (ML-2) will compare raw binary features vs. full features to quantify their contribution.

### Models Saved
| Model | Path |
| :--- | :--- |
| Random Forest | `data/processed/surrogate_random_forest.pkl` |
| Gradient Boosting | `data/processed/surrogate_gradient_boosting.pkl` |
| XGBoost | `data/processed/surrogate_xgboost.pkl` |

---

## Planned Experiments

| Experiment | Objective | Status |
| :--- | :--- | :---: |
| **ML-1** | 3-model comparison: RF vs. GBM vs. XGBoost | ✅ Complete |
| **ML-2** | Feature ablation: raw binary vs. full features | 🔜 Planned |
| **ML-3** | Active learning rounds: R² improvement per round | 🔜 Planned |
| **ML-4** | Cross-instance transfer: train on cap41, evaluate on cap81 | 🔜 Planned |
