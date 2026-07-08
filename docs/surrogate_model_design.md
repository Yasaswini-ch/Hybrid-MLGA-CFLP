# Surrogate Model Design Notes: Feature Engineering, Model Selection & Hyperparameters

This document details the engineering specifications, feature design rationale, model selection criteria, and hyperparameter choices for our ML surrogate fitness approximation system.

---

## 1. Feature Engineering Design

### Primary Features: Raw Binary Chromosome
The foundation of our feature vector is the raw binary facility opening vector $\mathbf{y} \in \{0,1\}^m$:

$$\mathbf{f}^{(raw)} = [y_1, y_2, \dots, y_m]$$

Each element $y_i$ directly encodes whether facility $i$ contributes its capacity to the system. This is the primary signal the surrogate must learn from.

**Why binary features work**: The total transportation cost $Z^*(\mathbf{y})$ is a highly structured function of $\mathbf{y}$. When facility $i$ is open ($y_i = 1$), customers near facility $i$ can be routed there cheaply. The surrogate learns these spatial cost correlations from the training data.

### Engineered Scalar Features
We augment the raw binary features with four problem-specific scalar aggregates that capture higher-level structural properties of a chromosome:

| Feature | Formula | Physical Meaning |
| :--- | :---: | :--- |
| **Active Count** | $\sum_{i} y_i$ | Total number of open warehouses |
| **Total Active Capacity** | $\sum_{i} s_i y_i$ | Aggregate system storage capacity |
| **Capacity Slack Ratio** | $(\sum s_i y_i - D) / D$ | Normalized excess capacity above demand |
| **Weighted Avg. Fixed Cost** | $\sum f_i y_i / \max(1, \sum y_i)$ | Mean fixed overhead per open warehouse |

These features provide the model with explicit knowledge of:
- **How many** facilities are open (active count → affects routing complexity)
- **How much** total capacity the system has (slack ratio → identifies capacity-tight vs. loose configurations)
- **How expensive** the opening configuration is (weighted fixed cost → helps predict fixed cost contribution)

### Full Feature Vector
$$\mathbf{f}^{(full)} = [y_1, \dots, y_m, \sum y_i, \sum s_i y_i, \text{slack}, \bar{f}_{open}]$$

**Dimensionality**: $m + 4$ (e.g., 16 + 4 = 20 features for `cap41.txt`)

We will evaluate whether these 4 additional engineered features improve R² meaningfully over the raw binary baseline in `docs/ml_experiments.md`.

---

## 2. Model Selection Criteria & Comparison

### Why Random Forest as the Primary Model
Random Forest was selected as our primary surrogate for four reasons:
1.  **Built-in Uncertainty Quantification**: The variance across tree predictions $\text{Var}(\hat{Z}) = \frac{1}{T-1}\sum_{t=1}^T (\hat{Z}_t - \hat{Z})^2$ provides a native, parameter-free uncertainty score for each prediction. This is essential for our confidence-aware evaluation strategy.
2.  **Excellent Tabular Performance**: Random Forest consistently performs well on structured tabular data with binary features, without requiring normalization or extensive hyperparameter tuning.
3.  **Interpretability**: Feature importance scores from Random Forest reveal which facility indices most strongly influence total cost — a valuable operations research insight.
4.  **Established Baseline**: Our existing `surrogate_rf.pkl` achieved R² = 0.9299 and MAPE = 0.770%, providing a validated starting point.

### Gradient Boosting (XGBoost/sklearn GBM) as Challenger
Gradient Boosting models are known to frequently outperform Random Forest on tabular regression tasks by learning additive ensembles of decision trees sequentially. We evaluate it as a challenger model to determine if higher R² is achievable.

### Multi-Layer Perceptron (Neural Network)
The MLP captures non-linear interaction patterns between facility combinations. With binary inputs, the MLP can learn complex "if facility A and B are open but C is closed, customers in region X face these routing costs" patterns that tree-based models may miss. However, it requires more training data and careful hyperparameter tuning.

---

## 3. Hyperparameter Specifications

### Random Forest
| Hyperparameter | Value | Rationale |
| :--- | :---: | :--- |
| `n_estimators` | 200 | 200 trees balance variance reduction vs. memory |
| `max_depth` | 15 | Sufficient depth to capture complex facility interaction patterns |
| `min_samples_leaf` | 1 | No pruning — small dataset allows full growth |
| `max_features` | `"sqrt"` | Default RF feature subsetting for diversity |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | All CPU cores |

### Gradient Boosting
| Hyperparameter | Value | Rationale |
| :--- | :---: | :--- |
| `n_estimators` | 300 | More trees for sequential ensemble |
| `learning_rate` | 0.05 | Slow learning rate with more trees → better generalization |
| `max_depth` | 6 | Shallower trees prevent overfitting in boosting |
| `subsample` | 0.8 | Stochastic gradient boosting for regularization |
| `random_state` | 42 | Reproducibility |

### Multi-Layer Perceptron
| Hyperparameter | Value | Rationale |
| :--- | :---: | :--- |
| `hidden_layer_sizes` | (128, 64, 32) | Three hidden layers with decreasing width |
| `activation` | `"relu"` | Rectified linear units for non-linear interactions |
| `solver` | `"adam"` | Adaptive learning rate for small datasets |
| `max_iter` | 1000 | Sufficient epochs for convergence |
| `early_stopping` | True | Prevent overfitting via validation loss monitoring |
| `random_state` | 42 | Reproducibility |

---

## 4. Confidence-Aware Evaluation Thresholds

For the Random Forest surrogate's confidence-aware mode in `hybrid_ga.py`:

| Parameter | Value | Meaning |
| :--- | :---: | :--- |
| `uncertainty_threshold` | 5% of mean predicted cost | Fall back to exact LP if inter-tree σ² > threshold |
| `elite_fraction` | Top 10% of population | Always exact-evaluate the best-predicted individuals |
| `warmup_generations` | First 20% of generations | Use exact LP for all individuals in early generations to build reliable elite set |

### Scientific Rationale for Warmup Period
In the early generations of a GA, the population is highly diverse and the GA explores broadly. The surrogate's accuracy is lowest precisely in these unexplored regions. Running exact LP for the first 20% of generations:
- Builds an elite individual database with verified exact costs
- Allows the surrogate to be applied with higher confidence in later (exploitation) generations
- Prevents the GA from being misled by surrogate errors during critical early exploration

---

## 5. Model Validation Strategy

### Train/Test Split: 80/20 with Stratification Proxy
Since our training data consists of (chromosome, cost) pairs, we use a standard 80/20 random split. For the full enumeration corpus (2,517 samples), this yields:
- Training set: 2,013 samples
- Test set: 504 samples

### Key Accuracy Targets
Based on our existing baseline (R² = 0.9299, MAPE = 0.770%), we target:

| Metric | Existing Baseline | Phase 4 Target |
| :--- | :---: | :---: |
| R² Score | 0.9299 | > **0.95** |
| MAPE | 0.770% | < **0.5%** |
| MAE (relative) | ~$30M | < **$20M** |
| Prediction Latency | - | < **0.1 ms** |

Improvements will come from:
1. GA-derived samples enriching the training corpus beyond full enumeration
2. Engineered scalar features providing explicit structural signals
3. Gradient Boosting potentially outperforming Random Forest on this dataset
