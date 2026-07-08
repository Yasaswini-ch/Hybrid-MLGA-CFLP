"""
feature_engineering.py
=======================
Transforms raw binary CFLP chromosomes into enriched ML-ready feature vectors
by combining the raw binary facility vector with four problem-specific scalar
aggregates that encode higher-level structural properties of each configuration.

Feature Vector Layout (m + 4 dimensions):
  [0 .. m-1] : Raw binary facility opening decisions y_i ∈ {0, 1}
  [m]        : Active facility count  Σ y_i
  [m+1]      : Total active capacity  Σ s_i * y_i
  [m+2]      : Capacity slack ratio   (Σ s_i * y_i - D) / D
  [m+3]      : Weighted avg. fixed cost  Σ f_i * y_i / max(1, Σ y_i)
"""

import numpy as np
from parser import CFLPDataset


class CFLPFeatureEngineer:
    """
    Transforms binary CFLP chromosomes into enriched ML feature vectors.

    Supports two feature modes:
      - 'raw'  : Only the m binary facility bits
      - 'full' : Raw binary bits + 4 engineered scalar aggregates
    """

    def __init__(self, dataset: CFLPDataset, mode: str = "full"):
        """
        Initializes the feature engineer.

        Args:
            dataset (CFLPDataset): Parsed CFLP instance (provides s_i, f_i, D).
            mode (str): Feature mode — 'raw' (m features) or 'full' (m+4 features).
        """
        if mode not in ("raw", "full"):
            raise ValueError(f"Invalid mode '{mode}'. Choose 'raw' or 'full'.")

        self.dataset = dataset
        self.mode = mode
        self.m = dataset.num_facilities
        self.total_demand = float(np.sum(dataset.demands))

        # Feature dimensionality
        self.n_features = self.m if mode == "raw" else self.m + 4

    def transform_one(self, y: np.ndarray) -> np.ndarray:
        """
        Transforms a single binary chromosome into a feature vector.

        Args:
            y (np.ndarray): Binary facility vector of length m.

        Returns:
            np.ndarray: Feature vector of length n_features.
        """
        y = np.array(y, dtype=np.float64)

        if self.mode == "raw":
            return y.copy()

        # --- Engineered Scalar Features ---

        # Feature 1: Active facility count
        # How many warehouses are open. Ranges from m_min to m.
        active_count = float(np.sum(y))

        # Feature 2: Total active capacity
        # The aggregate storage capacity across all open warehouses.
        total_capacity = float(np.sum(self.dataset.capacities * y))

        # Feature 3: Capacity slack ratio
        # Normalized excess capacity above total customer demand.
        # slack = 0 means capacity exactly equals demand (tight bound)
        # slack > 0 means surplus capacity (loose configuration)
        if self.total_demand > 0.0:
            slack_ratio = (total_capacity - self.total_demand) / self.total_demand
        else:
            slack_ratio = 0.0

        # Feature 4: Weighted average fixed cost
        # Mean fixed opening cost per active warehouse.
        # Low values indicate cost-efficient opening configurations.
        if active_count > 0:
            weighted_avg_fixed = float(np.sum(self.dataset.fixed_costs * y)) / active_count
        else:
            weighted_avg_fixed = 0.0

        # Concatenate raw binary + scalar features
        return np.concatenate([y, [active_count, total_capacity, slack_ratio, weighted_avg_fixed]])

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Transforms a matrix of binary chromosomes into a feature matrix.

        Args:
            Y (np.ndarray): Binary chromosome matrix of shape (N, m).

        Returns:
            np.ndarray: Feature matrix of shape (N, n_features).
        """
        Y = np.array(Y, dtype=np.float64)
        if Y.ndim == 1:
            # Single chromosome passed as 1D array
            return self.transform_one(Y).reshape(1, -1)

        return np.array([self.transform_one(row) for row in Y])

    def get_feature_names(self) -> list:
        """
        Returns human-readable names for each feature dimension.

        Returns:
            list: List of feature name strings.
        """
        names = [f"y_{i}" for i in range(self.m)]
        if self.mode == "full":
            names += ["active_count", "total_capacity", "slack_ratio", "avg_fixed_cost"]
        return names
