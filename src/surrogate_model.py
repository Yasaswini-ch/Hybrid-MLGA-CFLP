"""
surrogate_model.py
==================
A unified surrogate fitness approximation module supporting three regression
architectures for predicting CFLP transportation costs from binary facility
opening vectors.

Supported model types:
  - 'random_forest'       : sklearn RandomForestRegressor
                            + built-in uncertainty via inter-tree variance
  - 'gradient_boosting'   : sklearn GradientBoostingRegressor (GBM)
  - 'xgboost'             : xgboost XGBRegressor (if installed)
  - 'mlp'                 : sklearn MLPRegressor (neural network)

All models share a common interface:
  - fit(X_train, y_train)
  - predict(X) → y_pred
  - predict_with_uncertainty(X) → (y_pred, sigma)   [RF only; others return sigma=0]
  - save(path) / load(path)
"""

import os
import pickle
import time
from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


class CFLPSurrogateModel:
    """
    Unified surrogate model wrapper for CFLP fitness approximation.
    """

    SUPPORTED_TYPES = ("random_forest", "gradient_boosting", "xgboost", "mlp")

    def __init__(self, model_type: str = "random_forest"):
        """
        Initializes the surrogate with the specified model architecture.

        Args:
            model_type (str): Regressor type — 'random_forest', 'gradient_boosting',
                              'xgboost', or 'mlp'.
        """
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {self.SUPPORTED_TYPES}")

        self.model_type = model_type
        self.model = self._build_model()
        self.is_fitted = False
        self.train_time_sec = 0.0

    def _build_model(self):
        """Instantiates the underlying sklearn/xgboost regressor with research hyperparameters."""

        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=200,    # 200 trees: balances variance reduction and memory
                max_depth=15,        # Sufficient depth for facility interaction patterns
                min_samples_leaf=1,  # Full tree growth on small datasets
                max_features="sqrt", # Standard RF feature subsampling
                random_state=42,
                n_jobs=-1            # Parallelize across all CPU cores
            )

        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=300,    # More trees for sequential boosting
                learning_rate=0.05,  # Slow learning rate → better generalization
                max_depth=6,         # Shallower trees prevent overfitting in boosting
                subsample=0.8,       # Stochastic gradient boosting for regularization
                random_state=42
            )

        elif self.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0        # Suppress XGBoost internal logs
                )
            except ImportError:
                raise ImportError("XGBoost is not installed. Run: pip install xgboost")

        elif self.model_type == "mlp":
            return MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),  # Three hidden layers, narrowing
                activation="relu",
                solver="adam",
                max_iter=1000,
                early_stopping=True,   # Monitor val loss to prevent overfitting
                validation_fraction=0.1,
                random_state=42
            )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the surrogate model on the provided training data.

        Args:
            X_train (np.ndarray): Feature matrix of shape (N_train, n_features).
            y_train (np.ndarray): Target transport cost array of shape (N_train,).
        """
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.train_time_sec = time.time() - t0
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates cost predictions for an array of feature vectors.

        Args:
            X (np.ndarray): Feature matrix of shape (N, n_features).

        Returns:
            np.ndarray: Predicted transport costs of shape (N,).
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate model has not been trained yet. Call fit() first.")
        return self.model.predict(X)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts costs AND provides uncertainty estimates for each prediction.

        For Random Forest: uncertainty = standard deviation of individual tree predictions.
        For other models: uncertainty = 0 (no native uncertainty quantification).

        Args:
            X (np.ndarray): Feature matrix of shape (N, n_features).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                y_pred  — Predicted costs of shape (N,)
                sigma   — Prediction uncertainty (std dev) of shape (N,)
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate model has not been trained yet. Call fit() first.")

        if self.model_type == "random_forest":
            # Collect individual tree predictions: shape (N, n_estimators)
            tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
            # tree_preds shape: (n_estimators, N) → transpose to (N, n_estimators)
            tree_preds = tree_preds.T
            y_pred = np.mean(tree_preds, axis=1)
            sigma = np.std(tree_preds, axis=1)
            return y_pred, sigma
        else:
            # No native uncertainty for GBM / XGBoost / MLP
            y_pred = self.predict(X)
            sigma = np.zeros_like(y_pred)
            return y_pred, sigma

    def save(self, save_path: str) -> None:
        """
        Serializes the trained model to disk via pickle.

        Args:
            save_path (str): Full path to the output .pkl file.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an untrained model.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Surrogate model [{self.model_type}] saved to: {save_path}")

    @classmethod
    def load(cls, load_path: str) -> "CFLPSurrogateModel":
        """
        Loads a serialized surrogate model from disk.

        Args:
            load_path (str): Path to a previously saved .pkl file.

        Returns:
            CFLPSurrogateModel: The loaded, fitted model.
        """
        with open(load_path, "rb") as f:
            model = pickle.load(f)
        print(f"  Surrogate model [{model.model_type}] loaded from: {load_path}")
        return model
