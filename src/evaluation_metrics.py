"""
evaluation_metrics.py
=====================
A unified metrics computation module for evaluating surrogate model prediction
quality and quantifying the computational speedup of the Hybrid ML-GA over the
Classical GA.

Metrics computed:
  - MAE  : Mean Absolute Error (in dollars)
  - RMSE : Root Mean Square Error (in dollars)
  - R²   : Coefficient of determination (proportion of variance explained)
  - MAPE : Mean Absolute Percentage Error (in %)
  - Latency Speedup Factor: LP solve time / surrogate prediction time
"""

import time
from typing import Dict, Tuple
import numpy as np


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes a full suite of regression accuracy metrics comparing surrogate
    predictions to exact LP-solved ground-truth costs.

    Args:
        y_true (np.ndarray): Exact LP-solved objective costs (ground truth).
        y_pred (np.ndarray): Surrogate model cost predictions.

    Returns:
        Dict[str, float]: Dictionary containing all computed metric values.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot compute metrics on an empty array.")
    if n != len(y_pred):
        raise ValueError(f"y_true and y_pred must have the same length: {n} vs {len(y_pred)}")

    # --- Mean Absolute Error (MAE) ---
    # Average absolute dollar-amount difference between predicted and true costs.
    # Lower is better. Units: dollars ($).
    mae = np.mean(np.abs(y_true - y_pred))

    # --- Root Mean Square Error (RMSE) ---
    # Square root of the average squared error. Penalizes large deviations more
    # heavily than MAE. Units: dollars ($).
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # --- R² Score (Coefficient of Determination) ---
    # Measures the proportion of variance in y_true explained by y_pred.
    # R² = 1.0 is perfect; R² = 0.0 means the model is no better than predicting the mean.
    # R² < 0 means the model is worse than the mean — a sign of severe underfitting.
    ss_res = np.sum((y_true - y_pred) ** 2)          # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) # Total sum of squares
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    # --- Mean Absolute Percentage Error (MAPE) ---
    # Average percentage difference, relative to the true value.
    # Critical for cost prediction: a MAPE of 1% on a $4B cost means ~$40M average error.
    # Guard against division by zero for zero-cost configurations (should not occur in CFLP).
    nonzero_mask = y_true != 0.0
    if np.sum(nonzero_mask) == 0:
        mape = 0.0
    else:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape_pct": float(mape),
        "n_samples": int(n)
    }


def compute_latency_speedup(surrogate_model,
                             X_sample: np.ndarray,
                             lp_time_per_eval_ms: float) -> Dict[str, float]:
    """
    Measures the per-prediction latency of the surrogate model and computes
    the speedup factor relative to an exact LP evaluation.

    Args:
        surrogate_model: Trained sklearn-compatible regressor with a predict() method.
        X_sample (np.ndarray): Sample feature matrix to benchmark prediction latency on.
        lp_time_per_eval_ms (float): Measured LP solve time per evaluation in milliseconds.

    Returns:
        Dict[str, float]: Latency and speedup metrics.
    """
    n_warmup = min(10, len(X_sample))
    n_benchmark = len(X_sample)

    # Warmup pass (JIT compilation / caching effects)
    surrogate_model.predict(X_sample[:n_warmup])

    # Timed benchmark pass
    start = time.perf_counter()
    surrogate_model.predict(X_sample)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    surrogate_ms_per_eval = elapsed_ms / n_benchmark
    speedup_factor = lp_time_per_eval_ms / surrogate_ms_per_eval if surrogate_ms_per_eval > 0 else float("inf")

    return {
        "surrogate_ms_per_eval": float(surrogate_ms_per_eval),
        "lp_ms_per_eval": float(lp_time_per_eval_ms),
        "speedup_factor": float(speedup_factor),
        "n_benchmark_samples": int(n_benchmark)
    }


def print_metrics_report(regression_metrics: Dict[str, float],
                          latency_metrics: Dict[str, float] = None,
                          model_name: str = "Surrogate Model") -> None:
    """
    Prints a formatted, publication-quality metrics report to the console.

    Args:
        regression_metrics (Dict[str, float]): Output from compute_regression_metrics().
        latency_metrics (Dict[str, float]): Output from compute_latency_speedup() (optional).
        model_name (str): Display name of the model being reported.
    """
    width = 64
    print("=" * width)
    print(f"  {model_name.upper()} ACCURACY METRICS REPORT")
    print("=" * width)
    print(f"  Evaluation Samples : {regression_metrics['n_samples']:,}")
    print(f"  R² Score           : {regression_metrics['r2']:.6f}  (target: > 0.95)")
    print(f"  MAE                : ${regression_metrics['mae']:>20,.2f}")
    print(f"  RMSE               : ${regression_metrics['rmse']:>20,.2f}")
    print(f"  MAPE               : {regression_metrics['mape_pct']:.4f}%  (target: < 0.5%)")

    if latency_metrics:
        print("-" * width)
        print(f"  LATENCY & EFFICIENCY")
        print(f"  Surrogate Latency  : {latency_metrics['surrogate_ms_per_eval']:.4f} ms / eval")
        print(f"  LP Baseline        : {latency_metrics['lp_ms_per_eval']:.2f} ms / eval")
        print(f"  Speedup Factor     : {latency_metrics['speedup_factor']:,.1f}x")

    print("=" * width)
