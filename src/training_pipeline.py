"""
training_pipeline.py
====================
End-to-end surrogate model training orchestrator.

Workflow:
  1. Load the existing ML training corpus (X_raw, y_transport)
  2. Compute total objective costs: y_total = y_transport + Σ f_i * y_i
  3. Apply feature engineering (raw or full mode)
  4. 80/20 train/test split
  5. Train all three surrogate architectures
  6. Evaluate prediction accuracy metrics
  7. Save best-performing model and full comparative report
"""

import os
import time
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

from parser import CFLPDataset
from feature_engineering import CFLPFeatureEngineer
from surrogate_model import CFLPSurrogateModel
from evaluation_metrics import (
    compute_regression_metrics,
    compute_latency_speedup,
    print_metrics_report
)


class SurrogateTrainingPipeline:
    """
    Orchestrates the full surrogate model training, evaluation, and persistence workflow.
    """

    # LP solve time per evaluation for speedup computation (measured from Phase 3 on cap41)
    LP_TIME_PER_EVAL_MS = 12.3

    def __init__(self,
                 dataset: CFLPDataset,
                 corpus_path: str,
                 model_save_dir: str,
                 feature_mode: str = "full",
                 test_size: float = 0.2,
                 random_state: int = 42,
                 fixed_validation_data: Tuple[np.ndarray, np.ndarray] = None):
        """
        Args:
            dataset (CFLPDataset): Parsed CFLP instance (for feature engineering context).
            corpus_path (str): Path to the .npz training corpus file.
            model_save_dir (str): Directory to save trained model .pkl files.
            feature_mode (str): 'raw' or 'full' feature engineering mode.
            test_size (float): Fraction of data for the test split (default 0.20).
                               Ignored when fixed_validation_data is provided.
            random_state (int): Reproducibility seed.
            fixed_validation_data (Tuple[np.ndarray, np.ndarray], optional): (X_raw, y_transport)
                pair to use as a STABLE held-out validation set instead of an internal
                train_test_split. When provided, the entire corpus loaded from corpus_path
                is used as training data, and metrics are computed against this fixed set
                instead. Same raw format as the corpus (X: binary chromosomes, y: transport
                costs) -- feature engineering and total-cost computation are applied to it
                identically to the training corpus. Defaults to None (original behavior).
        """
        self.dataset = dataset
        self.corpus_path = corpus_path
        self.model_save_dir = model_save_dir
        self.feature_mode = feature_mode
        self.test_size = test_size
        self.random_state = random_state
        self.fixed_validation_data = fixed_validation_data

        self.engineer = CFLPFeatureEngineer(dataset, mode=feature_mode)

    def _prepare(self, X_raw: np.ndarray, y_transport: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes total objective costs and applies feature engineering to a raw
        (chromosome, transport_cost) pair. Shared by both the training corpus and
        the fixed validation set so both are transformed identically.
        """
        fixed_costs_per_sample = X_raw @ self.dataset.fixed_costs  # shape (N,)
        y_total = y_transport + fixed_costs_per_sample
        X_features = self.engineer.transform(X_raw)
        return X_features, y_total

    def _load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the raw corpus, computes total objective costs,
        and applies feature engineering.

        Returns:
            Tuple (X_features, y_total):
                X_features — engineered feature matrix, shape (N, n_features)
                y_total    — total objective costs (fixed + transport), shape (N,)
        """
        print("[Pipeline] Loading training corpus...")
        data = np.load(self.corpus_path)
        X_raw = data["X"].astype(np.float64)  # Binary chromosome matrix (N, m)
        y_transport = data["y"].astype(np.float64)  # LP-optimal transport costs (N,)

        print(f"  Corpus: {X_raw.shape[0]:,} samples × {X_raw.shape[1]} facilities")

        # Apply feature engineering
        print(f"  Applying feature engineering (mode='{self.feature_mode}')...")
        X_features, y_total = self._prepare(X_raw, y_transport)
        print(f"  Feature matrix shape: {X_features.shape}")

        return X_features, y_total

    def run(self, model_types: Tuple[str, ...] = ("random_forest", "gradient_boosting", "xgboost")) -> Dict[str, Any]:
        """
        Trains and evaluates all specified surrogate model architectures.

        Args:
            model_types: Tuple of model type strings to train and compare.

        Returns:
            Dict: Results including metrics per model and the best model reference.
        """
        # --- Step 1: Load and Prepare Data ---
        X, y = self._load_and_prepare_data()

        # --- Step 2: Train/Test Split ---
        if self.fixed_validation_data is not None:
            # Stable validation set supplied externally (e.g. by active learning, to make
            # R² comparable across retraining rounds): train on the ENTIRE corpus, evaluate
            # against the fixed set instead of carving out a fresh split each call.
            X_val_raw, y_val_raw = self.fixed_validation_data
            X_test, y_test = self._prepare(X_val_raw.astype(np.float64), y_val_raw.astype(np.float64))
            X_train, y_train = X, y
            print(f"\n[Pipeline] Train: {len(X_train):,} samples | "
                  f"Fixed Validation: {len(X_test):,} samples (external, unchanged)")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            print(f"\n[Pipeline] Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")

        results = {}
        best_model = None
        best_r2 = -float("inf")

        print("\n" + "=" * 64)
        print("  TRAINING & EVALUATING SURROGATE MODELS")
        print("=" * 64)

        for model_type in model_types:
            print(f"\n  >> Training [{model_type}]...")

            surrogate = CFLPSurrogateModel(model_type=model_type)
            surrogate.fit(X_train, y_train)

            y_pred = surrogate.predict(X_test)
            metrics = compute_regression_metrics(y_test, y_pred)
            latency = compute_latency_speedup(surrogate.model, X_test, self.LP_TIME_PER_EVAL_MS)

            print_metrics_report(metrics, latency, model_name=model_type)

            # Save individual model
            save_path = os.path.join(self.model_save_dir, f"surrogate_{model_type}.pkl")
            surrogate.save(save_path)

            results[model_type] = {
                "surrogate": surrogate,
                "metrics": metrics,
                "latency": latency,
                "save_path": save_path,
                "train_time_sec": surrogate.train_time_sec
            }

            # Track best model by R²
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_model = surrogate
                results["best_model_type"] = model_type

        # --- Final Comparative Summary ---
        self._print_comparative_table(results, model_types)

        results["best_model"] = best_model
        results["X_train"] = X_train
        results["X_test"] = X_test
        results["y_train"] = y_train
        results["y_test"] = y_test

        return results

    def _print_comparative_table(self, results: Dict[str, Any], model_types: Tuple[str, ...]) -> None:
        """Prints a side-by-side comparative metrics table for all trained models."""
        print("\n" + "=" * 90)
        print(f"  {'COMPARATIVE SURROGATE MODEL EVALUATION SUMMARY':^86}")
        print("=" * 90)
        print(f"  {'Model Type':<22} | {'R²':>8} | {'MAPE (%)':>10} | {'MAE ($)':>18} | {'Speedup':>10}")
        print("-" * 90)
        for model_type in model_types:
            if model_type in results:
                m = results[model_type]["metrics"]
                l = results[model_type]["latency"]
                marker = " <- BEST" if model_type == results.get("best_model_type") else ""
                print(f"  {model_type:<22} | {m['r2']:>8.6f} | {m['mape_pct']:>9.4f}% | "
                      f"${m['mae']:>17,.2f} | {l['speedup_factor']:>9.1f}x{marker}")
        print("=" * 90)


def main():
    """
    Standalone execution: trains all surrogate models on a cap41 corpus.

    The corpus (data/processed/cflp_dataset.npz) is not shipped -- it is a
    regenerable artifact, not source. If it doesn't exist yet, it is bootstrapped
    here from scratch using the same GA-derived sampling approach as
    benchmark_hybrid_ga.py (HybridMLGASolver with surrogate=None, i.e. every
    candidate evaluated with the exact LP solver and logged), so this script is
    runnable standalone from a clean checkout with no prerequisite steps.
    """
    base_dir = os.path.dirname(__file__)
    raw_path = os.path.join(base_dir, "..", "data", "raw", "cap41.txt")
    corpus_path = os.path.join(base_dir, "..", "data", "processed", "cflp_dataset.npz")
    model_save_dir = os.path.join(base_dir, "..", "data", "processed")

    dataset = CFLPDataset(raw_path)

    if not os.path.exists(corpus_path):
        print(f"[Pipeline] No existing corpus at {corpus_path} -- bootstrapping one "
              f"from scratch via GA-derived sampling (cap41, pop=30, gen=15)...")
        from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
        from dataset_generator import CFLPDatasetGenerator

        bootstrap_ga = HybridMLGASolver(dataset=dataset, surrogate=None,
                                         pop_size=30, n_generations=15, random_seed=42)
        boot_result = bootstrap_ga.solve()
        X, y = extract_training_data_from_ga(boot_result, dataset=dataset)
        CFLPDatasetGenerator(dataset).save(X, y, corpus_path)
        print(f"[Pipeline] Bootstrap corpus saved: {X.shape[0]} unique samples -> {corpus_path}")

    pipeline = SurrogateTrainingPipeline(
        dataset=dataset,
        corpus_path=corpus_path,
        model_save_dir=model_save_dir,
        feature_mode="full"
    )

    results = pipeline.run(model_types=("random_forest", "gradient_boosting", "xgboost"))
    print(f"\n  Best model: [{results['best_model_type']}]")
    return results


if __name__ == "__main__":
    main()
