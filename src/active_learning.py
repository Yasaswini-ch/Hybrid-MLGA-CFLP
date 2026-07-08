"""
active_learning.py
===================
Surrogate Active Learning Loop.

Iteratively refines the machine learning surrogate models by collecting exact
LP evaluations from Hybrid ML-GA runs, appending them to the training corpus,
and retraining the surrogates.

This resolves the exploration-exploitation challenge by progressively
improving the surrogate's accuracy on the specific regions of the search space
that the Genetic Algorithm actually visits.
"""

import os
import time
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

from parser import CFLPDataset
from dataset_generator import CFLPDatasetGenerator
from feature_engineering import CFLPFeatureEngineer
from training_pipeline import SurrogateTrainingPipeline
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
from evaluation_metrics import compute_regression_metrics


class SurrogateActiveLearner:
    """
    Manages the active learning refinement loop for CFLP surrogates.
    """

    def __init__(self,
                 dataset_path: str,
                 corpus_path: str,
                 model_save_dir: str,
                 feature_mode: str = "full",
                 random_seed: int = 42):
        """
        Args:
            dataset_path (str): Path to the raw CFLP benchmark file (e.g. cap41.txt).
            corpus_path (str): Path to the .npz training corpus file to read and write.
            model_save_dir (str): Directory where trained surrogate models are saved.
            feature_mode (str): Feature mode - 'raw' or 'full'.
            random_seed (int): Reproducibility seed.
        """
        self.dataset_path = dataset_path
        self.corpus_path = corpus_path
        self.model_save_dir = model_save_dir
        self.feature_mode = feature_mode
        self.random_seed = random_seed

        # Parse problem instance
        self.dataset = CFLPDataset(dataset_path)

        # Initialize dataset generator and feature engineer
        self.generator = CFLPDatasetGenerator(self.dataset)
        self.engineer = CFLPFeatureEngineer(self.dataset, mode=feature_mode)

    def run_active_learning(self,
                            n_rounds: int = 3,
                            pop_size: int = 50,
                            n_generations: int = 50,
                            mode: str = "confidence_aware",
                            model_type: str = "random_forest") -> List[Dict[str, Any]]:
        """
        Executes n_rounds of active learning.

        Workflow:
          1. Train the initial surrogate model on the current corpus.
          2. Run the Hybrid ML-GA in confidence-aware mode.
          3. Extract exact LP evaluations executed during the GA (warmup/fallback).
          4. Append new exact samples to the corpus and save to disk.
          5. Retrain the surrogate on the augmented corpus and compute test R².
          6. Repeat.

        Returns:
            List[Dict[str, Any]]: List of dictionary results documenting each round.
        """
        print("\n" + "=" * 90)
        print(f"  STARTING ACTIVE LEARNING PIPELINE: {n_rounds} ROUNDS ON {os.path.basename(self.dataset_path)}")
        print("=" * 90)

        history = []

        # --- Fixed Validation Set: carved out ONCE from the initial corpus, before
        # any training round, and never touched again. Every retraining round's
        # train_test_split previously operated on an ever-growing corpus with a
        # fixed random_state -- the split's random_state was fixed, but a different-
        # sized input array produces a different-composition split each time, so R²
        # across rounds was not actually comparable. Holding out a fixed validation
        # set once and reusing it unchanged fixes this while the training corpus
        # continues to grow with new GA samples as before.
        print("\n[Setup] Carving fixed validation set from initial corpus (one-time only)...")
        X_initial, y_initial = self.generator.load(self.corpus_path)
        X_train_pool, X_val, y_train_pool, y_val = train_test_split(
            X_initial, y_initial, test_size=0.2, random_state=self.random_seed
        )
        self.X_val, self.y_val = X_val, y_val
        validation_path = os.path.join(self.model_save_dir, "validation_set_fixed.npz")
        self.generator.save(X_val, y_val, validation_path)
        print(f"  Fixed validation set: {len(X_val)} samples (persisted at {validation_path}, "
              f"never augmented for the remainder of this run)")

        # Re-save the corpus WITHOUT the held-out validation rows: this is the growable
        # training pool that subsequent rounds append new GA samples to.
        self.generator.save(X_train_pool, y_train_pool, self.corpus_path)
        print(f"  Training pool (growable): {len(X_train_pool)} samples")

        # --- Round 0: Initial Surrogate Training ---
        print("\n[Round 0] Training Initial Surrogate Model...")
        pipeline = SurrogateTrainingPipeline(
            dataset=self.dataset,
            corpus_path=self.corpus_path,
            model_save_dir=self.model_save_dir,
            feature_mode=self.feature_mode,
            random_state=self.random_seed,
            fixed_validation_data=(self.X_val, self.y_val)
        )
        results = pipeline.run(model_types=(model_type,))
        initial_metrics = results[model_type]["metrics"]

        # Use the in-memory fitted model directly rather than reloading from disk --
        # SurrogateTrainingPipeline.run() saves to a FIXED path (surrogate_{model_type}.pkl)
        # every call, so that path would otherwise get overwritten by every subsequent
        # round regardless of whether the new model is better. The in-memory object is
        # unaffected by that; the canonical path is explicitly re-synced below instead.
        surrogate = results[model_type]["surrogate"]

        # Track the best-known model separately from whatever the pipeline most recently
        # trained, and persist it to a STABLE path that later (possibly worse) rounds
        # cannot clobber. This is what makes retraining "adaptive" rather than just
        # "always overwrite with the latest run" -- a round that makes the surrogate
        # worse must not replace a round that made it better.
        best_r2 = initial_metrics["r2"]
        best_surrogate = surrogate
        best_model_path = os.path.join(self.model_save_dir, f"surrogate_{model_type}_best.pkl")
        plain_model_path = os.path.join(self.model_save_dir, f"surrogate_{model_type}.pkl")
        best_surrogate.save(best_model_path)
        # Round 0 has nothing to compare against, so it is always "accepted": the plain
        # path (what every other consumer in the project loads by convention) already
        # matches, since the pipeline just wrote it there -- but re-save explicitly so
        # the invariant "plain path == accepted model" holds from the very first round.
        best_surrogate.save(plain_model_path)

        # Load raw corpus to track exact counts
        X_corpus, y_corpus = self.generator.load(self.corpus_path)

        round_res = {
            "round": 0,
            "corpus_size": len(X_corpus),
            "r2": initial_metrics["r2"],
            "mape": initial_metrics["mape_pct"],
            "mae": initial_metrics["mae"],
            "new_samples_added": 0,
            "model_adopted": True,
            "best_r2_so_far": best_r2,
        }
        history.append(round_res)

        for r in range(1, n_rounds + 1):
            print("\n" + "-" * 90)
            print(f"  ACTIVE LEARNING ROUND {r} / {n_rounds}")
            print("-" * 90)

            # --- Step 1: Run Hybrid GA with Current Surrogate ---
            print(f"  Running Hybrid ML-GA in [{mode}] mode (Pop={pop_size}, Gens={n_generations})...")
            solver = HybridMLGASolver(
                dataset=self.dataset,
                surrogate=surrogate,
                pop_size=pop_size,
                n_generations=n_generations,
                mode=mode,
                uncertainty_threshold_pct=5.0,
                warmup_fraction=0.20,
                random_seed=self.random_seed + r  # Vary seed per round
            )
            ga_result = solver.solve()
            
            # --- Step 2: Extract Exact Evaluations ---
            # Uses extract_training_data_from_ga (hybrid_ga.py) rather than a hand-rolled
            # total-cost-to-transport-cost conversion: this is the single source of truth
            # for that conversion, and it also de-duplicates chromosomes re-evaluated
            # across generations (elitism + population convergence) before they ever
            # reach the corpus.
            exact_logs = ga_result["exact_evaluations_log"]
            print(f"  GA finished. Total exact LP evaluations during search: {len(exact_logs):,}")

            if len(exact_logs) == 0:
                print("  [Warning] No exact evaluations were logged in this round. Skipping augment.")
                round_res = {
                    "round": r,
                    "corpus_size": len(X_corpus),
                    "r2": history[-1]["r2"],
                    "mape": history[-1]["mape"],
                    "mae": history[-1]["mae"],
                    "new_samples_added": 0,
                    "model_adopted": False,
                    "best_r2_so_far": best_r2,
                }
                history.append(round_res)
                continue

            X_new, y_new_transport = extract_training_data_from_ga(ga_result, dataset=self.dataset)

            # --- Step 3: Append to Corpus and Save ---
            print(f"  Appending new evaluations to training corpus...")
            corpus_size_before = len(X_corpus)
            X_corpus, y_corpus = self.generator.append(
                X_existing=X_corpus,
                y_existing=y_corpus,
                X_new=X_new,
                y_new=y_new_transport
            )
            new_samples = len(X_corpus) - corpus_size_before

            # Save augmented corpus
            self.generator.save(X_corpus, y_corpus, self.corpus_path)

            # --- Step 4: Retrain Surrogate Model ---
            # Evaluated against the SAME fixed validation set carved out before round 0
            # (self.X_val, self.y_val), not a fresh split of the now-larger corpus -- this
            # is what makes new_metrics["r2"] directly comparable to best_r2 below.
            print(f"  Retraining surrogate model on augmented corpus...")
            pipeline = SurrogateTrainingPipeline(
                dataset=self.dataset,
                corpus_path=self.corpus_path,
                model_save_dir=self.model_save_dir,
                feature_mode=self.feature_mode,
                random_state=self.random_seed,
                fixed_validation_data=(self.X_val, self.y_val)
            )
            retrain_results = pipeline.run(model_types=(model_type,))
            new_metrics = retrain_results[model_type]["metrics"]
            candidate_surrogate = retrain_results[model_type]["surrogate"]

            # --- Step 5: Validate Improvement Before Adopting ---
            # "Adaptive" retraining must not blindly replace a good model with a worse
            # one just because a round happened to run. Only adopt the newly retrained
            # model if it is not worse (by R²) than the best model seen so far; otherwise
            # keep using the best-known model for the next round's GA, and leave both the
            # stable best_model_path AND the plain (conventionally-named) model file
            # untouched -- pipeline.run() unconditionally wrote a possibly-worse model to
            # plain_model_path already, so on rejection we must overwrite it back with the
            # previous best, otherwise any consumer loading surrogate_{type}.pkl directly
            # would silently receive the rejected model.
            r2_change = new_metrics["r2"] - best_r2
            mape_change = new_metrics["mape_pct"] - history[-1]["mape"]

            if new_metrics["r2"] >= best_r2:
                surrogate = candidate_surrogate
                best_r2 = new_metrics["r2"]
                best_surrogate = candidate_surrogate
                best_surrogate.save(best_model_path)
                best_surrogate.save(plain_model_path)
                adopted = True
                verdict = "ADOPTED (R2 improved or held steady)"
            else:
                surrogate = best_surrogate
                best_surrogate.save(plain_model_path)
                adopted = False
                verdict = f"REJECTED (R2 regressed from {best_r2:.4f} to {new_metrics['r2']:.4f}) -- keeping previous best model"

            round_res = {
                "round": r,
                "corpus_size": len(X_corpus),
                "r2": new_metrics["r2"],
                "mape": new_metrics["mape_pct"],
                "mae": new_metrics["mae"],
                "new_samples_added": new_samples,
                "model_adopted": adopted,
                "best_r2_so_far": best_r2,
            }
            history.append(round_res)

            print(f"\n  [Round {r} Summary]")
            print(f"    Total Corpus Size : {len(X_corpus):,} samples ({new_samples:,} new unique)")
            print(f"    R² Score          : {new_metrics['r2']:.6f}  (change vs. best: {r2_change:+.6f})")
            print(f"    MAPE              : {new_metrics['mape_pct']:.4f}%  (change: {mape_change:+.4f}%)")
            print(f"    MAE               : ${new_metrics['mae']:,.2f}")
            print(f"    Decision          : {verdict}")

        # --- Final Multi-Round Summary Report ---
        print("\n" + "=" * 90)
        print(f"  ACTIVE LEARNING LOOP COMPLETED SUMMARY REPORT")
        print("=" * 90)
        print(f"  {'Round':<6} | {'Corpus Size':>12} | {'New Samples':>12} | {'R² Score':>10} | {'MAPE (%)':>10} | {'Adopted':>8}")
        print("-" * 90)
        for h in history:
            print(f"  {h['round']:<6d} | {h['corpus_size']:>12,d} | {h['new_samples_added']:>12,d} | "
                  f"{h['r2']:>10.6f} | {h['mape']:>9.4f}% | {str(h['model_adopted']):>8}")
        print("=" * 90)
        print(f"  Best model (R²={best_r2:.6f}) persisted at: {best_model_path}")

        return history


def main():
    """Standalone active learning loop runner on cap41.txt."""
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, "..", "data", "raw", "cap41.txt")
    corpus_path = os.path.join(base_dir, "..", "data", "processed", "cflp_dataset.npz")
    model_save_dir = os.path.join(base_dir, "..", "data", "processed")

    learner = SurrogateActiveLearner(
        dataset_path=dataset_path,
        corpus_path=corpus_path,
        model_save_dir=model_save_dir,
        feature_mode="full"
    )

    learner.run_active_learning(
        n_rounds=3,
        pop_size=50,
        n_generations=50,
        mode="confidence_aware",
        model_type="random_forest"
    )


if __name__ == "__main__":
    main()
