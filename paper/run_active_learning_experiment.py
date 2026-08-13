"""
run_active_learning_experiment.py
===================================
Tests whether the already-implemented, quality-gated active-learning extension
(src/active_learning.py) actually closes the large-instance optimality gap
reported for the Hybrid ML-GA on capa4/capb4/capc4.

For each of the 3 largest instances:
  1. Copy the cached bootstrap corpus (from generate_all_convergence.py) into a
     dedicated active-learning corpus path (never mutate the shared cache).
  2. Run SurrogateActiveLearner.run_active_learning() for N_ROUNDS rounds,
     using the SAME pop/gen budget as the paper's large-instance solve phase
     (pop=100, gen=100), recording each round's R² and adopt/reject decision.
  3. Load the final best-adopted surrogate from disk.
  4. Run N_EVAL_RUNS confidence_aware HybridMLGASolver runs with THAT surrogate
     (same seeds/budget as benchmark_hybrid_ga.py's evaluation phase) to get a
     directly comparable Best/Avg/Worst gap, so pre- vs. post-active-learning
     performance can be reported side by side.
"""
import os
import sys
import shutil
import random
import time
import json

import numpy as np

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from parser import CFLPDataset
from active_learning import SurrogateActiveLearner
from hybrid_ga import HybridMLGASolver
from surrogate_model import CFLPSurrogateModel

_ROOT_DIR = os.path.normpath(os.path.join(_SRC_DIR, ".."))
RAW_DIR = os.path.join(_ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(_ROOT_DIR, "data", "processed")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

AL_MODEL_DIR = os.path.join(PROCESSED_DIR, "active_learning_experiment")
os.makedirs(AL_MODEL_DIR, exist_ok=True)

BASE_SEED = 42
N_ROUNDS = int(os.environ.get("AL_N_ROUNDS", 5))
N_EVAL_RUNS = int(os.environ.get("AL_N_EVAL_RUNS", 10))
SOLVE_POP, SOLVE_GEN = 100, 100
WARMUP_FRACTION = 0.15

CFLP_OPTIMAL = {
    "capa4": 17160612.23,
    "capb4": 13083203.74,
    "capc4": 11505861.86,
}

PRE_AL_BEST_GAP = {"capa4": 18.6533, "capb4": 12.0801, "capc4": 9.8240}
PRE_AL_AVG_GAP = {"capa4": 28.0084, "capb4": 16.7110, "capc4": 14.0591}

INSTANCES = sys.argv[1:] if len(sys.argv) > 1 else ["capa4", "capb4", "capc4"]

results = {}
progress_path = os.path.join(OUT_DIR, "active_learning_progress.json")


def save_progress():
    with open(progress_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    for name in INSTANCES:
        print(f"\n{'='*90}\n ACTIVE LEARNING EXPERIMENT: {name.upper()}\n{'='*90}", flush=True)

        cached_corpus = os.path.join(PROCESSED_DIR, f"convplot_corpus_{name}.npz")
        al_corpus = os.path.join(AL_MODEL_DIR, f"al_corpus_{name}.npz")
        shutil.copyfile(cached_corpus, al_corpus)
        print(f"  Copied cached corpus -> {al_corpus}", flush=True)

        instance_model_dir = os.path.join(AL_MODEL_DIR, name)
        os.makedirs(instance_model_dir, exist_ok=True)

        dataset_path = os.path.join(RAW_DIR, f"{name}.txt")

        t0 = time.time()
        learner = SurrogateActiveLearner(
            dataset_path=dataset_path,
            corpus_path=al_corpus,
            model_save_dir=instance_model_dir,
            random_seed=BASE_SEED,
        )
        history = learner.run_active_learning(
            n_rounds=N_ROUNDS,
            pop_size=SOLVE_POP,
            n_generations=SOLVE_GEN,
            mode="confidence_aware",
            model_type="random_forest",
        )
        al_time = time.time() - t0
        print(f"  Active learning loop finished in {al_time:.1f}s", flush=True)

        # Load the final best-adopted surrogate
        best_model_path = os.path.join(instance_model_dir, "surrogate_random_forest_best.pkl")
        best_surrogate = CFLPSurrogateModel.load(best_model_path)

        # Evaluate this surrogate exactly like benchmark_hybrid_ga.py's phase 2:
        # N_EVAL_RUNS confidence_aware runs, same pop/gen/warmup budget.
        dataset = CFLPDataset(dataset_path)
        optimal_cost = CFLP_OPTIMAL[name]
        costs = []
        for run in range(N_EVAL_RUNS):
            run_seed = BASE_SEED + run
            random.seed(run_seed)
            np.random.seed(run_seed)
            solver = HybridMLGASolver(
                dataset=dataset, surrogate=best_surrogate,
                pop_size=SOLVE_POP, n_generations=SOLVE_GEN,
                mode="confidence_aware", warmup_fraction=WARMUP_FRACTION,
                random_seed=run_seed,
            )
            result = solver.solve()
            costs.append(result["best_cost"])
            print(f"  [Eval run {run+1}/{N_EVAL_RUNS}] cost=${result['best_cost']:,.2f}", flush=True)

        best = float(np.min(costs))
        avg = float(np.mean(costs))
        worst = float(np.max(costs))
        best_gap = (best - optimal_cost) / optimal_cost * 100.0
        avg_gap = (avg - optimal_cost) / optimal_cost * 100.0

        results[name] = {
            "history": history,
            "al_time_sec": al_time,
            "post_al_costs": costs,
            "post_al_best": best,
            "post_al_avg": avg,
            "post_al_worst": worst,
            "post_al_best_gap_pct": best_gap,
            "post_al_avg_gap_pct": avg_gap,
            "pre_al_best_gap_pct": PRE_AL_BEST_GAP[name],
            "pre_al_avg_gap_pct": PRE_AL_AVG_GAP[name],
        }
        save_progress()

        print(f"\n  [{name.upper()} SUMMARY]", flush=True)
        print(f"    Pre-AL  Best Gap: {PRE_AL_BEST_GAP[name]:.4f}% | Avg Gap: {PRE_AL_AVG_GAP[name]:.4f}%", flush=True)
        print(f"    Post-AL Best Gap: {best_gap:.4f}% | Avg Gap: {avg_gap:.4f}%", flush=True)
        print(f"    Final R² (best-so-far): {history[-1]['best_r2_so_far']:.6f}", flush=True)

    print("\nALL DONE.", flush=True)


if __name__ == "__main__":
    main()
