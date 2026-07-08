"""
benchmark_hybrid_ga.py
=======================
Phase 4: Re-benchmark. Evaluates the CORRECTED Hybrid ML-GA framework
(bootstrap-mode GA-derived training data -> trained surrogate -> confidence_aware
GA using the predicted-cost-vs-current-best decision logic) on the same 15
OR-Library CFLP instances and reference optimals used by benchmark_statistical.py,
so results are directly comparable to the Classical GA baseline already on record
in docs/statistical_benchmark_results.csv.

For each instance:
  1. Run HybridMLGASolver(surrogate=None) once (bootstrap mode) to generate the
     initial GA-derived training corpus (Phase 1).
  2. Train a Random Forest surrogate on that corpus (Phase 1 -> Phase 3 path).
  3. Run HybridMLGASolver(surrogate=<trained model>, mode="confidence_aware")
     N_RUNS times with different seeds (Phase 2 decision logic active), recording
     cost, time, and exact/surrogate evaluation counts per run.
  4. Report Best / Average / Worst / Median / Std Dev / Gap vs. literature optimal,
     matching the Classical GA table's format for direct comparison.

All results are computed from actual runs -- no fabricated or copied numbers.
"""

import os
import sys
import time
import random
from typing import Dict, List, Any

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from parser import CFLPDataset
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
from dataset_generator import CFLPDatasetGenerator
from training_pipeline import SurrogateTrainingPipeline

_ROOT_DIR = os.path.normpath(os.path.join(_SRC_DIR, ".."))
RAW_DIR = os.path.join(_ROOT_DIR, "data", "raw")
DOCS_DIR = os.path.join(_ROOT_DIR, "docs")
PROCESSED_DIR = os.path.join(_ROOT_DIR, "data", "processed")

# ---- Configuration ----
N_RUNS = 10
BASE_SEED = 42

BOOTSTRAP_POP_SMALL = 30
BOOTSTRAP_GEN_SMALL = 15
BOOTSTRAP_POP_LARGE = 60
BOOTSTRAP_GEN_LARGE = 40

SMALL_POP = 120
SMALL_GEN = 100

LARGE_POP = 100
LARGE_GEN = 100

WARMUP_FRACTION = 0.15

INSTANCES = [
    "cap71", "cap72", "cap73", "cap74",
    "cap101", "cap102", "cap103", "cap104",
    "cap131", "cap132", "cap133", "cap134",
    "capa", "capb", "capc",
]

CFLP_OPTIMAL: Dict[str, float] = {
    "cap71": 932615.750,
    "cap72": 977799.400,
    "cap73": 1010641.450,
    "cap74": 1034976.975,
    "cap101": 796648.437,
    "cap102": 854704.200,
    "cap103": 893782.112,
    "cap104": 928941.750,
    "cap131": 793439.562,
    "cap132": 851495.325,
    "cap133": 893076.712,
    "cap134": 928941.750,
    "capa": 17156454.48,
    "capb": 12979071.58,
    "capc": 11505594.33,
}


def benchmark_instance(name: str) -> Dict[str, Any]:
    file_path = os.path.join(RAW_DIR, f"{name}.txt")
    if not os.path.exists(file_path):
        print(f"[ERROR] Data file {file_path} not found.")
        return {}

    print(f"\n========================================================")
    print(f" HYBRID ML-GA BENCHMARK: {name.upper()}")
    print(f"========================================================")

    dataset = CFLPDataset(file_path)
    is_large = name in ("capa", "capb", "capc")
    pop_size = LARGE_POP if is_large else SMALL_POP
    n_gen = LARGE_GEN if is_large else SMALL_GEN
    bootstrap_pop = BOOTSTRAP_POP_LARGE if is_large else BOOTSTRAP_POP_SMALL
    bootstrap_gen = BOOTSTRAP_GEN_LARGE if is_large else BOOTSTRAP_GEN_SMALL
    optimal_cost = CFLP_OPTIMAL[name]

    # --- Step 1: Bootstrap -- generate GA-derived training data (Phase 1) ---
    t_bootstrap0 = time.time()
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    bootstrap_ga = HybridMLGASolver(
        dataset=dataset, surrogate=None,
        pop_size=bootstrap_pop, n_generations=bootstrap_gen,
        random_seed=BASE_SEED
    )
    boot_result = bootstrap_ga.solve()
    bootstrap_time = time.time() - t_bootstrap0

    X, y = extract_training_data_from_ga(boot_result, dataset=dataset)
    corpus_path = os.path.join(PROCESSED_DIR, f"benchmark_corpus_{name}.npz")
    CFLPDatasetGenerator(dataset).save(X, y, corpus_path)
    print(f"  [Bootstrap] {X.shape[0]} unique training samples generated in {bootstrap_time:.1f}s")

    # --- Step 2: Train surrogate on GA-derived data (Phase 1 -> Phase 3 path) ---
    t_train0 = time.time()
    pipeline = SurrogateTrainingPipeline(
        dataset=dataset, corpus_path=corpus_path, model_save_dir=PROCESSED_DIR
    )
    train_results = pipeline.run(model_types=("random_forest",))
    train_time = time.time() - t_train0
    surrogate = train_results["random_forest"]["surrogate"]
    surrogate_r2 = train_results["random_forest"]["metrics"]["r2"]
    print(f"  [Train] R2={surrogate_r2:.4f} in {train_time:.1f}s")

    # --- Step 3: Run the corrected Hybrid GA N_RUNS times (Phase 2 decision logic) ---
    costs = []
    times = []
    exact_counts = []
    surrogate_counts = []

    for run in range(N_RUNS):
        run_seed = BASE_SEED + run
        random.seed(run_seed)
        np.random.seed(run_seed)

        t0 = time.time()
        solver = HybridMLGASolver(
            dataset=dataset, surrogate=surrogate,
            pop_size=pop_size, n_generations=n_gen,
            mode="confidence_aware", warmup_fraction=WARMUP_FRACTION,
            random_seed=run_seed
        )
        print(f"  --- Run {run + 1}/{N_RUNS} (Seed: {run_seed}) ---")
        result = solver.solve()
        elapsed = time.time() - t0

        costs.append(result["best_cost"])
        times.append(elapsed)
        exact_counts.append(result["exact_eval_count"])
        surrogate_counts.append(result["surrogate_eval_count"])

    print(f"\n[Run logs for {name.upper()}]")
    for run in range(N_RUNS):
        run_seed = BASE_SEED + run
        cost = costs[run]
        gap = ((cost - optimal_cost) / optimal_cost) * 100.0
        print(f"      Run {run + 1:2d}/{N_RUNS} (Seed: {run_seed}) | Cost: ${cost:,.2f} | "
              f"Gap: {gap:+.4f}% | Time: {times[run]:.2f}s | "
              f"exact={exact_counts[run]} surr={surrogate_counts[run]}")

    best = float(np.min(costs))
    avg = float(np.mean(costs))
    worst = float(np.max(costs))
    median = float(np.median(costs))
    std = float(np.std(costs))

    best_gap = ((best - optimal_cost) / optimal_cost) * 100.0
    avg_gap = ((avg - optimal_cost) / optimal_cost) * 100.0

    print(f"\n[{name.upper()} Summary]")
    print(f"  Optimal        : ${optimal_cost:,.2f}")
    print(f"  Hybrid Best    : ${best:,.2f} (Gap: {best_gap:+.4f}%)")
    print(f"  Hybrid Avg     : ${avg:,.2f} (Gap: {avg_gap:+.4f}%)")
    print(f"  Hybrid Worst   : ${worst:,.2f}")
    print(f"  Std Dev        : {std:.2f}")
    print(f"  Avg Solve Time : {np.mean(times):.2f}s (Total: {np.sum(times):.1f}s)")
    print(f"  Bootstrap Time : {bootstrap_time:.1f}s | Train Time: {train_time:.1f}s")
    print(f"  Avg Exact Evals: {np.mean(exact_counts):.0f} | Avg Surrogate Evals: {np.mean(surrogate_counts):.0f}")

    return {
        "Instance": name,
        "Optimal": optimal_cost,
        "Best": best,
        "Average": avg,
        "Worst": worst,
        "Median": median,
        "Std Dev": std,
        "Best Gap (%)": best_gap,
        "Avg Gap (%)": avg_gap,
        "Avg Solve Time (s)": float(np.mean(times)),
        "Total Solve Time (s)": float(np.sum(times)),
        "Bootstrap Time (s)": bootstrap_time,
        "Train Time (s)": train_time,
        "Surrogate R2": surrogate_r2,
        "Avg Exact Evals": float(np.mean(exact_counts)),
        "Avg Surrogate Evals": float(np.mean(surrogate_counts)),
    }


def print_table(results: List[Dict[str, Any]]) -> None:
    W = 130
    print("\n" + "=" * W)
    print("Hybrid ML-GA Results on CFLP Instances (bootstrap + confidence_aware, N_RUNS runs)".center(W))
    print("=" * W)
    print(
        f"| {'Instance':<9} | {'Optimal':>15} | {'Best':>15} | "
        f"{'Average':>15} | {'Worst':>15} | {'Std Dev':>12} | {'Best Gap%':>10} | {'Avg Gap%':>10} |"
    )
    print("-" * W)
    for r in results:
        print(
            f"| {r['Instance']:<9} | {r['Optimal']:>15,.2f} | "
            f"{r['Best']:>15,.2f} | {r['Average']:>15,.2f} | "
            f"{r['Worst']:>15,.2f} | {r['Std Dev']:>12,.2f} | "
            f"{r['Best Gap (%)']:>9.4f}% | {r['Avg Gap (%)']:>9.4f}% |"
        )
    print("=" * W)


def main():
    t_start = time.time()
    results = []

    instances_to_run = sys.argv[1:] if len(sys.argv) > 1 else INSTANCES

    for name in instances_to_run:
        res = benchmark_instance(name)
        if res:
            results.append(res)
            # Save incrementally so partial progress survives if the run is interrupted
            df = pd.DataFrame(results)
            csv_path = os.path.join(DOCS_DIR, "hybrid_benchmark_results.csv")
            df.to_csv(csv_path, index=False)

    total_elapsed = time.time() - t_start
    print(f"\n========================================================")
    print(f" HYBRID BENCHMARK COMPLETE! Total time: {total_elapsed / 60.0:.2f} minutes")
    print(f"========================================================")

    print_table(results)

    csv_path = os.path.join(DOCS_DIR, "hybrid_benchmark_results.csv")
    print(f"[CSV SAVED] {csv_path}")


if __name__ == "__main__":
    main()
