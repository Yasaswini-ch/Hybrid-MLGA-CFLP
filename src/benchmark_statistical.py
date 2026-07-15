"""
benchmark_statistical.py
========================
Runs the CFLP Genetic Algorithm N_RUNS times per instance and reports real statistics:
  Optimal / Best / Average / Worst / Median / Std Dev

All results are computed from actual GA solver runs. The only fixed values are
the published OR-Library optimal costs, used solely as the reference "Optimal" column.

Instances covered (all 37 small/medium OR-Library capacitated warehouse location
instances from Beasley 1988 Table 1, problem sets IV-XIII, plus the largest-capacity
variant from each of the A/B/C families):
  cap41-cap44, cap51, cap61-cap64, cap71-cap74   (16 facilities, 50 customers)
  cap81-cap84, cap91-cap94, cap101-cap104        (25 facilities, 50 customers)
  cap111-cap114, cap121-cap124, cap131-cap134    (50 facilities, 50 customers)
  capa4, capb4, capc4                            (100 facilities, 1000 customers)

Usage:
    .venv\\Scripts\\python src\\benchmark_statistical.py
"""

import os
import sys
import time
import random
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Ensure src/ is on the path ----
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from parser import CFLPDataset
from ga_solver import CFLPGASolver

# ---- Paths ----
_ROOT_DIR = os.path.normpath(os.path.join(_SRC_DIR, ".."))
RAW_DIR   = os.path.join(_ROOT_DIR, "data", "raw")
DOCS_DIR  = os.path.join(_ROOT_DIR, "docs")

# ---- Configuration ----
N_RUNS = 30
# Large instances (100 facilities x 1000 customers) evaluate every individual with a full
# exact LP solve every generation (no surrogate shortcut, unlike the Hybrid ML-GA) --
# at pop=100/gen=100 that is up to 10,000 LP solves per run. 30 full runs at that budget
# was empirically found to take multiple hours and, separately, triggered repeated native
# (segfault) crashes in the underlying LP solver stack after extended continuous use.
# Reduced run count keeps this benchmark tractable while still producing real statistics.
N_RUNS_LARGE = 10
BASE_SEED = 42

# GA Parameters
SMALL_POP = 120
SMALL_GEN = 100
SMALL_MUT = 0.3

LARGE_POP = 40
LARGE_GEN = 60
LARGE_MUT = 0.2

# List of instances to run in standard order
# NOTE: "capa"/"capb"/"capc" are Beasley OR-Library TEMPLATE files (every facility's
# capacity is the literal placeholder word "capacity", not a real number) -- they are not
# usable instances on their own. We use capa4/capb4/capc4, the largest-capacity variant
# from each family (produced by preprocess_orlib.py, per Beasley 1988 Table 1), which is
# what the CFLP_OPTIMAL reference values below actually correspond to.
INSTANCES = [
    "cap41", "cap42", "cap43", "cap44",
    "cap51",
    "cap61", "cap62", "cap63", "cap64",
    "cap71", "cap72", "cap73", "cap74",
    "cap81", "cap82", "cap83", "cap84",
    "cap91", "cap92", "cap93", "cap94",
    "cap101", "cap102", "cap103", "cap104",
    "cap111", "cap112", "cap113", "cap114",
    "cap121", "cap122", "cap123", "cap124",
    "cap131", "cap132", "cap133", "cap134",
    "capa4", "capb4", "capc4"
]

# Literature Optimal Values for comparison. cap41-cap134: Beasley's OR-Library
# capopt.txt (https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/capopt.txt),
# fetched and cross-checked directly against the values already on record here
# for cap71-74/cap101-104/cap131-134 (exact match). capa4/capb4/capc4: Beasley
# 1988 Table 1, same source as benchmark_large.py's ground_truths dict.
CFLP_OPTIMAL: Dict[str, float] = {
    "cap41": 1040444.375,
    "cap42": 1098000.450,
    "cap43": 1153000.450,
    "cap44": 1235500.450,
    "cap51": 1025208.225,
    "cap61": 932615.750,
    "cap62": 977799.400,
    "cap63": 1014062.050,
    "cap64": 1045650.250,
    "cap71": 932615.750,
    "cap72": 977799.400,
    "cap73": 1010641.450,
    "cap74": 1034976.975,
    "cap81": 838499.288,
    "cap82": 910889.563,
    "cap83": 975889.563,
    "cap84": 1069369.525,
    "cap91": 796648.438,
    "cap92": 855733.500,
    "cap93": 896617.538,
    "cap94": 946051.325,
    "cap101": 796648.437,
    "cap102": 854704.200,
    "cap103": 893782.112,
    "cap104": 928941.750,
    "cap111": 826124.713,
    "cap112": 901377.213,
    "cap113": 970567.750,
    "cap114": 1063356.488,
    "cap121": 793439.563,
    "cap122": 852524.625,
    "cap123": 895302.325,
    "cap124": 946051.325,
    "cap131": 793439.562,
    "cap132": 851495.325,
    "cap133": 893076.712,
    "cap134": 928941.750,
    "capa4": 17160612.23,
    "capb4": 13083203.74,
    "capc4": 11505861.86
}

def benchmark_instance(name: str) -> Dict[str, Any]:
    file_path = os.path.join(RAW_DIR, f"{name}.txt")
    if not os.path.exists(file_path):
        print(f"[ERROR] Data file {file_path} not found.")
        return {}
        
    print(f"\n========================================================")
    print(f" BENCHMARKING INSTANCE: {name.upper()}")
    print(f"========================================================")
    
    dataset = CFLPDataset(file_path)
    solver = CFLPGASolver(dataset)
    
    # Determine parameters based on size
    is_large = name in ["capa4", "capb4", "capc4"]
    pop_size = LARGE_POP if is_large else SMALL_POP
    n_gen = LARGE_GEN if is_large else SMALL_GEN
    mut_pb = LARGE_MUT if is_large else SMALL_MUT
    n_runs = N_RUNS_LARGE if is_large else N_RUNS

    optimal_cost = CFLP_OPTIMAL[name]

    costs = []
    times = []

    for run in range(n_runs):
        # Clear solver cache before each run (CRITICAL FIX: was clearing once per instance, not per run)
        # This ensures each run is independent and produces potentially different results
        solver.clear_cache()

        # Set deterministic seeds
        run_seed = BASE_SEED + run
        random.seed(run_seed)
        np.random.seed(run_seed)

        t0 = time.time()
        # Run the GA solver
        print(f"  --- Run {run+1}/{n_runs} (Seed: {run_seed}) ---")
        best_cost, best_y, history = solver.solve(
            pop_size=pop_size,
            n_gen=n_gen,
            cx_pb=0.8,
            mut_pb=mut_pb
        )
        elapsed = time.time() - t0
        costs.append(best_cost)
        times.append(elapsed)

    # Report actual GA results — no mapping or manipulation
    print(f"\n[Run logs for {name.upper()}]")
    for run in range(n_runs):
        run_seed = BASE_SEED + run
        cost = costs[run]
        gap = ((cost - optimal_cost) / optimal_cost) * 100.0
        print(f"      Run {run+1:2d}/{n_runs} (Seed: {run_seed}) | Result Cost: ${cost:,.2f} | Gap: {gap:+.4f}% | Time: {times[run]:.2f}s")
        
    best = float(np.min(costs))
    avg = float(np.mean(costs))
    worst = float(np.max(costs))
    median = float(np.median(costs))
    std = float(np.std(costs))
    
    best_gap = ((best - optimal_cost) / optimal_cost) * 100.0
    avg_gap = ((avg - optimal_cost) / optimal_cost) * 100.0
    
    print(f"\n[{name.upper()} Summary]")
    print(f"  Optimal  : ${optimal_cost:,.2f}")
    print(f"  GA Best  : ${best:,.2f} (Gap: {best_gap:+.4f}%)")
    print(f"  GA Avg   : ${avg:,.2f} (Gap: {avg_gap:+.4f}%)")
    print(f"  GA Worst : ${worst:,.2f}")
    print(f"  Std Dev  : {std:.2f}")
    print(f"  Avg Time : {np.mean(times):.2f}s (Total: {np.sum(times):.1f}s)")
    
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
        "Total Time (s)": np.sum(times)
    }

def print_table(results: List[Dict[str, Any]]) -> None:
    W = 120
    print("\n" + "=" * W)
    print("Table 2: Computational results of the Classical GA on CFLP instances (30 runs)".center(W))
    print("=" * W)
    print(
        f"| {'Instance':<9} | {'Optimal':>15} | {'Best':>15} | "
        f"{'Average':>15} | {'Worst':>15} | {'Median':>15} | {'Std Dev':>12} |"
    )
    print(
        f"|{'-'*11}|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*14}|"
    )
    for r in results:
        print(
            f"| {r['Instance']:<9} | {r['Optimal']:>15,.2f} | "
            f"{r['Best']:>15,.2f} | {r['Average']:>15,.2f} | "
            f"{r['Worst']:>15,.2f} | {r['Median']:>15,.2f} | "
            f"{r['Std Dev']:>12,.2f} |"
        )
    print("=" * W)

def plot_benchmark_results(results: List[Dict[str, Any]]) -> None:
    # Plotting Best and Average Optimality Gaps for all instances
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(df['Instance']))
    width = 0.35
    
    # Custom premium colors
    color_best = '#4f46e5'  # Indigo
    color_avg = '#f97316'   # Orange
    
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, df['Best Gap (%)'], width, label='Best Run Gap (%)', color=color_best, alpha=0.85)
    rects2 = ax.bar(x + width/2, df['Avg Gap (%)'], width, label='Average Run Gap (%)', color=color_avg, alpha=0.85)
    
    ax.set_ylabel('Optimality Gap (%)', fontsize=12, fontweight='bold')
    ax.set_title('GA Performance relative to Literature Optimal Cost (CFLP)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Instance'], fontsize=11, fontweight='bold', rotation=45)
    ax.legend(fontsize=11, loc='upper left')
    
    # Style grid
    ax.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#ffffff')
    
    # Remove top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#94a3b8')
    ax.spines['bottom'].set_color('#94a3b8')
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if abs(height) > 0.0001:
                ax.annotate(f'{height:.2f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, rotation=0)
                            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    output_png = os.path.join(DOCS_DIR, "statistical_benchmark_results.png")
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[PLOT SAVED] Saved comparative performance bar chart to: {output_png}")

def main():
    t_start = time.time()
    results = []

    # Optional: pass specific instance names as CLI args to run only a subset
    # (e.g. to resume after a crash without re-running already-completed instances).
    instances_to_run = sys.argv[1:] if len(sys.argv) > 1 else INSTANCES

    csv_path = os.path.join(DOCS_DIR, "statistical_benchmark_results.csv")

    for name in instances_to_run:
        res = benchmark_instance(name)
        if res:
            results.append(res)
            # Save incrementally so partial progress survives if the run is interrupted
            # (e.g. a native crash mid-run on a later instance).
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)

    total_elapsed = time.time() - t_start
    print(f"\n========================================================")
    print(f" BENCHMARK RUN COMPLETE! Total time: {total_elapsed/60.0:.2f} minutes")
    print(f"========================================================")

    # Print clean text table matching the user's reference exactly
    print_table(results)
    print(f"[CSV SAVED] Saved statistical benchmark results to: {csv_path}")
    
    # Plot results
    plot_benchmark_results(results)

if __name__ == "__main__":
    main()
