import os
import time
import numpy as np
import pandas as pd
from parser import CFLPDataset
from baseline import MILPSolver, GreedySolver
from ga_solver import CFLPGASolver

# ---- Configuration ----
# Unlike benchmark_statistical.py / benchmark_hybrid_ga.py, this script has no
# small-vs-large split: all 12 instances here are the same size (100 facilities,
# 1,000 customers), so a single GA budget applies to every one of them.
MILP_TIMEOUT_SEC = 180
GA_POP = 50
GA_GEN = 50
GA_CX_PB = 0.8
GA_MUT_PB = 0.2

def run_benchmarks():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "..", "data", "raw")
    names = [
        "capa1", "capa2", "capa3", "capa4",
        "capb1", "capb2", "capb3", "capb4",
        "capc1", "capc2", "capc3", "capc4"
    ]
    
    # Ground truth optimal costs from literature (for gap calculation)
    # This helps verify our MILP solver's results too!
    ground_truths = {
        "capa1": 19241056.93,
        "capa2": 18438329.78,
        "capa3": 17765201.95,
        "capa4": 17160612.23,
        "capb1": 13657464.23,
        "capb2": 13362529.34,
        "capb3": 13199213.19,
        "capb4": 13083203.74,
        "capc1": 11647410.50,
        "capc2": 11570437.68,
        "capc3": 11519169.78,
        "capc4": 11505861.86
    }
    
    results = []
    
    print("=" * 110)
    print(f"{'OR-LIBRARY LARGE-SCALE CFLP BENCHMARKS':^110}")
    print("=" * 110)
    
    for name in names:
        file_path = os.path.join(raw_dir, f"{name}.txt")
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}")
            continue
            
        print(f"\n>>> Benchmarking instance: {name} ...")
        dataset = CFLPDataset(file_path)
        
        # 1. Run MILP Solver
        # NOTE: at this scale (100 facilities x 1000 customers), CBC frequently cannot
        # prove optimality within any practical time budget -- this is a normal, expected
        # characteristic of exact MILP solvers on NP-hard problems this size, not a bug.
        # MILPSolver.solve() reports this honestly via the returned status string
        # ("Time Limit (Feasible, Not Proven Optimal)" vs "Optimal") rather than claiming
        # a false optimum.
        print("  - Running MILP solver...")
        t0 = time.time()
        milp = MILPSolver(dataset)
        m_cost, m_y, _, m_status = milp.solve(timeout_sec=MILP_TIMEOUT_SEC)
        m_time = time.time() - t0
        m_active = int(np.sum(m_y))

        # 2. Run Greedy Heuristic
        print("  - Running Greedy solver...")
        t0 = time.time()
        greedy = GreedySolver(dataset)
        g_cost, g_y, _ = greedy.solve()
        g_time = time.time() - t0
        g_active = int(np.sum(g_y))

        # 3. Run Classical GA
        print("  - Running Classical GA...")
        # pop=10/gen=10 was too small a budget for 100-facility instances and occasionally
        # returned the raw infeasibility penalty (1e12) instead of a real solution.
        # pop=50/gen=50 reliably finds a feasible solution while still running quickly.
        t0 = time.time()
        ga = CFLPGASolver(dataset)
        ga_cost, ga_y, ga_history = ga.solve(pop_size=GA_POP, n_gen=GA_GEN, cx_pb=GA_CX_PB, mut_pb=GA_MUT_PB)
        ga_time = time.time() - t0
        ga_active = int(np.sum(ga_y))
        
        # Calculate gaps
        gt = ground_truths[name]
        m_gap = ((m_cost - gt) / gt) * 100.0
        g_gap = ((g_cost - m_cost) / m_cost) * 100.0
        ga_gap = ((ga_cost - m_cost) / m_cost) * 100.0
        
        print(f"  * MILP Cost  : ${m_cost:,.2f} (Opened {m_active}/100, Time: {m_time:.2f}s, GT Gap: {m_gap:.4f}%, Status: {m_status})")
        print(f"  * Greedy Cost: ${g_cost:,.2f} (Opened {g_active}/100, Time: {g_time:.4f}s, Gap: {g_gap:.4f}%)")
        print(f"  * GA Cost    : ${ga_cost:,.2f} (Opened {ga_active}/100, Time: {ga_time:.2f}s, Gap: {ga_gap:.4f}%)")

        results.append({
            "dataset": name,
            "ground_truth": gt,
            "milp_cost": m_cost,
            "milp_status": m_status,
            "milp_time": m_time,
            "milp_active": m_active,
            "milp_gap": m_gap,
            "greedy_cost": g_cost,
            "greedy_time": g_time,
            "greedy_active": g_active,
            "greedy_gap": g_gap,
            "ga_cost": ga_cost,
            "ga_time": ga_time,
            "ga_active": ga_active,
            "ga_gap": ga_gap
        })

    # Print the summary table in markdown format
    df = pd.DataFrame(results)
    output_csv_path = os.path.join(base_dir, "..", "docs", "large_benchmark_results.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    print("\n" + "=" * 110)
    print(f"{'FINAL LARGE-SCALE BENCHMARK COMPARISON':^110}")
    print("=" * 110)
    print("NOTE: milp_gap is vs. ground_truth (literature optimum). greedy_gap and ga_gap")
    print("are vs. milp_cost (the exact/reference solver in this comparison), NOT vs. ground_truth.")
    print("| Dataset | Ground Truth | MILP Cost | MILP Gap (%) | MILP Status | MILP Time (s) | Greedy Cost | Greedy Gap vs MILP (%) | GA Cost | GA Gap vs MILP (%) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(f"| {r['dataset']} | ${r['ground_truth']:,.2f} | ${r['milp_cost']:,.2f} | {r['milp_gap']:.2f}% | {r['milp_status']} | {r['milp_time']:.2f}s | ${r['greedy_cost']:,.2f} | {r['greedy_gap']:.2f}% | ${r['ga_cost']:,.2f} | {r['ga_gap']:.2f}% |")

if __name__ == "__main__":
    run_benchmarks()
