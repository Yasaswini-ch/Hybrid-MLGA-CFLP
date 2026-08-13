"""
run_significance_tests.py
===========================
Aggregates raw_runs_<name>.json files (produced by collect_raw_runs.py, one
per instance, each holding 10 matched-seed raw costs for both the classical
GA and the Hybrid ML-GA) into:
  - raw_runs_all.csv           (long format: instance, solver, run, seed, cost, gap_pct)
  - significance_test_results.json  (Wilcoxon signed-rank test per instance-size
    group and overall, matching the same 10 seeds pairwise between solvers)
"""
import os
import json
import csv

import numpy as np
from scipy.stats import wilcoxon

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SMALL_MEDIUM = [
    "cap41", "cap42", "cap43", "cap44", "cap51",
    "cap61", "cap62", "cap63", "cap64",
    "cap71", "cap72", "cap73", "cap74",
    "cap81", "cap82", "cap83", "cap84",
    "cap91", "cap92", "cap93", "cap94",
    "cap101", "cap102", "cap103", "cap104",
]
LARGE50 = ["cap111", "cap112", "cap113", "cap114",
           "cap121", "cap122", "cap123", "cap124",
           "cap131", "cap132", "cap133", "cap134"]
XLARGE = ["capa4", "capb4", "capc4"]
ALL_INSTANCES = SMALL_MEDIUM + LARGE50 + XLARGE

GROUPS = {
    "small_medium (16-25 facilities)": SMALL_MEDIUM,
    "large50 (50 facilities)": LARGE50,
    "xlarge (100 facilities)": XLARGE,
    "overall (all 40)": ALL_INSTANCES,
}


def load_all():
    data = {}
    missing = []
    for name in ALL_INSTANCES:
        path = os.path.join(OUT_DIR, f"raw_runs_{name}.json")
        if not os.path.exists(path):
            missing.append(name)
            continue
        with open(path) as f:
            data[name] = json.load(f)
    return data, missing


def main():
    data, missing = load_all()
    if missing:
        print(f"[WARNING] Missing raw_runs files for {len(missing)} instances: {missing}")
        print("Proceeding with the instances that ARE available.")

    # ---- long-format CSV ----
    csv_path = os.path.join(OUT_DIR, "raw_runs_all.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance", "solver", "run", "seed", "cost", "gap_pct"])
        for name, d in data.items():
            for run in range(len(d["classical_costs"])):
                writer.writerow([name, "classical", run, 42 + run, d["classical_costs"][run], d["classical_gaps_pct"][run]])
            for run in range(len(d["hybrid_costs"])):
                writer.writerow([name, "hybrid", run, 42 + run, d["hybrid_costs"][run], d["hybrid_gaps_pct"][run]])
    print(f"Saved {csv_path}")

    # ---- Wilcoxon signed-rank test per group (paired by instance+seed) ----
    test_results = {}
    for group_name, instances in GROUPS.items():
        available = [n for n in instances if n in data]
        if not available:
            continue
        classical_gaps = []
        hybrid_gaps = []
        for name in available:
            classical_gaps.extend(data[name]["classical_gaps_pct"])
            hybrid_gaps.extend(data[name]["hybrid_gaps_pct"])

        classical_gaps = np.array(classical_gaps)
        hybrid_gaps = np.array(hybrid_gaps)
        n_pairs = len(classical_gaps)

        diffs = hybrid_gaps - classical_gaps
        if np.all(diffs == 0):
            stat, p = float("nan"), 1.0
        else:
            try:
                stat, p = wilcoxon(hybrid_gaps, classical_gaps)
            except ValueError as e:
                stat, p = float("nan"), float("nan")
                print(f"[{group_name}] Wilcoxon failed: {e}")

        mean_classical = float(np.mean(classical_gaps))
        mean_hybrid = float(np.mean(hybrid_gaps))
        better = "classical" if mean_classical < mean_hybrid else ("hybrid" if mean_hybrid < mean_classical else "tie")

        test_results[group_name] = {
            "n_instances": len(available),
            "n_pairs": n_pairs,
            "mean_classical_gap_pct": mean_classical,
            "mean_hybrid_gap_pct": mean_hybrid,
            "better_method": better,
            "wilcoxon_statistic": float(stat) if stat == stat else None,
            "p_value": float(p) if p == p else None,
            "significant_at_0.05": bool(p < 0.05) if p == p else None,
        }

        print(f"\n[{group_name}] n_instances={len(available)} n_pairs={n_pairs}")
        print(f"  Mean gap -- Classical: {mean_classical:.4f}% | Hybrid: {mean_hybrid:.4f}%")
        print(f"  Wilcoxon statistic={stat}, p-value={p}")
        print(f"  Significant at alpha=0.05: {p < 0.05 if p == p else 'N/A'}")
        print(f"  Better method (lower mean gap): {better}")

    out_path = os.path.join(OUT_DIR, "significance_test_results.json")
    with open(out_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
