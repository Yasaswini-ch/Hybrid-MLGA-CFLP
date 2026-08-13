"""
compare_rf_vs_gb.py
=====================
Compares the Hybrid ML-GA under a Random Forest surrogate (raw_runs_<name>.json,
the architecture used for the paper's headline results) against the same
framework under a Gradient Boosting surrogate (gb_runs_<name>.json), with the
Classical GA baseline for reference.

The comparison is controlled: identical cached bootstrap corpora, identical seed
sequence (42-51), identical population/generation budgets. Only the surrogate
architecture differs.

Outputs:
  - rf_vs_gb_comparison.csv       per-instance gaps for all three methods
  - rf_vs_gb_tests.json           per-group means + Wilcoxon RF-vs-GB and GB-vs-Classical
"""
import os
import json
import csv

import numpy as np
from scipy.stats import wilcoxon

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SMALL_MEDIUM = ["cap41","cap42","cap43","cap44","cap51","cap61","cap62","cap63","cap64",
                "cap71","cap72","cap73","cap74","cap81","cap82","cap83","cap84",
                "cap91","cap92","cap93","cap94","cap101","cap102","cap103","cap104"]
LARGE50 = ["cap111","cap112","cap113","cap114","cap121","cap122","cap123","cap124",
           "cap131","cap132","cap133","cap134"]
XLARGE = ["capa4","capb4","capc4"]
ALL_INSTANCES = SMALL_MEDIUM + LARGE50 + XLARGE

GROUPS = {
    "small_medium (16-25 fac)": SMALL_MEDIUM,
    "large50 (50 fac)": LARGE50,
    "xlarge (100 fac)": XLARGE,
    "overall (all 40)": ALL_INSTANCES,
}


def main():
    data = {}
    missing = []
    for name in ALL_INSTANCES:
        rf_p = os.path.join(OUT_DIR, f"raw_runs_{name}.json")
        gb_p = os.path.join(OUT_DIR, f"gb_runs_{name}.json")
        if not (os.path.exists(rf_p) and os.path.exists(gb_p)):
            missing.append(name)
            continue
        with open(rf_p) as f:
            rf = json.load(f)
        with open(gb_p) as f:
            gb = json.load(f)
        data[name] = {
            "optimal": rf["optimal"],
            "classical_gaps": rf["classical_gaps_pct"],
            "rf_gaps": rf["hybrid_gaps_pct"],
            "gb_gaps": gb["gb_gaps_pct"],
            "gb_mape": gb["surrogate_mape_pct"],
        }
    if missing:
        print(f"[WARNING] missing {len(missing)}: {missing}")

    # ---- per-instance CSV ----
    csv_path = os.path.join(OUT_DIR, "rf_vs_gb_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "classical_best_gap", "rf_best_gap", "gb_best_gap",
                    "classical_avg_gap", "rf_avg_gap", "gb_avg_gap", "gb_surrogate_mape"])
        for name in ALL_INSTANCES:
            if name not in data:
                continue
            d = data[name]
            w.writerow([name,
                        min(d["classical_gaps"]), min(d["rf_gaps"]), min(d["gb_gaps"]),
                        np.mean(d["classical_gaps"]), np.mean(d["rf_gaps"]), np.mean(d["gb_gaps"]),
                        d["gb_mape"]])
    print(f"Saved {csv_path}")

    # ---- group stats + tests ----
    results = {}
    for gname, instances in GROUPS.items():
        avail = [n for n in instances if n in data]
        if not avail:
            continue
        cls, rf, gb = [], [], []
        best_cls, best_rf, best_gb = [], [], []
        for n in avail:
            cls.extend(data[n]["classical_gaps"])
            rf.extend(data[n]["rf_gaps"])
            gb.extend(data[n]["gb_gaps"])
            best_cls.append(min(data[n]["classical_gaps"]))
            best_rf.append(min(data[n]["rf_gaps"]))
            best_gb.append(min(data[n]["gb_gaps"]))
        cls, rf, gb = np.array(cls), np.array(rf), np.array(gb)

        def safe_wilcoxon(a, b):
            if np.all(a - b == 0):
                return None, 1.0
            try:
                s, p = wilcoxon(a, b)
                return float(s), float(p)
            except ValueError:
                return None, None

        s_rfgb, p_rfgb = safe_wilcoxon(rf, gb)
        s_gbcl, p_gbcl = safe_wilcoxon(gb, cls)

        results[gname] = {
            "n_instances": len(avail),
            "n_pairs": len(cls),
            "mean_gap_classical": float(cls.mean()),
            "mean_gap_rf": float(rf.mean()),
            "mean_gap_gb": float(gb.mean()),
            "mean_bestrun_gap_classical": float(np.mean(best_cls)),
            "mean_bestrun_gap_rf": float(np.mean(best_rf)),
            "mean_bestrun_gap_gb": float(np.mean(best_gb)),
            "wilcoxon_rf_vs_gb_p": p_rfgb,
            "gb_better_than_rf": bool(gb.mean() < rf.mean()),
            "wilcoxon_gb_vs_classical_p": p_gbcl,
            "gb_better_than_classical": bool(gb.mean() < cls.mean()),
        }

        print(f"\n[{gname}] {len(avail)} instances, {len(cls)} pairs")
        print(f"  mean gap  -- Classical {cls.mean():8.4f}% | RF {rf.mean():8.4f}% | GB {gb.mean():8.4f}%")
        print(f"  best-run  -- Classical {np.mean(best_cls):8.4f}% | RF {np.mean(best_rf):8.4f}% | GB {np.mean(best_gb):8.4f}%")
        print(f"  Wilcoxon RF vs GB: p={p_rfgb}  (GB better: {gb.mean() < rf.mean()})")
        print(f"  Wilcoxon GB vs Classical: p={p_gbcl}  (GB better: {gb.mean() < cls.mean()})")

    out = os.path.join(OUT_DIR, "rf_vs_gb_tests.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
