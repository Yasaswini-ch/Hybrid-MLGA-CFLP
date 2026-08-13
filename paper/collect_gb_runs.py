"""
collect_gb_runs.py
===================
Re-runs the Hybrid ML-GA evaluation phase using a GRADIENT BOOSTING surrogate
instead of the Random Forest used for the paper's headline results.

Deliberately controlled comparison: for each instance this reuses the SAME
cached bootstrap corpus (data/processed/convplot_corpus_<name>.npz) and the
SAME seed sequence (42-51) as collect_raw_runs.py used for the Random Forest
numbers in raw_runs_<name>.json. The only thing that differs is the surrogate
architecture, so any difference in outcome is attributable to that choice
alone rather than to different training data or different random seeds.

Writes gb_runs_<name>.json per instance. Safe to run several copies in
parallel over disjoint instance lists.
"""
import os
import sys
import json
import random
import time

import numpy as np

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from parser import CFLPDataset
from hybrid_ga import HybridMLGASolver
from training_pipeline import SurrogateTrainingPipeline

_ROOT_DIR = os.path.normpath(os.path.join(_SRC_DIR, ".."))
RAW_DIR = os.path.join(_ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(_ROOT_DIR, "data", "processed")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Per-process model dir so parallel workers cannot overwrite each other's
# surrogate_gradient_boosting.pkl (SurrogateTrainingPipeline saves to a fixed
# filename inside model_save_dir).
GB_MODEL_DIR = os.path.join(PROCESSED_DIR, "gb_experiment", f"pid{os.getpid()}")
os.makedirs(GB_MODEL_DIR, exist_ok=True)

BASE_SEED = 42
N_RUNS = 10
MODEL_TYPE = "gradient_boosting"

HYBRID_SMALL_POP, HYBRID_SMALL_GEN = 120, 100
HYBRID_LARGE_POP, HYBRID_LARGE_GEN = 100, 100
WARMUP_FRACTION = 0.15

LARGE100 = {"capa4", "capb4", "capc4"}

CFLP_OPTIMAL = {
    "cap41": 1040444.375, "cap42": 1098000.450, "cap43": 1153000.450, "cap44": 1235500.450,
    "cap51": 1025208.225,
    "cap61": 932615.750, "cap62": 977799.400, "cap63": 1014062.050, "cap64": 1045650.250,
    "cap71": 932615.750, "cap72": 977799.400, "cap73": 1010641.450, "cap74": 1034976.975,
    "cap81": 838499.288, "cap82": 910889.563, "cap83": 975889.563, "cap84": 1069369.525,
    "cap91": 796648.438, "cap92": 855733.500, "cap93": 896617.538, "cap94": 946051.325,
    "cap101": 796648.437, "cap102": 854704.200, "cap103": 893782.112, "cap104": 928941.750,
    "cap111": 826124.713, "cap112": 901377.213, "cap113": 970567.750, "cap114": 1063356.488,
    "cap121": 793439.563, "cap122": 852524.625, "cap123": 895302.325, "cap124": 946051.325,
    "cap131": 793439.562, "cap132": 851495.325, "cap133": 893076.712, "cap134": 928941.750,
    "capa4": 17160612.23, "capb4": 13083203.74, "capc4": 11505861.86,
}


def main():
    instances = sys.argv[1:]
    if not instances:
        print("Usage: python collect_gb_runs.py <instance1> [instance2] ...")
        sys.exit(1)

    for name in instances:
        out_path = os.path.join(OUT_DIR, f"gb_runs_{name}.json")
        if os.path.exists(out_path):
            print(f"[SKIP] {name}: already done", flush=True)
            continue

        print(f"\n=== {name} (gradient_boosting) ===", flush=True)
        dataset = CFLPDataset(os.path.join(RAW_DIR, f"{name}.txt"))
        optimal = CFLP_OPTIMAL[name]
        is_large = name in LARGE100
        pop = HYBRID_LARGE_POP if is_large else HYBRID_SMALL_POP
        gen = HYBRID_LARGE_GEN if is_large else HYBRID_SMALL_GEN

        corpus_path = os.path.join(PROCESSED_DIR, f"convplot_corpus_{name}.npz")
        t_train = time.time()
        pipeline = SurrogateTrainingPipeline(
            dataset=dataset, corpus_path=corpus_path, model_save_dir=GB_MODEL_DIR
        )
        train_results = pipeline.run(model_types=(MODEL_TYPE,))
        train_time = time.time() - t_train
        surrogate = train_results[MODEL_TYPE]["surrogate"]
        metrics = train_results[MODEL_TYPE]["metrics"]

        costs = []
        times = []
        for run in range(N_RUNS):
            seed = BASE_SEED + run
            random.seed(seed)
            np.random.seed(seed)
            t0 = time.time()
            solver = HybridMLGASolver(
                dataset=dataset, surrogate=surrogate,
                pop_size=pop, n_generations=gen,
                mode="confidence_aware", warmup_fraction=WARMUP_FRACTION,
                random_seed=seed,
            )
            result = solver.solve()
            elapsed = time.time() - t0
            costs.append(result["best_cost"])
            times.append(elapsed)
            print(f"  [gb run {run+1}/{N_RUNS}] seed={seed} cost=${result['best_cost']:,.2f} ({elapsed:.1f}s)", flush=True)

        best = float(np.min(costs))
        avg = float(np.mean(costs))
        worst = float(np.max(costs))

        payload = {
            "instance": name,
            "optimal": optimal,
            "model_type": MODEL_TYPE,
            "gb_costs": costs,
            "gb_gaps_pct": [(c - optimal) / optimal * 100.0 for c in costs],
            "gb_best": best,
            "gb_avg": avg,
            "gb_worst": worst,
            "gb_best_gap_pct": (best - optimal) / optimal * 100.0,
            "gb_avg_gap_pct": (avg - optimal) / optimal * 100.0,
            "surrogate_r2": metrics["r2"],
            "surrogate_mape_pct": metrics["mape_pct"],
            "train_time_sec": train_time,
            "avg_solve_time_sec": float(np.mean(times)),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  saved {out_path} | best gap {payload['gb_best_gap_pct']:.4f}%", flush=True)


if __name__ == "__main__":
    main()
