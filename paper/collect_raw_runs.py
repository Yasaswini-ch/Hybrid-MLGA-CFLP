"""
collect_raw_runs.py
=====================
Worker script: given a list of instance names on the command line, for each one:
  1. Run the classical GA for 10 matched seeds (42-51), same pop/gen/mut budget
     as benchmark_statistical.py's group config, capturing raw per-run costs.
  2. Retrain a Random Forest surrogate from the CACHED bootstrap corpus
     (data/processed/convplot_corpus_<name>.npz, produced earlier by
     generate_all_convergence.py -- no re-bootstrapping needed), then run the
     Hybrid ML-GA confidence_aware solver for the SAME 10 seeds, same pop/gen
     budget as benchmark_hybrid_ga.py's group config, capturing raw per-run costs.
  3. Write results to raw_runs_<name>.json.

Designed to be launched multiple times in parallel (one process per chunk of
instances) so that all 40 instances' worth of matched-run data can be
collected in parallel across CPU cores instead of one at a time.
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
from ga_solver import CFLPGASolver
from hybrid_ga import HybridMLGASolver
from training_pipeline import SurrogateTrainingPipeline

_ROOT_DIR = os.path.normpath(os.path.join(_SRC_DIR, ".."))
RAW_DIR = os.path.join(_ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(_ROOT_DIR, "data", "processed")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_SEED = 42
N_RUNS = 10

SMALL_POP, SMALL_GEN, SMALL_MUT = 120, 100, 0.3
LARGE_POP, LARGE_GEN, LARGE_MUT = 40, 60, 0.2

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


def collect_classical(name, dataset):
    is_large = name in LARGE100
    pop = LARGE_POP if is_large else SMALL_POP
    gen = LARGE_GEN if is_large else SMALL_GEN
    mut = LARGE_MUT if is_large else SMALL_MUT

    solver = CFLPGASolver(dataset)
    costs = []
    for run in range(N_RUNS):
        solver.clear_cache()
        seed = BASE_SEED + run
        random.seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        best_cost, _, _ = solver.solve(pop_size=pop, n_gen=gen, cx_pb=0.8, mut_pb=mut)
        costs.append(best_cost)
        print(f"    [classical run {run+1}/{N_RUNS}] seed={seed} cost=${best_cost:,.2f} ({time.time()-t0:.1f}s)", flush=True)
    return costs


def collect_hybrid(name, dataset):
    is_large = name in LARGE100
    pop = HYBRID_LARGE_POP if is_large else HYBRID_SMALL_POP
    gen = HYBRID_LARGE_GEN if is_large else HYBRID_SMALL_GEN

    corpus_path = os.path.join(PROCESSED_DIR, f"convplot_corpus_{name}.npz")
    pipeline = SurrogateTrainingPipeline(dataset=dataset, corpus_path=corpus_path, model_save_dir=PROCESSED_DIR)
    train_results = pipeline.run(model_types=("random_forest",))
    surrogate = train_results["random_forest"]["surrogate"]

    costs = []
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
        costs.append(result["best_cost"])
        print(f"    [hybrid run {run+1}/{N_RUNS}] seed={seed} cost=${result['best_cost']:,.2f} ({time.time()-t0:.1f}s)", flush=True)
    return costs


def main():
    instances = sys.argv[1:]
    if not instances:
        print("Usage: python collect_raw_runs.py <instance1> [instance2] ...")
        sys.exit(1)

    for name in instances:
        print(f"\n=== {name} ===", flush=True)
        out_path = os.path.join(OUT_DIR, f"raw_runs_{name}.json")
        if os.path.exists(out_path):
            print(f"  [SKIP] {out_path} already exists", flush=True)
            continue

        dataset = CFLPDataset(os.path.join(RAW_DIR, f"{name}.txt"))
        optimal = CFLP_OPTIMAL[name]

        print(f"  collecting classical GA (10 matched runs)...", flush=True)
        classical_costs = collect_classical(name, dataset)

        print(f"  collecting hybrid ML-GA (10 matched runs)...", flush=True)
        hybrid_costs = collect_hybrid(name, dataset)

        result = {
            "instance": name,
            "optimal": optimal,
            "classical_costs": classical_costs,
            "hybrid_costs": hybrid_costs,
            "classical_gaps_pct": [(c - optimal) / optimal * 100.0 for c in classical_costs],
            "hybrid_gaps_pct": [(c - optimal) / optimal * 100.0 for c in hybrid_costs],
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
