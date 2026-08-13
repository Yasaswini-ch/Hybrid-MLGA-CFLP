"""
generate_all_convergence.py
============================
Runs ONE classical-GA solve and ONE bootstrap+train+confidence_aware-solve
of the Hybrid ML-GA per OR-Library instance (all 40), using the exact same
pop/gen/mutation budgets as benchmark_statistical.py / benchmark_hybrid_ga.py,
and records the per-generation best-cost convergence history for each.

Produces:
  - all_convergence_data.csv       (long format: instance, solver, generation, best_cost)
  - all_instances_classical_convergence_grid.png
  - all_instances_hybrid_convergence_grid.png

This is a single representative run per instance per solver (not the full
30-run / 10-run statistical benchmark) -- it exists purely to visualise
convergence shape across all 40 instances for the paper, using the project's
own recorded per-instance seed-0 result as ground truth for "did it actually
converge to something sane on this seed", not as a new statistical claim.
"""
import os
import sys
import time
import random
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_SRC_DIR = os.path.normpath(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from parser import CFLPDataset
from ga_solver import CFLPGASolver
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
from dataset_generator import CFLPDatasetGenerator
from training_pipeline import SurrogateTrainingPipeline

_ROOT_DIR = os.path.normpath(os.path.join(_SRC_DIR, ".."))
RAW_DIR = os.path.join(_ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(_ROOT_DIR, "data", "processed")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_SEED = 42

SMALL_POP, SMALL_GEN, SMALL_MUT = 120, 100, 0.3
LARGE_POP, LARGE_GEN, LARGE_MUT = 40, 60, 0.2

HYBRID_SMALL_SOLVE_POP, HYBRID_SMALL_SOLVE_GEN = 120, 100
HYBRID_LARGE_SOLVE_POP, HYBRID_LARGE_SOLVE_GEN = 100, 100
BOOTSTRAP_POP_SMALL, BOOTSTRAP_GEN_SMALL = 30, 15
BOOTSTRAP_POP_LARGE, BOOTSTRAP_GEN_LARGE = 60, 40
WARMUP_FRACTION = 0.15

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
    "capa4", "capb4", "capc4",
]

LARGE100 = {"capa4", "capb4", "capc4"}

results = {}
progress_path = os.path.join(OUT_DIR, "all_convergence_progress.json")


def save_progress():
    with open(progress_path, "w") as f:
        json.dump(results, f)


def run_classical(name, dataset):
    is_large = name in LARGE100
    pop = LARGE_POP if is_large else SMALL_POP
    gen = LARGE_GEN if is_large else SMALL_GEN
    mut = LARGE_MUT if is_large else SMALL_MUT

    solver = CFLPGASolver(dataset)
    solver.clear_cache()
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    t0 = time.time()
    best_cost, best_y, history = solver.solve(pop_size=pop, n_gen=gen, cx_pb=0.8, mut_pb=mut)
    elapsed = time.time() - t0
    return best_cost, history.get("min_cost", []), elapsed


def run_hybrid(name, dataset):
    is_large = name in LARGE100
    solve_pop = HYBRID_LARGE_SOLVE_POP if is_large else HYBRID_SMALL_SOLVE_POP
    solve_gen = HYBRID_LARGE_SOLVE_GEN if is_large else HYBRID_SMALL_SOLVE_GEN
    boot_pop = BOOTSTRAP_POP_LARGE if is_large else BOOTSTRAP_POP_SMALL
    boot_gen = BOOTSTRAP_GEN_LARGE if is_large else BOOTSTRAP_GEN_SMALL

    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    t0 = time.time()
    bootstrap_ga = HybridMLGASolver(
        dataset=dataset, surrogate=None,
        pop_size=boot_pop, n_generations=boot_gen,
        random_seed=BASE_SEED,
    )
    boot_result = bootstrap_ga.solve()
    bootstrap_time = time.time() - t0

    X, y = extract_training_data_from_ga(boot_result, dataset=dataset)
    corpus_path = os.path.join(PROCESSED_DIR, f"convplot_corpus_{name}.npz")
    CFLPDatasetGenerator(dataset).save(X, y, corpus_path)

    t1 = time.time()
    pipeline = SurrogateTrainingPipeline(dataset=dataset, corpus_path=corpus_path, model_save_dir=PROCESSED_DIR)
    train_results = pipeline.run(model_types=("random_forest",))
    train_time = time.time() - t1
    surrogate = train_results["random_forest"]["surrogate"]

    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    t2 = time.time()
    solver = HybridMLGASolver(
        dataset=dataset, surrogate=surrogate,
        pop_size=solve_pop, n_generations=solve_gen,
        mode="confidence_aware", warmup_fraction=WARMUP_FRACTION,
        random_seed=BASE_SEED,
    )
    result = solver.solve()
    solve_time = time.time() - t2

    hist = result.get("history", {}).get("min_cost", [])
    return result["best_cost"], hist, bootstrap_time, train_time, solve_time


def main():
    t_start = time.time()
    instances_to_run = sys.argv[1:] if len(sys.argv) > 1 else INSTANCES
    for idx, name in enumerate(instances_to_run):
        print(f"\n[{idx+1}/{len(INSTANCES)}] === {name} ===", flush=True)
        file_path = os.path.join(RAW_DIR, f"{name}.txt")
        dataset = CFLPDataset(file_path)

        print(f"  running classical GA...", flush=True)
        c_cost, c_hist, c_time = run_classical(name, dataset)
        print(f"  classical: cost={c_cost:,.2f} gens={len(c_hist)} time={c_time:.1f}s", flush=True)

        print(f"  running hybrid ML-GA (bootstrap+train+solve)...", flush=True)
        h_cost, h_hist, boot_t, train_t, solve_t = run_hybrid(name, dataset)
        print(f"  hybrid: cost={h_cost:,.2f} gens={len(h_hist)} "
              f"boot={boot_t:.1f}s train={train_t:.1f}s solve={solve_t:.1f}s", flush=True)

        results[name] = {
            "classical_cost": c_cost,
            "classical_history": c_hist,
            "classical_time": c_time,
            "hybrid_cost": h_cost,
            "hybrid_history": h_hist,
            "hybrid_bootstrap_time": boot_t,
            "hybrid_train_time": train_t,
            "hybrid_solve_time": solve_t,
        }
        save_progress()

    total = time.time() - t_start
    print(f"\nALL DONE in {total/60.0:.1f} minutes", flush=True)

    done_instances = [n for n in INSTANCES if n in results]

    # ---- write long-format CSV ----
    import csv
    csv_path = os.path.join(OUT_DIR, "all_convergence_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance", "solver", "generation", "raw_gen_min_cost", "running_best_cost"])
        for name in done_instances:
            r = results[name]
            c_running = np.minimum.accumulate(r["classical_history"]) if r["classical_history"] else []
            h_running = np.minimum.accumulate(r["hybrid_history"]) if r["hybrid_history"] else []
            for g, (c, cb) in enumerate(zip(r["classical_history"], c_running)):
                writer.writerow([name, "classical", g, c, cb])
            for g, (c, cb) in enumerate(zip(r["hybrid_history"], h_running)):
                writer.writerow([name, "hybrid", g, c, cb])
    print(f"Saved {csv_path}", flush=True)

    # ---- grid figures (small multiples, normalized to each instance's own final cost) ----
    def make_grid(solver_key, hist_key, title, out_name):
        fig, axes = plt.subplots(5, 8, figsize=(24, 14))
        axes = axes.flatten()
        for i, name in enumerate(done_instances):
            ax = axes[i]
            hist = results[name][hist_key]
            if len(hist) == 0:
                ax.set_visible(False)
                continue
            # history["min_cost"] is the raw per-generation population minimum
            # (can be non-monotonic under the surrogate, e.g. confidence_aware
            # mode only re-verifies chromosomes that *beat* the incumbent, so a
            # later generation's raw min can be a surrogate-noisy value that's
            # worse than an earlier verified best). Convert to running best-so-far
            # (cumulative min) to match the "best cost so far" semantics of the
            # existing single-instance convergence plots elsewhere in this paper.
            running_best = np.minimum.accumulate(hist)
            final = running_best[-1]
            ratio = [h / final if final != 0 else 1.0 for h in running_best]
            ax.plot(ratio, color="#2b8cbe" if solver_key == "classical" else "#e6550d", linewidth=1.2)
            ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_title(name, fontsize=8)
            ax.tick_params(labelsize=6)
        for j in range(len(done_instances), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(title, fontsize=14)
        fig.text(0.5, 0.005, "Generation (x-axis) vs. Best Cost / Final Best Cost (y-axis)", ha="center", fontsize=9)
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        out_path = os.path.join(OUT_DIR, out_name)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}", flush=True)

    make_grid("classical", "classical_history",
               "Classical GA Convergence, Normalized to Final Best Cost (All 40 OR-Library Instances)",
               "all_instances_classical_convergence_grid.png")
    make_grid("hybrid", "hybrid_history",
               "Hybrid ML-GA Convergence, Normalized to Final Best Cost (All 40 OR-Library Instances)",
               "all_instances_hybrid_convergence_grid.png")


if __name__ == "__main__":
    main()
