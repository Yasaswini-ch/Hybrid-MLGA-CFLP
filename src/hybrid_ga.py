"""
hybrid_ga.py
============
Hybrid Machine Learning + Genetic Algorithm Solver for the CFLP.

Architecture:
  Extends the Phase 3 ModularCFLPGASolver by replacing the expensive SciPy LP
  sub-problem fitness evaluation with a pre-trained ML surrogate model that
  predicts transportation costs in microseconds.

Two Evaluation Modes:
  1. 'pure_surrogate'   — All fitness evaluations use the ML model (zero LP calls).
                          Maximum speed. Suitable for high-quality surrogates (MAPE < 1%).

  2. 'confidence_aware' — ML surrogate predicts a cost for every individual; the
                          exact LP solver is only invoked to verify that prediction
                          when it suggests a NEW BEST solution, i.e.
                          predicted_cost < best_cost_so_far (see _evaluate_individual
                          and _evaluate_population_batch below). Otherwise the
                          prediction is trusted directly. Best balance of speed and
                          accuracy: most of the population is evaluated in
                          microseconds, while every genuinely promising candidate is
                          still verified exactly before being trusted.

Warmup Period:
  The first `warmup_fraction` of total generations always use exact LP evaluations,
  regardless of mode. This builds a verified elite chromosome set and gives the
  surrogate a chance to be applied on well-explored regions only.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deap import base, creator, tools

from parser import CFLPDataset
from baseline import MILPSolver, GreedySolver
from genetic_algorithm import ModularCFLPGASolver
from surrogate_model import CFLPSurrogateModel
from feature_engineering import CFLPFeatureEngineer
from fitness import CFLPFitnessEvaluator


class HybridMLGASolver:
    """
    Hybrid ML + GA solver for CFLP. Uses a pre-trained surrogate model to
    accelerate fitness evaluations within a Genetic Algorithm loop.
    """

    def __init__(self,
                 dataset: CFLPDataset,
                 surrogate: Optional[CFLPSurrogateModel] = None,
                 pop_size: int = 50,
                 n_generations: int = 100,
                 cx_pb: float = 0.80,
                 mut_pb: float = 0.20,
                 heuristic_ratio: float = 0.50,
                 elite_count: int = 1,
                 mode: str = "pure_surrogate",
                 uncertainty_threshold_pct: float = 5.0,
                 elite_fraction: float = 0.10,
                 warmup_fraction: float = 0.20,
                 random_seed: int = 42):
        """
        Args:
            dataset: Parsed CFLP instance.
            surrogate: A fitted CFLPSurrogateModel instance, or None to run in
                       bootstrap mode (no surrogate available yet). In bootstrap
                       mode every individual is evaluated with the exact LP solver
                       and logged to exact_evaluations_log, regardless of `mode`
                       or `warmup_fraction` — this is how the initial training
                       corpus is produced before any surrogate exists.
            pop_size: Population size.
            n_generations: Total number of generations.
            cx_pb: Crossover probability.
            mut_pb: Mutation probability.
            heuristic_ratio: Fraction of population to initialize with heuristic seeding.
            elite_count: Number of elite individuals to carry over.
            mode: 'pure_surrogate' or 'confidence_aware'. Ignored when surrogate is None.
            uncertainty_threshold_pct: Accepted for backward compatibility with existing
                                       callers (e.g. active_learning.py) but currently UNUSED --
                                       the confidence_aware decision rule is purely
                                       predicted_cost < best_cost_so_far (see the module
                                       docstring above), not an uncertainty threshold.
            elite_fraction: Accepted for backward compatibility but currently UNUSED,
                            same reason as uncertainty_threshold_pct above.
            warmup_fraction: Fraction of generations that always use exact LP (warmup period).
                             Ignored when surrogate is None (every generation is exact).
            random_seed: Reproducibility seed.
        """
        if mode not in ("pure_surrogate", "confidence_aware"):
            raise ValueError(f"Unknown mode '{mode}'. Choose 'pure_surrogate' or 'confidence_aware'.")

        self.dataset = dataset
        self.surrogate = surrogate
        self.bootstrap_mode = surrogate is None
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cx_pb = cx_pb
        self.mut_pb = mut_pb
        self.heuristic_ratio = heuristic_ratio
        self.elite_count = elite_count
        self.mode = mode
        self.uncertainty_threshold_pct = uncertainty_threshold_pct
        self.elite_fraction = elite_fraction
        self.warmup_fraction = warmup_fraction
        self.random_seed = random_seed

        self.m = dataset.num_facilities
        self.n_customers = dataset.num_customers

        # Feature engineer and exact LP evaluator (for fallback)
        self.feature_engineer = CFLPFeatureEngineer(dataset, mode="full")
        self.exact_evaluator = CFLPFitnessEvaluator(dataset)

        # Tracking statistics
        self.total_surrogate_evals = 0
        self.total_exact_evals = 0
        self.exact_evaluations_log = []  # List of (chromosome_list, exact_cost) for active learning

        # Incumbent best cost, updated during solve(). Read by _evaluate_individual()
        # and _evaluate_population_batch() to decide whether a predicted cost
        # indicates potential to outperform the current best solution.
        self.best_overall_cost = float("inf")

        np.random.seed(random_seed)
        self._setup_deap()

    def _setup_deap(self):
        """Registers DEAP types and operators — compatible with Phase 3 pattern."""
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("bit", np.random.randint, 0, 2)
        self.toolbox.register("individual",
                              tools.initRepeat,
                              creator.Individual,
                              self.toolbox.bit, n=self.m)
        self.toolbox.register("population",
                              tools.initRepeat, list,
                              self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / self.m)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    # ------------------------------------------------------------------
    # CORE: Hybrid Fitness Evaluation
    # ------------------------------------------------------------------
    def _evaluate_individual(self,
                              individual: list,
                              generation: int,
                              predicted_cost: Optional[float] = None) -> float:
        """
        Evaluates a single individual's fitness using either the surrogate or exact LP.

        Decision logic:
          0. Bootstrap mode (no surrogate loaded) → always exact LP
          1. Warmup period (first warmup_fraction generations) → always exact LP
          2. pure_surrogate mode → always surrogate, never exact
          3. confidence_aware mode → predict cost first; only compute exact LP if
             predicted_cost < self.best_overall_cost (the surrogate indicates this
             candidate has the potential to outperform the current best solution).
             Otherwise the surrogate's prediction is trusted directly.

        Returns:
            float: Total objective cost (fixed + transport).
        """
        warmup_gens = int(self.n_generations * self.warmup_fraction)

        if self.bootstrap_mode or generation < warmup_gens:
            cost = self.exact_evaluator.evaluate(individual)[0]  # evaluate() returns (cost,) tuple
            self.total_exact_evals += 1
            self.exact_evaluations_log.append((list(individual), cost))
            return cost

        if self.mode == "pure_surrogate":
            y = np.array(individual, dtype=np.float64).reshape(1, -1)
            X_feat = self.feature_engineer.transform(y)
            predicted = float(self.surrogate.predict(X_feat)[0])
            self.total_surrogate_evals += 1
            return predicted

        # mode == "confidence_aware": predicted-cost-vs-current-best decision
        if predicted_cost is None:
            y = np.array(individual, dtype=np.float64).reshape(1, -1)
            X_feat = self.feature_engineer.transform(y)
            predicted_cost = float(self.surrogate.predict(X_feat)[0])

        if predicted_cost < self.best_overall_cost:
            # Predicted cost indicates potential to beat the current best -> verify exactly
            cost = self.exact_evaluator.evaluate(individual)[0]
            self.total_exact_evals += 1
            self.exact_evaluations_log.append((list(individual), cost))
            return cost
        else:
            # Predicted cost does not threaten the current best -> trust the prediction
            self.total_surrogate_evals += 1
            return predicted_cost

    def _evaluate_population_batch(self,
                                   population: list,
                                   generation: int) -> List[float]:
        """
        Batch-evaluates the entire population, using surrogate predictions for
        most individuals and exact LP for those requiring verification.

        For confidence_aware mode, predicts the cost of every individual in one
        batch call, then only computes exact LP for individuals whose predicted
        cost is below self.best_overall_cost — i.e. the predicted cost indicates
        potential to outperform the current best solution.

        Returns:
            List[float]: Fitness costs for all individuals in population order.
        """
        warmup_gens = int(self.n_generations * self.warmup_fraction)

        # Bootstrap mode (no surrogate) or warmup period: exact LP for every individual
        if self.bootstrap_mode or generation < warmup_gens:
            costs = []
            for ind in population:
                costs.append(self._evaluate_individual(ind, generation))
            return costs

        # Batch surrogate prediction for all individuals
        Y = np.array([list(ind) for ind in population], dtype=np.float64)
        X_feat = self.feature_engineer.transform(Y)

        y_pred = self.surrogate.predict(X_feat)

        costs = []
        for k, ind in enumerate(population):
            if self.mode == "pure_surrogate":
                cost = float(y_pred[k])
                self.total_surrogate_evals += 1
            else:
                # confidence_aware: predicted cost vs. current best decides exact verification
                predicted = float(y_pred[k])

                if predicted < self.best_overall_cost:
                    # Predicted cost indicates potential to beat current best -> verify exactly
                    cost = self.exact_evaluator.evaluate(list(ind))[0]  # evaluate() returns (cost,) tuple
                    self.total_exact_evals += 1
                    self.exact_evaluations_log.append((list(ind), cost))
                else:
                    # Predicted cost does not threaten current best -> trust the prediction
                    cost = predicted
                    self.total_surrogate_evals += 1

            costs.append(cost)

        return costs

    # ------------------------------------------------------------------
    # MAIN: Evolutionary Loop
    # ------------------------------------------------------------------
    def solve(self) -> Dict[str, Any]:
        """
        Runs the Hybrid ML-GA evolutionary loop.

        Returns:
            Dict with keys: best_cost, best_individual, history, elapsed_time,
                            surrogate_eval_count, exact_eval_count
        """
        t0 = time.time()

        # Initialize population (with heuristic seeding)
        pop = self.toolbox.population(n=self.pop_size)

        history = {"min_cost": [], "avg_cost": [], "gen": []}

        if self.bootstrap_mode:
            print(f"[HybridMLGA] BOOTSTRAP MODE (no surrogate) | Pop={self.pop_size} | "
                  f"Gens={self.n_generations} | All generations use exact LP")
        else:
            print(f"[HybridMLGA] Mode={self.mode} | Pop={self.pop_size} | "
                  f"Gens={self.n_generations} | Warmup={int(self.n_generations*self.warmup_fraction)} gens")

        # Reset incumbent tracking in case solve() is called more than once on this instance
        self.best_overall_cost = float("inf")
        best_overall_ind = None

        for gen in range(self.n_generations):
            # Batch evaluate population — reads self.best_overall_cost as the
            # incumbent from all prior generations to decide predicted-cost
            # vs. current-best exact-verification triggers (confidence_aware mode)
            costs = self._evaluate_population_batch(pop, gen)
            for ind, cost in zip(pop, costs):
                ind.fitness.values = (cost,)

            # Track best (updates self.best_overall_cost for the NEXT generation's decisions)
            gen_min_cost = min(costs)
            gen_avg_cost = float(np.mean(costs))
            if gen_min_cost < self.best_overall_cost:
                self.best_overall_cost = gen_min_cost
                best_idx = costs.index(gen_min_cost)
                best_overall_ind = list(pop[best_idx])

            history["min_cost"].append(gen_min_cost)
            history["avg_cost"].append(gen_avg_cost)
            history["gen"].append(gen)

            if gen % 10 == 0 or gen == self.n_generations - 1:
                warmup_tag = " [WARMUP]" if gen < int(self.n_generations * self.warmup_fraction) else ""
                print(f"  Gen {gen:3d}{warmup_tag}: best=${gen_min_cost:,.2f} | "
                      f"avg=${gen_avg_cost:,.2f} | "
                      f"exact={self.total_exact_evals} surr={self.total_surrogate_evals}")

            # Evolve
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.cx_pb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < self.mut_pb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Elitism: preserve best individual
            if best_overall_ind is not None:
                elite_ind = creator.Individual(best_overall_ind)
                elite_ind.fitness.values = (self.best_overall_cost,)
                offspring[0] = elite_ind

            pop[:] = offspring

        elapsed = time.time() - t0

        # Final verification with exact LP
        print(f"\n[HybridMLGA] Verifying best chromosome with exact LP...")
        exact_final_cost = self.exact_evaluator.evaluate(best_overall_ind)[0]  # returns (cost,)
        surrogate_error_pct = abs(self.best_overall_cost - exact_final_cost) / exact_final_cost * 100
        if self.bootstrap_mode:
            print(f"  Best cost (exact, bootstrap mode): ${exact_final_cost:,.2f}")
        else:
            print(f"  Surrogate best:  ${self.best_overall_cost:,.2f}")
            print(f"  Exact LP verify: ${exact_final_cost:,.2f}")
            print(f"  Surrogate error on best: {surrogate_error_pct:.4f}%")

        return {
            "best_cost": exact_final_cost,
            "best_surrogate_cost": self.best_overall_cost,
            "best_individual": best_overall_ind,
            "surrogate_error_pct": surrogate_error_pct,
            "history": history,
            "elapsed_time": elapsed,
            "surrogate_eval_count": self.total_surrogate_evals,
            "exact_eval_count": self.total_exact_evals,
            "exact_evaluations_log": self.exact_evaluations_log,
            "bootstrap_mode": self.bootstrap_mode,
        }


def run_comparison_experiment(dataset_path: str,
                               model_dir: str,
                               output_dir: str) -> None:
    """
    Runs the full 3-tier comparative experiment:
      Tier 1: Greedy Heuristic Baseline
      Tier 2: Classical GA (Phase 3 modular)
      Tier 3a: Hybrid ML-GA — Pure Surrogate (XGBoost)
      Tier 3b: Hybrid ML-GA — Confidence-Aware (Random Forest)

    Saves a convergence + runtime comparison plot to output_dir.
    """
    print("\n" + "=" * 70)
    print("  CFLP THREE-TIER SOLVER COMPARISON EXPERIMENT")
    print("=" * 70)

    dataset = CFLPDataset(dataset_path)

    results = {}

    # --- Tier 1: Greedy Heuristic ---
    print("\n[Tier 1] Greedy Heuristic Baseline...")
    t0 = time.time()
    greedy = GreedySolver(dataset)
    greedy_cost, _, _ = greedy.solve()
    greedy_time = time.time() - t0
    results["greedy"] = {"cost": greedy_cost, "time": greedy_time}
    print(f"  Greedy cost: ${greedy_cost:,.2f} | Time: {greedy_time:.3f}s")

    # --- Tier 2: Classical GA ---
    print("\n[Tier 2] Classical GA (Lamarckian Repair Mode)...")
    t0 = time.time()
    classical_ga = ModularCFLPGASolver(dataset=dataset, mode="repair", heuristic_ratio=0.5)
    best_cost_ga, best_ind_ga, history_ga = classical_ga.solve(
        pop_size=50, n_gen=100, cx_pb=0.80, mut_pb=0.20, elite_count=1
    )
    classical_ga_time = time.time() - t0
    results["classical_ga"] = {
        "cost": best_cost_ga,
        "time": classical_ga_time,
        "history": history_ga
    }
    print(f"  Classical GA cost: ${best_cost_ga:,.2f} | Time: {classical_ga_time:.2f}s")

    # --- MILP Reference ---
    print("\n[Ref] MILP Exact Solver...")
    t0 = time.time()
    milp = MILPSolver(dataset)
    milp_cost, _, _, milp_status = milp.solve()
    milp_time = time.time() - t0
    print(f"  MILP cost: ${milp_cost:,.2f} | Status: {milp_status} | Time: {milp_time:.3f}s")

    # --- Tier 3a: Hybrid ML-GA -- Pure Surrogate (XGBoost) ---
    print("\n[Tier 3a] Hybrid ML-GA -- Pure Surrogate (XGBoost)...")
    xgb_model = CFLPSurrogateModel.load(
        os.path.join(model_dir, "surrogate_xgboost.pkl")
    )
    hybrid_xgb = HybridMLGASolver(
        dataset=dataset, surrogate=xgb_model,
        pop_size=50, n_generations=100,
        mode="pure_surrogate", random_seed=42
    )
    hybrid_xgb_result = hybrid_xgb.solve()
    results["hybrid_xgb"] = {
        "cost": hybrid_xgb_result["best_cost"],
        "surrogate_cost": hybrid_xgb_result["best_surrogate_cost"],
        "time": hybrid_xgb_result["elapsed_time"],
        "history": hybrid_xgb_result["history"],
        "surrogate_evals": hybrid_xgb_result["surrogate_eval_count"],
        "exact_evals": hybrid_xgb_result["exact_eval_count"],
        "surrogate_error_pct": hybrid_xgb_result["surrogate_error_pct"]
    }
    print(f"  Hybrid XGB cost: ${hybrid_xgb_result['best_cost']:,.2f} | "
          f"Time: {hybrid_xgb_result['elapsed_time']:.2f}s")

    # --- Tier 3b: Hybrid ML-GA — Confidence-Aware (Random Forest) ---
    print("\n[Tier 3b] Hybrid ML-GA -- Confidence-Aware (Random Forest)...")
    rf_model = CFLPSurrogateModel.load(
        os.path.join(model_dir, "surrogate_random_forest.pkl")
    )
    hybrid_rf = HybridMLGASolver(
        dataset=dataset, surrogate=rf_model,
        pop_size=50, n_generations=100,
        mode="confidence_aware",
        uncertainty_threshold_pct=5.0,
        warmup_fraction=0.20,
        random_seed=42
    )
    hybrid_rf_result = hybrid_rf.solve()
    results["hybrid_rf"] = {
        "cost": hybrid_rf_result["best_cost"],
        "time": hybrid_rf_result["elapsed_time"],
        "history": hybrid_rf_result["history"],
        "surrogate_evals": hybrid_rf_result["surrogate_eval_count"],
        "exact_evals": hybrid_rf_result["exact_eval_count"],
        "surrogate_error_pct": hybrid_rf_result["surrogate_error_pct"]
    }
    print(f"  Hybrid RF cost: ${hybrid_rf_result['best_cost']:,.2f} | "
          f"Time: {hybrid_rf_result['elapsed_time']:.2f}s")

    # --- Final Summary ---
    _print_final_summary(results, milp_cost)

    # --- Plot ---
    _plot_comparison(results, milp_cost, output_dir)


def _print_final_summary(results: Dict[str, Any], milp_cost: float) -> None:
    """Prints a structured 3-tier comparison table."""
    def gap(cost): return (cost - milp_cost) / milp_cost * 100 if milp_cost > 0 else 0

    print("\n" + "=" * 90)
    print(f"  {'FULL 3-TIER SOLVER COMPARISON SUMMARY':^86}")
    print("=" * 90)
    print(f"  {'Solver':<30} | {'Best Cost ($)':>18} | {'Gap (%)':>8} | {'Time (s)':>10}")
    print("-" * 90)
    print(f"  {'MILP Exact (Reference)':<30} | ${milp_cost:>17,.2f} | {'0.0000':>8}% | {'--':>10}")
    print(f"  {'Greedy Heuristic':<30} | ${results['greedy']['cost']:>17,.2f} | "
          f"{gap(results['greedy']['cost']):>8.4f}% | {results['greedy']['time']:>10.3f}s")
    print(f"  {'Classical GA (Repair)':<30} | ${results['classical_ga']['cost']:>17,.2f} | "
          f"{gap(results['classical_ga']['cost']):>8.4f}% | {results['classical_ga']['time']:>10.2f}s")
    print(f"  {'Hybrid ML-GA (XGBoost, Pure)':<30} | ${results['hybrid_xgb']['cost']:>17,.2f} | "
          f"{gap(results['hybrid_xgb']['cost']):>8.4f}% | {results['hybrid_xgb']['time']:>10.2f}s")
    print(f"  {'Hybrid ML-GA (RF, Conf-Aware)':<30} | ${results['hybrid_rf']['cost']:>17,.2f} | "
          f"{gap(results['hybrid_rf']['cost']):>8.4f}% | {results['hybrid_rf']['time']:>10.2f}s")
    print("=" * 90)

    # Speedup of Hybrid vs Classical
    if results['classical_ga']['time'] > 0:
        speedup_xgb = results['classical_ga']['time'] / results['hybrid_xgb']['time']
        speedup_rf = results['classical_ga']['time'] / results['hybrid_rf']['time']
        print(f"\n  Runtime Speedup (vs Classical GA):")
        print(f"    Hybrid XGBoost : {speedup_xgb:.1f}x faster")
        print(f"    Hybrid RF      : {speedup_rf:.1f}x faster")


def _plot_comparison(results: Dict[str, Any], milp_cost: float, output_dir: str) -> None:
    """Saves a 2-panel comparison plot: convergence curves + runtime bar chart."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CFLP Solver Comparison: Classical GA vs. Hybrid ML-GA", fontsize=13, fontweight="bold")

    # --- Panel 1: Convergence Curves ---
    ax1 = axes[0]
    ax1.axhline(milp_cost, color="black", linestyle="--", linewidth=1.5, label="MILP Optimum")
    ax1.axhline(results["greedy"]["cost"], color="red", linestyle=":", linewidth=1.5, label="Greedy Heuristic")

    if "history" in results["classical_ga"]:
        h = results["classical_ga"]["history"]
        ax1.plot(h["gen"], h["min_cost"], color="royalblue", linewidth=1.8, label="Classical GA")

    if "history" in results["hybrid_xgb"]:
        h = results["hybrid_xgb"]["history"]
        ax1.plot(h["gen"], h["min_cost"], color="darkorange", linewidth=1.8,
                 linestyle="-", label="Hybrid XGBoost (Pure)")

    if "history" in results["hybrid_rf"]:
        h = results["hybrid_rf"]["history"]
        ax1.plot(h["gen"], h["min_cost"], color="green", linewidth=1.8,
                 linestyle="-.", label="Hybrid RF (Conf-Aware)")

    ax1.set_xlabel("Generation", fontsize=11)
    ax1.set_ylabel("Best Cost ($)", fontsize=11)
    ax1.set_title("Convergence Curves", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(9, 9))
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Runtime Comparison ---
    ax2 = axes[1]
    labels = ["Greedy", "Classical GA", "Hybrid XGB\n(Pure)", "Hybrid RF\n(Conf-Aware)"]
    times = [
        results["greedy"]["time"],
        results["classical_ga"]["time"],
        results["hybrid_xgb"]["time"],
        results["hybrid_rf"]["time"]
    ]
    colors = ["salmon", "royalblue", "darkorange", "green"]
    bars = ax2.bar(labels, times, color=colors, edgecolor="black", linewidth=0.7)
    ax2.set_ylabel("Execution Time (seconds)", fontsize=11)
    ax2.set_title("Runtime Comparison", fontsize=11)
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3,
                 f"{t:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "hybrid_ga_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison plot saved to: {save_path}")


def extract_training_data_from_ga(result: dict, dataset=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts training data from exact evaluations collected during a hybrid GA run.

    CRITICAL: GA collects TOTAL COST (fixed + transport), but TrainingPipeline expects
    TRANSPORT COST ONLY. If dataset is provided, this function corrects the format.
    If dataset is None, returns raw data (use at your own risk).

    Use this to enable GA-derived sampling workflow:
      1. Run hybrid GA with warmup period (ensures some exact evaluations)
      2. Extract training data via this function
      3. Train ML surrogate on GA-derived data
      4. Run hybrid GA again with new surrogate

    Args:
        result (dict): Result dict returned by HybridMLGASolver.solve()
                       Must contain key 'exact_evaluations_log'
        dataset (CFLPDataset, optional): Required to correct data format from total cost to transport cost.
                                         If None, returns raw data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) training dataset
            X shape (N, m) — binary chromosomes
            y shape (N,)   — transport costs only (if dataset provided) OR total costs (if dataset=None)

    Raises:
        KeyError: If result missing 'exact_evaluations_log'
        ValueError: If exact_evaluations_log is empty
    """
    if "exact_evaluations_log" not in result:
        raise KeyError("Result dict missing 'exact_evaluations_log'. Ensure GA was configured with warmup_fraction > 0.")

    exact_log = result["exact_evaluations_log"]
    if not exact_log:
        raise ValueError("No exact evaluations logged (warmup period may be 0 or exceeded generation limit).")

    chromosomes = np.array([item[0] for item in exact_log], dtype=np.int32)
    costs_total = np.array([item[1] for item in exact_log], dtype=np.float64)

    # De-duplicate by chromosome: elitism and population convergence naturally cause
    # the GA to re-evaluate the same chromosome across generations. Duplicate rows add
    # no training signal and risk train/test leakage in SurrogateTrainingPipeline.
    n_before = chromosomes.shape[0]
    _, unique_indices = np.unique(chromosomes, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    chromosomes = chromosomes[unique_indices]
    costs_total = costs_total[unique_indices]
    n_dupes = n_before - chromosomes.shape[0]
    if n_dupes > 0:
        print(f"[extract_training_data_from_ga] Removed {n_dupes:,} duplicate chromosome(s) "
              f"({n_before:,} -> {chromosomes.shape[0]:,} unique samples)")

    # CRITICAL FIX: GA collects TOTAL COST (fixed + transport)
    # but TrainingPipeline expects TRANSPORT COST ONLY
    if dataset is not None:
        fixed_costs = chromosomes @ dataset.fixed_costs
        costs_transport = costs_total - fixed_costs
        return chromosomes, costs_transport
    else:
        return chromosomes, costs_total


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    run_comparison_experiment(
        dataset_path=os.path.join(base_dir, "..", "data", "raw", "cap41.txt"),
        model_dir=os.path.join(base_dir, "..", "data", "processed"),
        output_dir=os.path.join(base_dir, "..", "docs")
    )
