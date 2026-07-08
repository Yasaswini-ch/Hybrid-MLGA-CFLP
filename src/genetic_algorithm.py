import os
import random
import time
from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from deap import base, creator, tools

from parser import CFLPDataset
from baseline import MILPSolver
from baseline_solver import GreedyBaselineSolver
from chromosome import CFLPChromosome
from population import CFLPPopulationGenerator
from fitness import CFLPFitnessEvaluator
from repair import CFLPFeasibilityRepairer
from selection import tournament_select, roulette_select, apply_elitism
from crossover import single_point_crossover, two_point_crossover, uniform_crossover
from mutation import bit_flip_mutation

class ModularCFLPGASolver:
    """
    A highly modular, research-grade classical Genetic Algorithm (GA) solver for the CFLP.
    Integrates decoupled chromosome, population, selection, crossover, mutation, and repair modules.
    """
    
    def __init__(self, 
                 dataset: CFLPDataset, 
                 mode: str = "repair", 
                 heuristic_ratio: float = 0.5):
        """
        Initializes the Genetic Algorithm Solver.
        
        Args:
            dataset (CFLPDataset): Parsed CFLP instance.
            mode (str): Optimization constraint handling mode - 'repair' or 'penalty'.
            heuristic_ratio (float): Fraction of initial population to seed heuristically (0.0 to 1.0).
        """
        self.dataset = dataset
        self.m = dataset.num_facilities
        self.mode = mode.lower()
        self.heuristic_ratio = heuristic_ratio
        
        # Instantiate modular components
        self.pop_gen = CFLPPopulationGenerator(self.dataset)
        self.evaluator = CFLPFitnessEvaluator(self.dataset)
        self.repairer = CFLPFeasibilityRepairer(self.dataset)
        
        # Setup DEAP
        self._setup_deap()
        
    def _setup_deap(self) -> None:
        """
        Registers DEAP definitions and wraps custom modular operators in the toolbox.
        """
        # 1. Define Fitness objective: Minimization (negative weight)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            
        # 2. Define Individual container: List wrapping our minimization fitness
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
        self.toolbox = base.Toolbox()
        
        # 3. Register custom modular evaluation function
        self.toolbox.register("evaluate", self.evaluator.evaluate)
        
        # 4. Register custom modular evolutionary operators
        self.toolbox.register("mate", two_point_crossover)
        self.toolbox.register("mutate", bit_flip_mutation, indpb=(1.0 / self.m))
        self.toolbox.register("select", tournament_select, tournsize=3)
        
    def solve(self, 
              pop_size: int = 50, 
              n_gen: int = 100, 
              cx_pb: float = 0.8, 
              mut_pb: float = 0.2,
              elite_count: int = 1) -> Tuple[float, List[int], Dict[str, Any]]:
        """
        Executes the generational Genetic Algorithm with Elitism and custom constraint handling.
        
        Args:
            pop_size (int): Size of the population.
            n_gen (int): Number of generations to evolve.
            cx_pb (float): Crossover probability.
            mut_pb (float): Mutation probability.
            elite_count (int): Number of elite individuals to preserve each generation.
            
        Returns:
            Tuple[float, List[int], Dict[str, Any]]:
                - best_cost (float): Lowest cost discovered.
                - best_individual (List[int]): Fittest binary chromosome.
                - history (Dict[str, Any]): Convergence logs and diversity history.
        """
        start_time = time.time()
        
        # 1. Initialize Population using modular population generator
        raw_pop = self.pop_gen.create_population(pop_size, self.heuristic_ratio)
        
        # Convert raw lists into DEAP creator.Individual objects
        pop = [creator.Individual(ind) for ind in raw_pop]
        
        # 2. Feasibility repair of initial population (if in repair mode)
        if self.mode == "repair":
            for ind in pop:
                self.repairer.repair(ind)
                
        # 3. Evaluate initial population fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(self.toolbox.map(self.toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # 4. Tracking and Convergence logs
        history = {
            "gen": [],
            "min_cost": [],
            "avg_cost": [],
            "feasible_pct": [],
            "avg_diversity": []  # Average Hamming distance to best individual
        }
        
        # Find absolute best individual (Elitism initialization)
        best_ind = tools.selBest(pop, 1)[0]
        best_cost = best_ind.fitness.values[0]
        
        # Generational Evolutionary Loop
        for g in range(n_gen):
            # A. SELECTION: Select offspring from parents
            offspring = self.toolbox.select(pop, len(pop))
            # Clone offspring to decouple references
            offspring = list(map(self.toolbox.clone, offspring))
            
            # B. CROSSOVER: Recombine offspring pairs
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_pb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    
            # C. MUTATION: Apply bit-flip mutations
            for mutant in offspring:
                if random.random() < mut_pb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
                    
            # D. FEASIBILITY REPAIR: Apply Lamarckian Repair before fitness evaluation
            if self.mode == "repair":
                for ind in offspring:
                    if not ind.fitness.valid:  # Only repair modified individuals
                        self.repairer.repair(ind)
                        
            # E. EVALUATION: Re-evaluate un-fit individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(self.toolbox.map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            # F. ELITISM FILTER: Write previous generation's elite into current offspring
            offspring = apply_elitism(pop, offspring, elite_count)
            
            # G. UPDATE POPULATION
            pop[:] = offspring
            
            # H. RECORD STATS
            fits = [ind.fitness.values[0] for ind in pop]
            feasible_fits = [f for f in fits if f < 1e11]
            
            min_val = min(feasible_fits) if feasible_fits else 1e12
            avg_val = np.mean(feasible_fits) if feasible_fits else 1e12
            feas_pct = (len(feasible_fits) / len(pop)) * 100.0
            
            # Find current generation's best individual
            gen_best = tools.selBest(pop, 1)[0]
            if gen_best.fitness.values[0] < best_cost:
                best_ind = self.toolbox.clone(gen_best)
                best_cost = gen_best.fitness.values[0]
                
            # Compute structural population diversity around current best individual
            best_chrom = CFLPChromosome(np.array(best_ind, dtype=np.int32))
            hamming_distances = [
                best_chrom.hamming_distance(CFLPChromosome(np.array(ind, dtype=np.int32))) 
                for ind in pop
            ]
            avg_diversity = np.mean(hamming_distances)
            
            history["gen"].append(g)
            history["min_cost"].append(min_val)
            history["avg_cost"].append(avg_val)
            history["feasible_pct"].append(feas_pct)
            history["avg_diversity"].append(avg_diversity)
            
            if g % 10 == 0 or g == n_gen - 1:
                print(f"[{self.mode.upper()}] Gen {g:3d}: Min Cost = ${min_val:14,.2f} | Feasible = {feas_pct:5.1f}% | Div = {avg_diversity:4.2f}")
                
        history["total_time_sec"] = time.time() - start_time
        return best_cost, list(best_ind), history

def plot_experiments(penalty_hist: Dict[str, Any], 
                     repair_hist: Dict[str, Any], 
                     milp_cost: float,
                     greedy_cost: float,
                     save_path: str) -> None:
    """
    Plots the comparative convergence curves of Penalty vs. Repair GA modes, 
    along with MILP exact and Greedy heuristic reference baselines.
    """
    plt.figure(figsize=(12, 7))
    
    # 1. Plot Pure Penalty Min Cost
    plt.plot(penalty_hist["gen"], penalty_hist["min_cost"], 
             label="GA Pure Penalty Mode (Minimum)", color="#d62728", linestyle=":", linewidth=2)
             
    # 2. Plot Lamarckian Repair Min Cost
    plt.plot(repair_hist["gen"], repair_hist["min_cost"], 
             label="GA Lamarckian Repair Mode (Minimum)", color="#1f77b4", linewidth=2.5)
             
    # 3. Plot Lamarckian Repair Avg Cost
    plt.plot(repair_hist["gen"], repair_hist["avg_cost"], 
             label="GA Lamarckian Repair Mode (Population Average)", color="#ff7f0e", linestyle="--", linewidth=1.5)
             
    # 4. Plot Reference Baselines
    plt.axhline(y=milp_cost, color="#2ca02c", linestyle="-.", label="Exact MILP Solver (Optimal)", alpha=0.9, linewidth=2)
    plt.axhline(y=greedy_cost, color="#7f7f7f", linestyle="--", label="Nearest Feasible Heuristic", alpha=0.7, linewidth=1.5)
    
    plt.title("Modular CFLP GA Convergence Curve Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Objective Cost ($)", fontsize=12)
    
    # Format Y-axis with comma separators
    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda val, loc: f"${val:,.0f}"))
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nComparative convergence plot successfully saved to: {save_path}")

def run_experiments():
    """
    Executes a comprehensive, controlled scientific experiment on cap41.txt,
    comparing MILP exact, Greedy Baseline, Pure Penalty GA, and Lamarckian Repair GA.
    """
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "..", "data", "raw", "cap41.txt")
    plot_path = os.path.join(base_dir, "..", "docs", "cap41_ga_convergence.png")
    
    print("=" * 90)
    print(f"{'LAUNCHING MODULAR CLASSICAL GENETIC ALGORITHM EXPERIMENTS':^90}")
    print("=" * 90)
    
    dataset = CFLPDataset(file_path)
    
    # --- Stage 1: Exact MILP Baseline ---
    print("\n[Stage 1] Solving via Exact MILP Solver (CBC)...")
    milp = MILPSolver(dataset)
    milp_cost, milp_y, _, milp_time = milp.solve()
    print(f"  MILP Total Cost: ${milp_cost:,.2f} (Active footprint: {int(np.sum(milp_y))}/{dataset.num_facilities} open)")
    
    # --- Stage 2: Greedy Heuristic Baseline ---
    print("\n[Stage 2] Solving via Nearest Feasible Heuristic...")
    greedy_solver = GreedyBaselineSolver(dataset)
    greedy_sol = greedy_solver.solve()
    demands_safe = np.where(dataset.demands > 0, dataset.demands, 1.0)
    greedy_cost = np.sum(greedy_sol.y * dataset.fixed_costs) + np.sum(
        (greedy_sol.x / demands_safe[:, np.newaxis]) * dataset.transport_costs
    )
    print(f"  Greedy Total Cost: ${greedy_cost:,.2f} (Active footprint: {int(np.sum(greedy_sol.y))}/{dataset.num_facilities} open)")
    
    # GA Parameters for controlled experiments
    POP_SIZE = 50
    GENERATIONS = 100
    CROSSOVER_PB = 0.8
    MUTATION_PB = 0.2
    HEURISTIC_RATIO = 0.5  # Seed 50% heuristically, 50% randomly
    
    # --- Stage 3: GA Pure Penalty Mode ---
    print("\n[Stage 3] Running GA in Pure Penalty Mode...")
    penalty_solver = ModularCFLPGASolver(dataset, mode="penalty", heuristic_ratio=HEURISTIC_RATIO)
    p_best_cost, p_best_y, p_history = penalty_solver.solve(
        pop_size=POP_SIZE, n_gen=GENERATIONS, cx_pb=CROSSOVER_PB, mut_pb=MUTATION_PB
    )
    
    # --- Stage 4: GA Lamarckian Repair Mode ---
    print("\n[Stage 4] Running GA in Lamarckian Repair Mode...")
    repair_solver = ModularCFLPGASolver(dataset, mode="repair", heuristic_ratio=HEURISTIC_RATIO)
    r_best_cost, r_best_y, r_history = repair_solver.solve(
        pop_size=POP_SIZE, n_gen=GENERATIONS, cx_pb=CROSSOVER_PB, mut_pb=MUTATION_PB
    )
    
    # --- Scientific Comparative Summary ---
    p_opt_gap = ((p_best_cost - milp_cost) / milp_cost) * 100.0
    r_opt_gap = ((r_best_cost - milp_cost) / milp_cost) * 100.0
    greedy_opt_gap = ((greedy_cost - milp_cost) / milp_cost) * 100.0
    
    print("\n" + "=" * 90)
    print(f"{'EXPERIMENTAL COMPARATIVE REPORT ON cap41.txt':^90}")
    print("=" * 90)
    print(f"{'Solver / Configuration':<30} | {'Objective Cost ($)':<22} | {'Active Set':<12} | {'Optimality Gap (%)':<18}")
    print("-" * 90)
    print(f"{'Exact MILP (CBC)':<30} | ${milp_cost:<21,.2f} | {int(np.sum(milp_y)):>2}/16       | {'0.0000%':<18}")
    print(f"{'Greedy Heuristic Baseline':<30} | ${greedy_cost:<21,.2f} | {int(np.sum(greedy_sol.y)):>2}/16       | {greedy_opt_gap:<17.4f}%")
    print(f"{'GA Pure Penalty Mode':<30} | ${p_best_cost:<21,.2f} | {int(np.sum(p_best_y)):>2}/16       | {p_opt_gap:<17.4f}%")
    print(f"{'GA Lamarckian Repair Mode':<30} | ${r_best_cost:<21,.2f} | {int(np.sum(r_best_y)):>2}/16       | {r_opt_gap:<17.4f}%")
    print("-" * 90)
    print(f"GA Execution Times: Penalty Mode = {p_history['total_time_sec']:.2f}s | Repair Mode = {r_history['total_time_sec']:.2f}s")
    print("=" * 90)
    
    # Save comparison convergence plot
    plot_experiments(p_history, r_history, milp_cost, greedy_cost, plot_path)

if __name__ == "__main__":
    run_experiments()
