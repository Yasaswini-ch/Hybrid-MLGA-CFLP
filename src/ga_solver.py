import os
import random
import time
from typing import List, Tuple, Dict, Any
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import matplotlib.pyplot as plt

from deap import base, creator, tools
from parser import CFLPDataset
from baseline import MILPSolver

class CFLPGASolver:
    """
    An Optimized Classical Genetic Algorithm Solver for the Capacitated Facility Location Problem (CFLP)
    using the DEAP framework and SciPy's in-memory HiGHS Linear Programming solver.
    
    Representation:
        - Chromosome: Binary array of length m (num_facilities), where:
          y[i] = 1 indicates facility i is open.
          y[i] = 0 indicates facility i is closed.
    """
    
    def __init__(self, dataset: CFLPDataset):
        self.dataset = dataset
        self.num_facilities = dataset.num_facilities
        self.num_customers = dataset.num_customers
        self.total_demand = np.sum(dataset.demands)
        
        # Determine minimum facilities required for physical capacity feasibility
        self.min_facilities_needed = int(np.ceil(self.total_demand / dataset.capacities[0]))
        
        # Determine maximum facilities allowed to open to constrain search space size
        if self.num_facilities <= 50:
            self.max_facilities_to_open = self.num_facilities
        else:
            self.max_facilities_to_open = self.min_facilities_needed + 15
        
        # Cache for fitness evaluations to speed up identical chromosomes
        self.cache = {}
        
        # Configure DEAP environment
        self._setup_deap()
        
    def clear_cache(self) -> None:
        """Clears the evaluation cache."""
        self.cache.clear()
        
    def _setup_deap(self) -> None:
        """
        Initializes the DEAP creator, fitness definitions, and toolbox registration.
        """
        # 1. Define Optimization Objective: Minimization
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            
        # 2. Define Individual Structure: A list of integers with our minimization fitness
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
        self.toolbox = base.Toolbox()
        
        # 3. Custom Individual Generator: Ensure a high chance of starting feasible
        self.toolbox.register("attr_binary", self._generate_smart_individual)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_binary)
        
        # 4. Define Population: A list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 5. Register Evolutionary Operators
        self.toolbox.register("evaluate", self.evaluate_fitness)
        self.toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
        # CORRECTED: Adaptive mutation probability (was hardcoded at 0.05 = 5%)
        # Standard GA: mutation probability = 1/chromosome_length = 1/num_facilities
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=(1.0 / self.num_facilities))
        self.toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection
        
    def _generate_smart_individual(self) -> List[int]:
        """
        Generates a random binary chromosome with at least `min_facilities_needed` 
        facilities set to open. For large instances, we constrain the initial opened 
        facilities to speed up execution.
        """
        individual = [0] * self.num_facilities

        # CORRECTED: Removed artificial constraint on large instances
        # (Previously limited large instances to min_facilities + 8, reducing exploration)
        # Now allow full range from min_facilities_needed to all_facilities for all instances
        num_to_open = random.randint(self.min_facilities_needed, self.num_facilities)

        # Randomly select which indices to open
        open_indices = random.sample(range(self.num_facilities), num_to_open)
        for idx in open_indices:
            individual[idx] = 1
            
        return individual
        
    def evaluate_fitness(self, individual: List[int]) -> Tuple[float,]:
        """
        Evaluates the fitness of a chromosome by solving the continuous 
        transportation sub-problem in-memory using SciPy's HiGHS solver.
        This avoids process creation and disk I/O overhead.
        
        Args:
            individual (List[int]): The binary chromosome representation.
            
        Returns:
            Tuple[float,]: A 1-tuple containing the total cost (fitness).
        """
        # Check evaluation cache first
        chromo_key = tuple(individual)
        if chromo_key in self.cache:
            return self.cache[chromo_key]
            
        y_val = np.array(individual)
        
        # --- A. Physical Capacity and Max Open Facilities Check ---
        open_capacity = np.sum(self.dataset.capacities * y_val)
        num_open = np.sum(y_val)
        if open_capacity < self.total_demand or num_open > self.max_facilities_to_open:
            self.cache[chromo_key] = (1e12,)
            return (1e12,)  # Massive penalty cost
            
        # --- B. Solve Continuous Transportation LP via SciPy ---
        open_indices = np.where(y_val == 1)[0]
        num_open = len(open_indices)
        
        # --- B1. Quick UFLP Feasibility Check (Mathematical Shortcut) ---
        # For each customer, find the cheapest open facility
        cheapest_idx = np.argmin(self.dataset.transport_costs[:, open_indices], axis=1)
        
        # Calculate the capacity loaded on each open facility
        loaded_demands = np.zeros(num_open)
        np.add.at(loaded_demands, cheapest_idx, self.dataset.demands)
        
        # If this assignment satisfies all capacity constraints, UFLP assignment is optimal for CFLP!
        if np.all(loaded_demands <= self.dataset.capacities[open_indices]):
            transport_cost = np.sum(self.dataset.transport_costs[np.arange(self.num_customers), open_indices[cheapest_idx]])
            fixed_cost = np.sum(self.dataset.fixed_costs * y_val)
            total_cost = fixed_cost + transport_cost
            result = (total_cost,)
            self.cache[chromo_key] = result
            return result
            
        # Variables: w[j, k] representing fraction of demand from open facility open_indices[k] to customer j.
        # Total number of continuous variables = num_customers * num_open.
        
        # 1. Objective Coefficients (Flattened cost array)
        c = self.dataset.transport_costs[:, open_indices].flatten()
        
        # 2. Equality Constraints: Customer demands satisfied (sum_{k} w_{j,k} == 1.0)
        n = self.num_customers
        rows_eq = np.repeat(np.arange(n), num_open)
        cols_eq = np.arange(n * num_open)
        data_eq = np.ones(n * num_open)
        A_eq = sp.coo_matrix((data_eq, (rows_eq, cols_eq)), shape=(n, n * num_open)).tocsr()
        b_eq = np.ones(n)
        
        # 3. Inequality Constraints: Open facility capacity bounds (sum_{j} d_j w_{j,k} <= capacities[k])
        rows_ub = np.repeat(np.arange(num_open), n)
        cols_ub = np.arange(n * num_open).reshape(n, num_open).T.flatten()
        data_ub = np.tile(self.dataset.demands, num_open)
        A_ub = sp.coo_matrix((data_ub, (rows_ub, cols_ub)), shape=(num_open, n * num_open)).tocsr()
        b_ub = self.dataset.capacities[open_indices]
        
        # 4. Bounds (w_jk >= 0)
        bounds = [(0.0, None)] * len(c)
        
        # 5. Solve LP in-memory using SciPy's Highs solver
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res.success:
            self.cache[chromo_key] = (1e12,)
            return (1e12,)
            
        # Sum optimal transport costs + fixed costs of open facilities
        transport_cost = res.fun
        fixed_cost = np.sum(self.dataset.fixed_costs * y_val)
        total_cost = fixed_cost + transport_cost
        
        result = (total_cost,)
        self.cache[chromo_key] = result
        return result

    def solve(self, 
              pop_size: int = 50, 
              n_gen: int = 100, 
              cx_pb: float = 0.8, 
              mut_pb: float = 0.2) -> Tuple[float, List[int], Dict[str, Any]]:
        """
        Executes the generational Genetic Algorithm with elitism.
        
        Args:
            pop_size (int): Size of the population.
            n_gen (int): Number of generations.
            cx_pb (float): Crossover probability.
            mut_pb (float): Mutation probability.
            
        Returns:
            Tuple[float, List[int], Dict[str, Any]]:
                - best_cost (float): Lowest cost discovered.
                - best_individual (List[int]): Fittest binary chromosome.
                - history (Dict[str, Any]): Dictionary containing evolution history for plotting.
        """
        start_time = time.time()

        # NOTE: A ThreadPool-based parallel evaluation path used to be enabled here for
        # instances with >50 facilities. It reliably caused native segmentation faults
        # (confirmed by direct testing: fresh, isolated processes crashed with SIGSEGV
        # specifically and only on 100-facility instances, never on <=50-facility ones,
        # even at drastically reduced population/generation budgets). SciPy's linprog
        # (HiGHS backend) and sparse matrix construction are not guaranteed thread-safe;
        # concurrent calls from multiple ThreadPool worker threads sharing the same
        # process/native library state corrupted memory under large-instance workloads.
        # Sequential evaluation is slower but correct.
        pool = None
        
        # Create initial population
        pop = self.toolbox.population(n=pop_size)
        
        # Evaluate initial population fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(self.toolbox.map(self.toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Track statistics
        history = {
            "gen": [],
            "min_cost": [],
            "avg_cost": [],
            "feasible_pct": []
        }

        # Save absolute best individual (Elitism)
        best_ind = tools.selBest(pop, 1)[0]
        best_cost = best_ind.fitness.values[0]

        # Convergence detection (early termination)
        best_cost_history = [best_cost]
        stagnation_counter = 0
        STAGNATION_LIMIT = 10  # Stop if no improvement for N consecutive generations
        MIN_IMPROVEMENT_THRESHOLD = 0.0001  # Improvement < 0.01% counts as stagnation

        # Run generational loop
        for g in range(n_gen):
            # 1. Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # 2. Apply Crossover (on pairs)
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_pb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    
            # 3. Apply Mutation
            for mutant in offspring:
                if random.random() < mut_pb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
                    
            # 4. Re-evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(self.toolbox.map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            # 5. Elitism: Replace worst offspring with the best so far
            offspring_best = tools.selBest(offspring, 1)[0]
            if offspring_best.fitness.values[0] > best_cost:
                worst_idx = np.argmax([ind.fitness.values[0] for ind in offspring])
                offspring[worst_idx] = self.toolbox.clone(best_ind)
            else:
                best_ind = self.toolbox.clone(offspring_best)
                best_cost = best_ind.fitness.values[0]
                
            pop[:] = offspring
            
            # Record generation stats
            fits = [ind.fitness.values[0] for ind in pop]
            feasible_fits = [f for f in fits if f < 1e11]
            
            min_val = min(feasible_fits) if feasible_fits else 1e12
            avg_val = np.mean(feasible_fits) if feasible_fits else 1e12
            feas_pct = (len(feasible_fits) / len(pop)) * 100.0
            
            history["gen"].append(g)
            history["min_cost"].append(min_val)
            history["avg_cost"].append(avg_val)
            history["feasible_pct"].append(feas_pct)

            # CONVERGENCE CHECK (NEW): Detect stagnation and terminate early
            if best_cost < best_cost_history[-1]:
                # Improvement detected
                improvement_pct = ((best_cost_history[-1] - best_cost) / best_cost_history[-1]) * 100.0
                if improvement_pct < MIN_IMPROVEMENT_THRESHOLD:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0  # Reset counter if improvement is significant
                best_cost_history.append(best_cost)
            else:
                # No improvement
                stagnation_counter += 1

            # Terminate early if stagnation threshold reached
            if stagnation_counter >= STAGNATION_LIMIT:
                if g % 10 == 0 or g == n_gen - 1:
                    print(f"Gen {g:3d}: Min Cost = ${min_val:15,.2f} | Feasible = {feas_pct:5.1f}% | Best Cost So Far = ${best_cost:15,.2f} [TERMINATING - Stagnation detected]")
                print(f"[Convergence] Early termination at generation {g}/{n_gen} (no improvement for {STAGNATION_LIMIT} generations)")
                break

            if g % 10 == 0 or g == n_gen - 1:
                print(f"Gen {g:3d}: Min Cost = ${min_val:15,.2f} | Feasible = {feas_pct:5.1f}% | Best Cost So Far = ${best_cost:15,.2f}")
                
        history["total_time_sec"] = time.time() - start_time
        
        if pool is not None:
            pool.close()
            pool.join()
            self.toolbox.register("map", map)  # Restore default sequential map
            
        return best_cost, list(best_ind), history

def plot_convergence(history: Dict[str, Any], dataset_name: str, save_path: str) -> None:
    """
    Plots the minimum and average fitness convergence over generations.
    """
    plt.figure(figsize=(10, 6))
    
    gens = history["gen"]
    mins = history["min_cost"]
    avgs = history["avg_cost"]
    
    plt.plot(gens, mins, label="Best Cost (Min)", color="#1f77b4", linewidth=2.5)
    plt.plot(gens, avgs, label="Population Average Cost", color="#ff7f0e", linestyle="--", linewidth=1.5)
    
    plt.title(f"Genetic Algorithm Convergence Curve ({dataset_name})", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Total Cost ($)", fontsize=12)
    
    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:,.0f}"))
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Convergence plot saved successfully to: {save_path}")

def run_classical_ga():
    """
    Verifies and runs the Genetic Algorithm on cap41.txt and compares it directly
    with the optimal solution obtained from the MILP baseline solver.
    """
    base_dir = os.path.dirname(__file__)
    raw_path = os.path.join(base_dir, "..", "data", "raw", "cap41.txt")
    docs_dir = os.path.join(base_dir, "..", "docs")
    
    print("=" * 80)
    print("RUNNING OPTIMIZED CLASSICAL GENETIC ALGORITHM ON cap41")
    print("=" * 80)
    
    dataset = CFLPDataset(raw_path)
    
    print("\n[Step 1] Running exact MILP solver for optimal benchmark...")
    milp = MILPSolver(dataset)
    opt_cost, opt_y, _, _ = milp.solve()
    print(f"MILP Optimal Cost: ${opt_cost:,.2f} (Opened: {np.sum(opt_y)}/{dataset.num_facilities} facilities)")
    
    print("\n[Step 2] Launching DEAP Genetic Algorithm (In-Memory SciPy Solves)...")
    ga_solver = CFLPGASolver(dataset)
    
    POP_SIZE = 50
    GENERATIONS = 100
    CROSSOVER_PROB = 0.8
    MUTATION_PROB = 0.2
    
    best_cost, best_y, history = ga_solver.solve(
        pop_size=POP_SIZE, 
        n_gen=GENERATIONS, 
        cx_pb=CROSSOVER_PROB, 
        mut_pb=MUTATION_PROB
    )
    
    opt_gap = ((best_cost - opt_cost) / opt_cost) * 100.0
    print("\n" + "=" * 80)
    print("FINAL COMPARATIVE RESULTS")
    print("=" * 80)
    print(f"MILP Optimal Cost: ${opt_cost:,.2f} (Opened: {int(np.sum(opt_y))}/16)")
    print(f"GA Solver Cost   : ${best_cost:,.2f} (Opened: {int(np.sum(best_y))}/16)")
    print(f"GA Optimality Gap: {opt_gap:.4f}%")
    print(f"GA Active Vector : {best_y}")
    print(f"GA Compute Time  : {history['total_time_sec']:.2f} seconds")
    print("=" * 80)
    
    plot_name = "cap41_ga_convergence.png"
    plot_path = os.path.join(docs_dir, plot_name)
    plot_convergence(history, dataset.name, plot_path)
    
if __name__ == "__main__":
    run_classical_ga()
