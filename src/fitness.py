from typing import Tuple
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from parser import CFLPDataset
from solution_representation import CFLPSolution
from cost_calculator import calculate_total_cost
from constraint_checker import is_feasible

class CFLPFitnessEvaluator:
    """
    Evaluates the objective fitness of binary facility opening vectors (chromosomes)
    by solving the continuous transportation sub-problem in-memory using SciPy's HiGHS solver.
    """
    
    def __init__(self, dataset: CFLPDataset):
        """
        Initializes the evaluator.
        
        Args:
            dataset (CFLPDataset): Parsed CFLP instance.
        """
        self.dataset = dataset
        self.m = dataset.num_facilities
        self.n = dataset.num_customers
        self.total_demand = np.sum(dataset.demands)
        
    def evaluate(self, individual: list) -> Tuple[float,]:
        """
        Calculates the exact total objective cost of a binary chromosome.
        
        Steps:
            1. Audits physical capacity limits. If insufficient, returns a penalty.
            2. Solves the continuous customer routing sub-problem via SciPy linprog (HiGHS).
            3. Reconstructs the full n x m flow allocation matrix x from flattened LP variables.
            4. Encapsulates in a structured CFLPSolution object.
            5. Verifies constraints via the double-precision constraint checker.
            6. Computes the true fixed + variable shipping cost via the vectorized cost calculator.
            
        Args:
            individual (list): Binary facility vector (genotype).
            
        Returns:
            Tuple[float,]: A 1-tuple containing the total objective cost (fitness).
        """
        y_val = np.array(individual, dtype=np.int32)
        
        # --- 1. Analytical Physical Capacity Check ---
        open_capacity = np.sum(self.dataset.capacities * y_val)
        if open_capacity < self.total_demand:
            return (1e12,)  # Return massive penalty for physically impossible configurations
            
        # Get list of active facility indices
        open_indices = np.where(y_val == 1)[0]
        num_open = len(open_indices)
        
        # If no facilities are open (and yet total_demand is 0, which is unlikely but safe to guard),
        # or if we somehow bypassed the capacity check, return a penalty.
        if num_open == 0:
            return (1e12,)
            
        # --- 2. Solve Continuous Transportation Sub-problem ---
        # --- Quick UFLP Feasibility Check (Mathematical Shortcut) ---
        # For each customer, find the cheapest open facility
        cheapest_idx = np.argmin(self.dataset.transport_costs[:, open_indices], axis=1)
        
        # Calculate the capacity loaded on each open facility
        loaded_demands = np.zeros(num_open)
        np.add.at(loaded_demands, cheapest_idx, self.dataset.demands)
        
        # If this assignment satisfies all capacity constraints, UFLP assignment is optimal for CFLP!
        if np.all(loaded_demands <= self.dataset.capacities[open_indices]):
            # Reconstruct flow matrix x_val in absolute units
            x_val = np.zeros((self.n, self.m), dtype=np.float64)
            for j in range(self.n):
                x_val[j, open_indices[cheapest_idx[j]]] = self.dataset.demands[j]
            
            solution = CFLPSolution(y_val, x_val)
            total_cost = calculate_total_cost(solution, self.dataset)
            return (total_cost,)
            
        # Variables: w[j, k] representing fraction of demand from open facility open_indices[k] to customer j.
        # Length of decision variables = n * num_open.
        
        # A. Objective Coefficients (Flattened cost array)
        c = self.dataset.transport_costs[:, open_indices].flatten()
        
        # B. Equality Constraints: Customer demands satisfied (sum_{k} w_{j,k} == 1.0)
        rows_eq = np.repeat(np.arange(self.n), num_open)
        cols_eq = np.arange(self.n * num_open)
        data_eq = np.ones(self.n * num_open)
        A_eq = sp.coo_matrix((data_eq, (rows_eq, cols_eq)), shape=(self.n, self.n * num_open)).tocsr()
        b_eq = np.ones(self.n)
        
        # C. Inequality Constraints: Active warehouse capacities (sum_{j} d_j w_{j,k} <= capacities[k])
        rows_ub = np.repeat(np.arange(num_open), self.n)
        cols_ub = np.arange(self.n * num_open).reshape(self.n, num_open).T.flatten()
        data_ub = np.tile(self.dataset.demands, num_open)
        A_ub = sp.coo_matrix((data_ub, (rows_ub, cols_ub)), shape=(num_open, self.n * num_open)).tocsr()
        b_ub = self.dataset.capacities[open_indices]
        
        # D. Non-negativity bounds (w_jk >= 0)
        bounds = [(0.0, None)] * len(c)
        
        # E. Solve continuous LP in-memory
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res.success:
            return (1e12,)  # LP failed to converge (infeasible allocation space)
            
        # --- 3. Reconstruct Dense Flow Matrix (x) in absolute units ---
        x_val = np.zeros((self.n, self.m), dtype=np.float64)
        for j in range(self.n):
            for k in range(num_open):
                idx_in_res = j * num_open + k
                fac_idx = open_indices[k]
                # Reconstruct in absolute flow units
                x_val[j, fac_idx] = res.x[idx_in_res] * self.dataset.demands[j]
                
        # --- 4. Build Structured Solution Object ---
        solution = CFLPSolution(y_val, x_val)
        
        # --- 5. Verify Constraints Dynamically ---
        feasible, errors = is_feasible(solution, self.dataset)
        if not feasible:
            # If the solver produces mathematical floating-point rounding violations that exceed tolerance
            return (1e12,)
            
        # --- 6. Calculate True Vectorized Objective Cost ---
        total_cost = calculate_total_cost(solution, self.dataset)
        
        return (total_cost,)
