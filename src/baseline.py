import os
import time
import glob
import re
from typing import Dict, Any, Tuple
import numpy as np
import pulp
from parser import CFLPDataset

class GreedySolver:
    """
    A Greedy Heuristic Solver for the Capacitated Facility Location Problem (CFLP).
    
    Strategy:
    1. Rank facilities based on their cost-efficiency ratio: fixed_cost / capacity (lower is better).
    2. Open facilities one-by-one in ranked order until their total capacity exceeds the total demand.
    3. Allocate customer demands to open facilities greedily based on cheapest transportation cost,
       respecting capacity constraints.
    """
    def __init__(self, dataset: CFLPDataset):
        self.dataset = dataset
        self.num_facilities = dataset.num_facilities
        self.num_customers = dataset.num_customers
        
    def solve(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Executes the greedy heuristic.
        
        Returns:
            Tuple[float, np.ndarray, np.ndarray]:
                - total_cost (float): Calculated minimum cost.
                - y_val (np.ndarray): Binary array of shape (num_facilities,) indicating open status (1 or 0).
                - x_val (np.ndarray): Flow matrix of shape (num_customers, num_facilities).
        """
        start_time = time.time()
        total_demand = np.sum(self.dataset.demands)
        
        # 1. Rank facilities by cost-to-capacity efficiency ratio
        # Avoid division by zero by adding a tiny epsilon for free facilities
        ratios = []
        for i in range(self.num_facilities):
            cap = self.dataset.capacities[i]
            cost = self.dataset.fixed_costs[i]
            ratio = cost / cap if cap > 0 else float('inf')
            ratios.append((ratio, i))
            
        ratios.sort()  # Sort ascending (cheapest cost per unit capacity first)
        
        # 2. Open facilities in ranked order until we have enough capacity to cover all demands
        y_val = np.zeros(self.num_facilities, dtype=int)
        accumulated_capacity = 0.0
        
        for _, i in ratios:
            y_val[i] = 1
            accumulated_capacity += self.dataset.capacities[i]
            if accumulated_capacity >= total_demand:
                break
                
        # 3. Greedy Customer Assignment
        # Allocate customer demands to open facilities
        # Flow matrix of shape (J, I)
        x_val = np.zeros((self.num_customers, self.num_facilities))
        remaining_capacity = self.dataset.capacities.copy() * y_val
        
        for j in range(self.num_customers):
            demand_left = self.dataset.demands[j]
            
            # Sort open facilities by transportation cost to this customer
            open_fac_indices = np.where(y_val == 1)[0]
            costs_to_open = [(self.dataset.transport_costs[j, i], i) for i in open_fac_indices]
            costs_to_open.sort()
            
            for cost, i in costs_to_open:
                if demand_left <= 0:
                    break
                # How much can we supply from facility i?
                allocated_flow = min(demand_left, remaining_capacity[i])
                if allocated_flow > 0:
                    x_val[j, i] = allocated_flow
                    remaining_capacity[i] -= allocated_flow
                    demand_left -= allocated_flow
                    
            # If after checking all open facilities, there's still demand left, the current open set
            # is physically unable to satisfy demand (e.g. due to capacity locking). 
            # We must open more facilities to prevent infeasibility.
            if demand_left > 0:
                # Open the next best closed facility
                closed_indices = np.where(y_val == 0)[0]
                if len(closed_indices) > 0:
                    # Find the cheapest transport cost among closed facilities
                    closed_costs = [(self.dataset.transport_costs[j, i], i) for i in closed_indices]
                    closed_costs.sort()
                    next_fac = closed_costs[0][1]
                    y_val[next_fac] = 1
                    remaining_capacity[next_fac] = self.dataset.capacities[next_fac]
                    
                    # Allocate remaining flow
                    allocated_flow = min(demand_left, remaining_capacity[next_fac])
                    x_val[j, next_fac] = allocated_flow
                    remaining_capacity[next_fac] -= allocated_flow
                    demand_left -= allocated_flow
                    
        # 4. Compute Total Cost
        fixed_cost_sum = np.sum(self.dataset.fixed_costs * y_val)
        # Safe division to compute fractional assignments for transport cost
        demands_scaled = np.where(self.dataset.demands > 0, self.dataset.demands, 1.0)
        x_frac = x_val / demands_scaled[:, np.newaxis]
        transport_cost_sum = np.sum(self.dataset.transport_costs * x_frac)
        total_cost = fixed_cost_sum + transport_cost_sum
        
        return total_cost, y_val, x_val


class MILPSolver:
    """
    An Exact Mathematical Solver for the Capacitated Facility Location Problem (CFLP) 
    using Mixed-Integer Linear Programming (MILP) modeled via PuLP.
    """
    def __init__(self, dataset: CFLPDataset):
        self.dataset = dataset
        self.num_facilities = dataset.num_facilities
        self.num_customers = dataset.num_customers
        
    def solve(self, timeout_sec: int = 60) -> Tuple[float, np.ndarray, np.ndarray, str]:
        """
        Solves the CFLP model to mathematical optimality using Coin-OR CBC solver.
        
        Args:
            timeout_sec (int): Maximum time in seconds allowed for the solver.
            
        Returns:
            Tuple[float, np.ndarray, np.ndarray, str]:
                - total_cost (float): Optimal cost.
                - y_val (np.ndarray): Binary array of shape (num_facilities,) indicating open status (1 or 0).
                - x_val (np.ndarray): Flow matrix of shape (num_customers, num_facilities).
                - status (str): Solver exit status ('Optimal', 'Infeasible', etc.).
        """
        # Create the minimization problem
        prob = pulp.LpProblem(f"CFLP_Exact_{self.dataset.name}", pulp.LpMinimize)
        
        # 1. Decision Variables
        # y[i] is 1 if facility i is open, 0 otherwise (Binary)
        y = pulp.LpVariable.dicts("y", range(self.num_facilities), cat=pulp.LpBinary)
        
        # x[j, i] is the flow of goods from facility i to customer j (Continuous >= 0)
        x = pulp.LpVariable.dicts("x", 
                                  ((j, i) for j in range(self.num_customers) for i in range(self.num_facilities)), 
                                  lowBound=0, 
                                  cat=pulp.LpContinuous)
        
        # 2. Objective Function
        # Minimize: fixed costs + transportation costs
        # CORRECTED: transport_costs[j,i] is unit cost; multiply directly by flow x[j,i]
        # (Previously divided by demand, which was mathematically incorrect)
        prob += (
            pulp.lpSum(self.dataset.fixed_costs[i] * y[i] for i in range(self.num_facilities)) +
            pulp.lpSum(self.dataset.transport_costs[j, i] * x[j, i]
                        for j in range(self.num_customers) for i in range(self.num_facilities))
        )
        
        # 3. Constraints
        # Constraint A: Satisfy all customer demand
        for j in range(self.num_customers):
            prob += pulp.lpSum(x[j, i] for i in range(self.num_facilities)) == self.dataset.demands[j]
            
        # Constraint B: Respect facility capacities
        for i in range(self.num_facilities):
            prob += pulp.lpSum(x[j, i] for j in range(self.num_customers)) <= self.dataset.capacities[i] * y[i]
            
        # 4. Solve the model
        print(f"[MILP Solver] Solving CFLP instance '{self.dataset.name}' ({self.num_facilities} facilities, {self.num_customers} customers) using CBC solver (timeout={timeout_sec}s)...")
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=timeout_sec)
        prob.solve(solver)

        # 5. Extract results
        # NOTE: pulp.LpStatus[prob.status] reports "Optimal" whenever CBC returns ANY
        # integer-feasible solution, even if CBC actually stopped on the time limit
        # without proving optimality (a known PuLP/CBC status-parsing gap). prob.sol_status
        # distinguishes this correctly: 1 = proven optimal, 2 = feasible but NOT proven
        # optimal (i.e. the time limit was hit before the branch-and-bound gap closed).
        status_str = pulp.LpStatus[prob.status]
        if status_str == "Optimal" and prob.sol_status != pulp.LpSolutionOptimal:
            status_str = "Time Limit (Feasible, Not Proven Optimal)"

        if status_str != "Optimal":
            print(f"[Warning] Solver finished with non-optimal status: {status_str}")
            
        # Retrieve objective value
        total_cost = pulp.value(prob.objective)
        
        # Map variables back to NumPy arrays
        y_val = np.zeros(self.num_facilities, dtype=int)
        for i in range(self.num_facilities):
            y_val[i] = int(round(pulp.value(y[i])))
            
        x_val = np.zeros((self.num_customers, self.num_facilities))
        for j in range(self.num_customers):
            for i in range(self.num_facilities):
                x_val[j, i] = pulp.value(x[j, i])
                
        return total_cost, y_val, x_val, status_str


def run_benchmarks():
    """
    Main function to run both baseline solvers across all OR-Library files
    and print comparative benchmarking metrics.
    """
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, "..", "data", "raw")
    
    # Dynamically discover all cap*.txt files in raw_dir
    file_pattern = os.path.join(raw_dir, "cap*.txt")
    raw_files = glob.glob(file_pattern)
    
    # Helper to sort files numerically (e.g. cap41 before cap100)
    def numerical_sort_key(file_path):
        filename = os.path.basename(file_path)
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0
        
    raw_files.sort(key=numerical_sort_key)
    files = [os.path.basename(f) for f in raw_files]
    
    print("=" * 90)
    print(f"{'CFLP BASELINE SOLVERS BENCHMARK RUN':^90}")
    print("=" * 90)
    
    print(f"{'Dataset':<8} | {'Solver':<8} | {'Total Cost':<14} | {'Active Facs':<11} | {'Time (ms)':<9} | {'Opt. Gap (%)':<12}")
    print("-" * 90)
    
    for filename in files:
        file_path = os.path.join(raw_dir, filename)
        if not os.path.exists(file_path):
            continue
            
        dataset = CFLPDataset(file_path)
        
        # 1. Solve using Greedy Heuristic
        t0 = time.time()
        greedy = GreedySolver(dataset)
        g_cost, g_y, g_x = greedy.solve()
        g_time = (time.time() - t0) * 1000.0  # in ms
        g_active = int(np.sum(g_y))
        
        # 2. Solve using MILP
        t0 = time.time()
        milp = MILPSolver(dataset)
        m_cost, m_y, m_x, m_status = milp.solve()
        m_time = (time.time() - t0) * 1000.0  # in ms
        m_active = int(np.sum(m_y))
        
        # Calculate Optimality Gap
        opt_gap = ((g_cost - m_cost) / m_cost) * 100.0
        
        # Print results row-by-row
        print(f"{dataset.name:<8} | {'Greedy':<8} | ${g_cost:<13,.2f} | {g_active:>2}/{dataset.num_facilities:<8} | {g_time:>9.2f} | {opt_gap:>10.2f}%")
        print(f"{'':<8} | {'MILP':<8} | ${m_cost:<13,.2f} | {m_active:>2}/{dataset.num_facilities:<8} | {m_time:>9.2f} | {'0.00%':>12}")
        print("-" * 90)

if __name__ == "__main__":
    run_benchmarks()
