import numpy as np
from parser import CFLPDataset
from solution_representation import CFLPSolution
from cost_calculator import calculate_total_cost
from constraint_checker import is_feasible

class GreedyBaselineSolver:
    """
    A baseline heuristic solver for the Capacitated Facility Location Problem (CFLP).
    
    Algorithm:
        1. Selects facility opening configurations (y) greedily: sorts warehouses by 
           opening cost efficiency (fixed_cost / capacity) and opens them sequentially 
           until their cumulative capacity covers at least the total system customer demand.
        2. Assigns customer demands (x) using the Nearest Feasible Facility Heuristic: 
           for each customer, sorts open warehouses by shipping cost and routes demand 
           to the cheapest warehouse with available remaining capacity.
        3. If localized capacity bottlenecks occur (leaving some demand unsatisfied), 
           the solver falls back to opening all warehouses (y_i = 1 for all i) to guarantee 
           a physically feasible baseline assignment.
    """
    
    def __init__(self, dataset: CFLPDataset):
        self.dataset = dataset
        
    def solve(self, force_open_all: bool = False) -> CFLPSolution:
        """
        Solves the CFLP instance using the Nearest Feasible Facility Heuristic.
        
        Args:
            force_open_all (bool): If True, skips greedy facility selection and 
                                  forces all warehouses open (y=1) immediately.
                                  
        Returns:
            CFLPSolution: Structured solution containing opening vector y and flow matrix x.
        """
        m = self.dataset.num_facilities
        n = self.dataset.num_customers
        
        total_demand = np.sum(self.dataset.demands)
        
        # --- Step 1: Facility Status Selection (y) ---
        y = np.zeros(m, dtype=np.int32)
        
        if force_open_all:
            y[:] = 1
        else:
            # Sort facilities by cost-to-capacity efficiency ratios
            # Avoid division by zero by adding epsilon
            efficiency = self.dataset.fixed_costs / (self.dataset.capacities + 1e-9)
            sorted_indices = np.argsort(efficiency)
            
            cumulative_capacity = 0.0
            for idx in sorted_indices:
                y[idx] = 1
                cumulative_capacity += self.dataset.capacities[idx]
                if cumulative_capacity >= total_demand:
                    break
                    
        # --- Step 2: Flow Allocation (x) ---
        x = np.zeros((n, m), dtype=np.float64)
        
        # Array to track remaining capacity at each warehouse
        remaining_capacity = self.dataset.capacities * y
        
        # Get list of open facility indices
        open_indices = np.where(y == 1)[0].tolist()
        
        # Loop through each customer and route demand greedily to nearest open warehouse
        for j in range(n):
            demand_left = self.dataset.demands[j]
            
            # Sort open facilities by transportation cost to customer j
            sorted_open_facs = sorted(open_indices, key=lambda i: self.dataset.transport_costs[j, i])
            
            for i in sorted_open_facs:
                if remaining_capacity[i] > 0.0:
                    # Allocate whichever is smaller: the customer's remaining demand or the facility's capacity
                    flow = min(demand_left, remaining_capacity[i])
                    x[j, i] = flow
                    remaining_capacity[i] -= flow
                    demand_left -= flow
                    
                    if demand_left <= 1e-7:
                        break
                        
            # If a customer's demand could not be fully met due to localized capacity bottlenecks
            if demand_left > 1e-7 and not force_open_all:
                # Fallback: Retry solver by forcing all facilities open to guarantee feasibility
                return self.solve(force_open_all=True)
                
        return CFLPSolution(y, x)


if __name__ == "__main__":
    # Test block to verify baseline solver on cap41.txt
    import os
    
    test_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cap41.txt")
    print(f"Loading test file: {test_path}")
    
    try:
        dataset = CFLPDataset(test_path)
        solver = GreedyBaselineSolver(dataset)
        
        print("\nSolving using Nearest Feasible Facility Heuristic...")
        sol = solver.solve()
        
        print("\nSolution successfully generated!")
        print(sol)
        
        # Verify feasibility
        feasible, errors = is_feasible(sol, dataset)
        print(f"Feasibility Check: {'FEASIBLE' if feasible else 'INFEASIBLE'}")
        if not feasible:
            print("Errors detected:")
            for err in errors:
                print(f"  * {err}")
            assert False
        else:
            total_cost = calculate_total_cost(sol, dataset)
            print(f"Total Heuristic Cost (Z): ${total_cost:,.2f}")
            print("Baseline optimization verified successfully!")
    except Exception as e:
        print(f"Test failed: {e}")
