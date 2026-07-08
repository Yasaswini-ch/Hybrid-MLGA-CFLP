import os
import numpy as np
from parser import CFLPDataset
from solution_representation import CFLPSolution
from cost_calculator import calculate_total_cost, calculate_fixed_costs, calculate_transportation_costs
from constraint_checker import is_feasible
from baseline_solver import GreedyBaselineSolver

def main():
    print("=" * 60)
    print("        CFLP PHASE 2 INTEGRATION VERIFICATION RUN")
    print("=" * 60)
    
    # Path to cap41.txt
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    file_path = os.path.join(raw_dir, "cap41.txt")
    
    if not os.path.exists(file_path):
        print(f"[Error] Benchmark dataset not found at: {file_path}")
        return
        
    print(f"1. Loading Beasley cap41 benchmark dataset...")
    dataset = CFLPDataset(file_path)
    print("   Dataset loaded successfully:")
    print(f"     - Warehouses (m): {dataset.num_facilities}")
    print(f"     - Customers  (n): {dataset.num_customers}")
    print(f"     - Total Demand  : {np.sum(dataset.demands):,.0f} units")
    print(f"     - Total Capacity: {np.sum(dataset.capacities):,.0f} units")
    print("-" * 60)
    
    print("2. Running Greedy Baseline Solver (Nearest Feasible Facility)...")
    solver = GreedyBaselineSolver(dataset)
    solution = solver.solve()
    print("   Solution generated successfully:")
    print(f"     - Active Warehouses Count: {np.sum(solution.y)} / {dataset.num_facilities}")
    print(f"     - Open Facilities Set    : {np.where(solution.y == 1)[0].tolist()}")
    print("-" * 60)
    
    print("3. Validating Structured Solution Mappings...")
    try:
        solution.validate_shapes(dataset.num_facilities, dataset.num_customers)
        print("   [PASS] Solution dimensions and binary integrity verified.")
        
        allocations = solution.convert_flow_to_allocations()
        print(f"   [PASS] Primary facility allocations index mapped (shape: {allocations.shape}).")
        print(f"          First 10 stores allocated to: {allocations[:10]}")
    except Exception as e:
        print(f"   [FAIL] Shape validation failed: {e}")
        return
    print("-" * 60)
    
    print("4. Executing Unified Feasibility Constraint Checks...")
    feasible, errors = is_feasible(solution, dataset)
    if feasible:
        print("   [PASS] Heuristic solution is 100% physically FEASIBLE.")
        print("          No demand, capacity, or closed-flow violations detected.")
    else:
        print("   [FAIL] Heuristic solution is INFEASIBLE.")
        print("          Violations logged:")
        for err in errors:
            print(f"            * {err}")
        return
    print("-" * 60)
    
    print("5. Evaluating Total Heuristic Cost (Objective Z)...")
    fixed_cost = calculate_fixed_costs(solution.y, dataset.fixed_costs)
    transport_cost = calculate_transportation_costs(solution.x, dataset.transport_costs)
    total_cost = calculate_total_cost(solution, dataset)
    
    print(f"   - Fixed Facility Opening Overhead: ${fixed_cost:,.2f}")
    print(f"   - Variable Transportation Shipping: ${transport_cost:,.2f}")
    print(f"   - Unified Objective Function Cost: ${total_cost:,.2f}")
    
    # Assert sanity check
    assert abs(total_cost - (fixed_cost + transport_cost)) < 1e-7
    print("   [PASS] Cost calculations are mathematically consistent.")
    print("-" * 60)
    
    print("6. Performing Robustness Infeasibility Perturbation Tests...")
    # Test 1: Under-satisfy customer 0 demand
    print("   * Perturbation A: Under-satisfying store 0 demand...")
    x_perturbed_a = solution.x.copy()
    x_perturbed_a[0, :] = 0.0  # Zero out store 0 flow allocation
    sol_perturbed_a = CFLPSolution(solution.y, x_perturbed_a)
    feasible_a, errors_a = is_feasible(sol_perturbed_a, dataset)
    print(f"     - Feasible: {feasible_a}")
    if not feasible_a:
        print(f"     - [PASS] Correctly caught demand satisfaction error: {errors_a[0]}")
    else:
        print("     - [FAIL] Perturbation was not caught!")
        return
        
    # Test 2: Shipping flow from closed facility
    print("   * Perturbation B: Shipping flow from closed facility...")
    y_perturbed_b = solution.y.copy()
    # Find a closed facility
    closed_indices = np.where(y_perturbed_b == 0)[0]
    if len(closed_indices) > 0:
        closed_idx = closed_indices[0]
        y_perturbed_b[closed_idx] = 0  # ensure y is closed
        # Route customer 0 demand to it
        x_perturbed_b = solution.x.copy()
        x_perturbed_b[0, :] = 0.0
        x_perturbed_b[0, closed_idx] = dataset.demands[0]  # Ship demand to closed warehouse
        sol_perturbed_b = CFLPSolution(y_perturbed_b, x_perturbed_b)
        feasible_b, errors_b = is_feasible(sol_perturbed_b, dataset)
        print(f"     - Feasible: {feasible_b}")
        if not feasible_b:
            print("     - [PASS] Correctly caught closed facility shipment error(s):")
            for err in errors_b:
                print(f"              * {err}")
        else:
            print("     - [FAIL] Closed facility shipment perturbation was not caught!")
            return
    else:
        print("     - No closed facilities available to perturb.")
        
    print("=" * 60)
    print("   ALL PHASE 2 INTEGRATION VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
