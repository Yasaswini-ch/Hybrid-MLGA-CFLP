import numpy as np
from solution_representation import CFLPSolution
from parser import CFLPDataset
from typing import Tuple, List

def check_demand_satisfaction(x: np.ndarray, demands: np.ndarray, tolerance: float = 1e-7) -> Tuple[bool, List[str]]:
    """
    Verifies that customer demands are fully met.
    
    Formula:
        Sum_{i in I} x_{ij} == d_j  (within double-precision tolerance) for all j in J.
        
    Args:
        x (np.ndarray): Flow allocation matrix of shape (n, m).
        demands (np.ndarray): Demand of each store, shape (n,).
        tolerance (float): Floating-point comparison boundary threshold (default 1e-7).
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, errors_list)
    """
    is_valid = True
    errors = []
    
    # Calculate sum of flows served to each customer j (sum along columns)
    allocated_demands = np.sum(x, axis=1)
    
    for j in range(demands.shape[0]):
        diff = abs(allocated_demands[j] - demands[j])
        if diff > tolerance:
            is_valid = False
            errors.append(
                f"Customer {j} demand violation: "
                f"Expected {demands[j]:.2f} units, served {allocated_demands[j]:.2f} units (Diff: {diff:.2e} > tolerance)."
            )
            
    return is_valid, errors


def check_facility_capacity(x: np.ndarray, y: np.ndarray, capacities: np.ndarray, tolerance: float = 1e-7) -> Tuple[bool, List[str]]:
    """
    Verifies that product flows from each warehouse do not exceed its physical capacity limit.
    
    Formula:
        Sum_{j in J} x_{ij} <= s_i * y_i (within tolerance) for all i in I.
        
    Args:
        x (np.ndarray): Flow allocation matrix of shape (n, m).
        y (np.ndarray): Binary facility status array of shape (m,).
        capacities (np.ndarray): Storage capacity limit of each warehouse, shape (m,).
        tolerance (float): Floating-point comparison boundary threshold (default 1e-7).
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, errors_list)
    """
    is_valid = True
    errors = []
    
    # Calculate total flow leaving each facility i (sum along rows)
    facility_flows = np.sum(x, axis=0)
    
    for i in range(y.shape[0]):
        limit = capacities[i] * y[i]
        # Allow a small float epsilon threshold overflow to prevent rounding errors
        if facility_flows[i] > limit + tolerance:
            is_valid = False
            errors.append(
                f"Facility {i} capacity violation: "
                f"Warehouse capacity is {limit:.2f} (y={y[i]}), "
                f"shipped flow is {facility_flows[i]:.2f} (Overflow: {facility_flows[i] - limit:.2e} > tolerance)."
            )
            
    return is_valid, errors


def check_flow_origin(x: np.ndarray, y: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Verifies that no products originate from closed facilities.
    
    Formula:
        y_i == 0 ==> x_{ij} == 0 for all j in J.
        
    Args:
        x (np.ndarray): Flow allocation matrix of shape (n, m).
        y (np.ndarray): Binary facility status array of shape (m,).
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, errors_list)
    """
    is_valid = True
    errors = []
    
    for i in range(y.shape[0]):
        if y[i] == 0:
            # If closed, the sum of flows must be strictly 0.0 (non-negativity assumed)
            total_closed_flow = np.sum(x[:, i])
            if total_closed_flow > 0.0:
                is_valid = False
                errors.append(
                    f"Closed facility shipment violation: "
                    f"Facility {i} is closed (y=0) but shipped {total_closed_flow:.2f} units of flow."
                )
                
    return is_valid, errors


def is_feasible(solution: CFLPSolution, dataset: CFLPDataset, tolerance: float = 1e-7) -> Tuple[bool, List[str]]:
    """
    Executes a unified feasibility assessment verifying demand satisfaction, capacity limits, 
    and flow origin constraints.
    
    Args:
        solution (CFLPSolution): Structured solution object holding y and x.
        dataset (CFLPDataset): Parsed OR-Library dataset holding capacities, demands, and costs.
        tolerance (float): Floating-point comparison tolerance (default 1e-7).
        
    Returns:
        Tuple[bool, List[str]]: (is_feasible_boolean, list_of_error_logs)
    """
    # 1. Dimensions validation
    solution.validate_shapes(dataset.num_facilities, dataset.num_customers)
    
    all_errors = []
    
    # 2. Demand Satisfaction
    dem_ok, dem_errs = check_demand_satisfaction(solution.x, dataset.demands, tolerance)
    all_errors.extend(dem_errs)
    
    # 3. Facility Capacity
    cap_ok, cap_errs = check_facility_capacity(solution.x, solution.y, dataset.capacities, tolerance)
    all_errors.extend(cap_errs)
    
    # 4. Closed Facility Flow Origin
    origin_ok, origin_errs = check_flow_origin(solution.x, solution.y)
    all_errors.extend(origin_errs)
    
    # Solution is feasible only if there are no violations
    feasible = (len(all_errors) == 0)
    return feasible, all_errors


if __name__ == "__main__":
    # Test block to verify constraint checker bounds
    print("Testing CFLP Constraint Checker...")
    
    # Mock dataset
    class MockDataset:
        def __init__(self):
            self.num_facilities = 3
            self.num_customers = 2
            self.capacities = np.array([100.0, 150.0, 75.0])
            self.demands = np.array([80.0, 90.0])
            
    mock_dataset = MockDataset()
    
    print("\n--- Testing Feasible Solution ---")
    # Open facility 0 and 1
    mock_y_ok = np.array([1, 1, 0])
    # Demands satisfy exactly: Customer 0 served by 0 (80), Customer 1 served by 1 (90)
    # Flows leaving: Facility 0 (80 <= 100), Facility 1 (90 <= 150), Facility 2 (0)
    mock_x_ok = np.array([
        [80.0,  0.0, 0.0],
        [ 0.0, 90.0, 0.0]
    ])
    
    sol_ok = CFLPSolution(mock_y_ok, mock_x_ok)
    ok, errs = is_feasible(sol_ok, mock_dataset)
    print(f"Feasible Status: {ok} (Errors found: {len(errs)})")
    assert ok == True
    assert len(errs) == 0
    
    print("\n--- Testing Infeasible Solution (Demand & Closed Warehouses violations) ---")
    # Open only facility 0, facility 1 is CLOSED
    mock_y_bad = np.array([1, 0, 0])
    # Customer 0: demand served is 75 (under-satisfied: expects 80)
    # Customer 1: served by facility 1 (90), which is CLOSED
    mock_x_bad = np.array([
        [75.0,  0.0, 0.0],
        [ 0.0, 90.0, 0.0]
    ])
    
    sol_bad = CFLPSolution(mock_y_bad, mock_x_bad)
    ok, errs = is_feasible(sol_bad, mock_dataset)
    print(f"Feasible Status: {ok}")
    print("Violations logged:")
    for err in errs:
        print(f"  * {err}")
    assert ok == False
    # 3 legitimate violations: Customer 0 demand shortfall, Facility 1 capacity
    # violation (closed facility has 0 capacity but received flow), and the
    # separate closed-facility-shipment check catching the same root cause.
    assert len(errs) == 3
    
    print("\n--- Testing Infeasible Solution (Capacity overflow violation) ---")
    mock_y_overflow = np.array([0, 1, 0])
    # Shipped flow from Facility 1 is 80 + 90 = 170 (Exceeds capacity limit: 150)
    mock_x_overflow = np.array([
        [0.0, 80.0, 0.0],
        [0.0, 90.0, 0.0]
    ])
    sol_overflow = CFLPSolution(mock_y_overflow, mock_x_overflow)
    ok, errs = is_feasible(sol_overflow, mock_dataset)
    print(f"Feasible Status: {ok}")
    print("Violations logged:")
    for err in errs:
        print(f"  * {err}")
    assert ok == False
    assert len(errs) == 1
    
    print("\nAssert passed: All constraint checking bounds function correctly!")
# 
