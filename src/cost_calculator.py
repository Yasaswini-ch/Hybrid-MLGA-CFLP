import numpy as np
from solution_representation import CFLPSolution
from parser import CFLPDataset

def calculate_fixed_costs(y: np.ndarray, fixed_costs: np.ndarray) -> float:
    """
    Computes the total fixed overhead cost incurred by opening facilities.
    
    Formula:
        Fixed Cost = Sum_{i in I} f_i * y_i
        
    Args:
        y (np.ndarray): Binary status array of shape (m,).
        fixed_costs (np.ndarray): Overhead cost of each warehouse, shape (m,).
        
    Returns:
        float: Total fixed facility opening cost.
    """
    # Ensure variables are mapped correctly as 1D arrays
    y_arr = np.array(y, dtype=np.int32)
    costs_arr = np.array(fixed_costs, dtype=np.float64)
    
    if y_arr.shape != costs_arr.shape:
        raise ValueError(
            f"Dimension mismatch in fixed costs calculation: "
            f"Facility vector shape is {y_arr.shape}, fixed costs vector shape is {costs_arr.shape}."
        )
        
    # Element-wise product and sum using fast NumPy vectorized functions
    return float(np.sum(y_arr * costs_arr))


def calculate_transportation_costs(x: np.ndarray, transport_costs: np.ndarray) -> float:
    """
    Computes the total variable shipping cost incurred by product flows.
    
    Formula:
        Transport Cost = Sum_{j in J} Sum_{i in I} c_{ij} * x_{ij}
        
    Args:
        x (np.ndarray): Continuous customer assignment flow matrix of shape (n, m).
        transport_costs (np.ndarray): Cost matrix of shape (n, m), where transport_costs[j, i]
                                      is the unit shipping cost from facility i to customer j.
                                      
    Returns:
        float: Total variable transportation shipping cost.
    """
    x_arr = np.array(x, dtype=np.float64)
    costs_arr = np.array(transport_costs, dtype=np.float64)
    
    if x_arr.shape != costs_arr.shape:
        raise ValueError(
            f"Dimension mismatch in transportation costs calculation: "
            f"Flow matrix shape is {x_arr.shape}, transport costs matrix shape is {costs_arr.shape}."
        )
        
    # Double-sum over element-wise products (Hadamard product and sum)
    return float(np.sum(x_arr * costs_arr))


def calculate_total_cost(solution: CFLPSolution, dataset: CFLPDataset) -> float:
    """
    Computes the unified objective function cost (Z) of a structured CFLP solution.
    
    Z = Fixed Cost + Transportation Cost
    
    Args:
        solution (CFLPSolution): Structured solution object holding y and x.
        dataset (CFLPDataset): Parsed OR-Library dataset holding capacities, demands, and costs.
        
    Returns:
        float: The total objective function cost (Z).
    """
    # Validate solution shapes before performing calculations
    solution.validate_shapes(dataset.num_facilities, dataset.num_customers)
    
    fixed = calculate_fixed_costs(solution.y, dataset.fixed_costs)
    # Safe division by demand to obtain fractional assignment
    demands_scaled = np.where(dataset.demands > 0, dataset.demands, 1.0)
    x_frac = solution.x / demands_scaled[:, np.newaxis]
    transport = calculate_transportation_costs(x_frac, dataset.transport_costs)
    
    return fixed + transport


if __name__ == "__main__":
    # Test block to verify cost calculator outputs
    print("Testing CFLP Cost Calculator...")
    
    # Mock dataset
    class MockDataset:
        def __init__(self):
            self.num_facilities = 3
            self.num_customers = 2
            self.fixed_costs = np.array([5000.0, 10000.0, 7500.0])
            self.demands = np.array([10.0, 15.0])
            self.transport_costs = np.array([
                [100.0, 200.0, 150.0],
                [300.0, 100.0, 250.0]
            ])
            
    mock_dataset = MockDataset()
    
    # Mock solution: Open facility 0 and 1, closed 2
    mock_y = np.array([1, 1, 0])
    
    # Customer flows: 
    # Customer 0: demand 10 units served by facility 0
    # Customer 1: demand 15 units served by facility 1
    mock_x = np.array([
        [10.0,  0.0, 0.0],
        [ 0.0, 15.0, 0.0]
    ])
    
    try:
        sol = CFLPSolution(mock_y, mock_x)
        
        # Expected fixed cost: 1 * 5000 + 1 * 10000 + 0 * 7500 = 15000.0
        fixed = calculate_fixed_costs(sol.y, mock_dataset.fixed_costs)
        print(f"Calculated Fixed Cost         : ${fixed:,.2f}")
        assert fixed == 15000.0
        
        # Expected transportation cost: 10 * 100 + 15 * 100 = 2500.0
        transport = calculate_transportation_costs(sol.x, mock_dataset.transport_costs)
        print(f"Calculated Transportation Cost : ${transport:,.2f}")
        assert transport == 2500.0
        
        # calculate_total_cost() divides flow by demand before applying transport_costs
        # (transport_costs[j,i] is the flat total cost to fully serve customer j from
        # facility i, not a per-unit rate -- see baseline.py's MILP objective comment
        # for the full evidence). Both customers here are served 100% by one facility
        # (flow == demand), so the fraction is 1.0 for each and transport reduces to
        # 100 + 100 = 200.0, not the raw absolute-flow product (2500.0) tested above.
        # Total cost: 15000 + 200 = 15200.0
        total = calculate_total_cost(sol, mock_dataset)
        print(f"Calculated Total Objective Cost: ${total:,.2f}")
        assert total == 15200.0
        
        print("Assert passed: All modular cost calculations are 100% correct!")
    except Exception as e:
        print(f"Test failed: {e}")
