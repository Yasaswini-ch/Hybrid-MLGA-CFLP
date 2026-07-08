import numpy as np
from typing import Dict, Any

class CFLPSolution:
    """
    A structured class to represent a candidate solution for the Capacitated 
    Facility Location Problem (CFLP).
    
    Attributes:
        y (np.ndarray): A binary integer array of shape (m,) where y[i] is 1 if 
                        facility i is open, and 0 if closed.
        x (np.ndarray): A continuous flow matrix of shape (n, m) where x[j, i] represents 
                        the flow of demand from facility i to customer j.
    """
    
    def __init__(self, y: np.ndarray, x: np.ndarray):
        # Convert inputs to NumPy arrays if they aren't already
        self.y = np.array(y, dtype=np.int32)
        self.x = np.array(x, dtype=np.float64)
        
    def validate_shapes(self, num_facilities: int, num_customers: int) -> bool:
        """
        Validates that the solution vectors and matrices conform to expected problem dimensions.
        
        Args:
            num_facilities (int): Expected number of potential warehouses (m).
            num_customers (int): Expected number of customers (n).
            
        Returns:
            bool: True if shapes conform, raises ValueError otherwise.
        """
        if self.y.ndim != 1 or self.y.shape[0] != num_facilities:
            raise ValueError(
                f"Shape mismatch for facility opening vector y: "
                f"Expected 1D array of shape ({num_facilities},), got {self.y.shape}."
            )
            
        if self.x.ndim != 2 or self.x.shape != (num_customers, num_facilities):
            raise ValueError(
                f"Shape mismatch for customer flow matrix x: "
                f"Expected 2D array of shape ({num_customers}, {num_facilities}), got {self.x.shape}."
            )
            
        # Verify binary constraints on y
        if not np.all((self.y == 0) | (self.y == 1)):
            raise ValueError("Integrity error: Facility vector y must contain strictly binary values (0 or 1).")
            
        return True
        
    def convert_flow_to_allocations(self) -> np.ndarray:
        """
        Converts the continuous customer assignment flow matrix `x` of shape (n, m) 
        into a discrete allocation index array of shape (n,).
        
        For each customer j, the allocated facility is selected as the facility i 
        supplying the largest share of their demand (the argmax flow).
        
        This conversion is highly useful for mapping continuous solution assignments 
        to discrete GA chromosomes and analyzing facility serving footprints.
        
        Returns:
            np.ndarray: An integer array of shape (num_customers,) where entry j 
                        is the index of the primary facility serving customer j.
        """
        if self.x.shape[0] == 0 or self.x.shape[1] == 0:
            return np.array([], dtype=np.int32)
            
        # Select the facility supplying maximum flow for each customer (row argmax)
        allocations = np.argmax(self.x, axis=1)
        return allocations

    def get_info(self) -> Dict[str, Any]:
        """
        Gathers basic descriptive statistics about the solution layout.
        
        Returns:
            Dict[str, Any]: A dictionary of descriptive parameters.
        """
        num_open = int(np.sum(self.y))
        total_flow = float(np.sum(self.x))
        active_indices = np.where(self.y == 1)[0].tolist()
        
        return {
            "num_facilities": self.y.shape[0],
            "num_customers": self.x.shape[0],
            "num_open_facilities": num_open,
            "total_flow_allocated": total_flow,
            "open_facilities_indices": active_indices
        }

    def __str__(self) -> str:
        info = self.get_info()
        return (
            f"--- Structured CFLP Solution ---\n"
            f"Dimension (m x n)     : {info['num_facilities']} x {info['num_customers']}\n"
            f"Active Warehouses     : {info['num_open_facilities']} / {info['num_facilities']} open\n"
            f"Open Warehouses Set   : {info['open_facilities_indices']}\n"
            f"Total Flow Allocated  : {info['total_flow_allocated']:.2f} units"
        )


if __name__ == "__main__":
    # Test block to verify CFLPSolution functionalities
    print("Testing CFLPSolution representation...")
    
    # Mock data for m=4 facilities, n=3 customers
    mock_y = np.array([1, 0, 1, 0])
    
    # Customer flows matrix of shape (3, 4)
    # Customer 0 served by facility 0 (flow 15.0)
    # Customer 1 served by facility 2 (flow 20.0)
    # Customer 2 served by facility 0 (flow 30.0)
    mock_x = np.array([
        [15.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0, 20.0,  0.0],
        [30.0,  0.0,  0.0,  0.0]
    ])
    
    try:
        solution = CFLPSolution(mock_y, mock_x)
        solution.validate_shapes(num_facilities=4, num_customers=3)
        print("Success! Solution shapes validated correctly.")
        print(solution)
        
        allocations = solution.convert_flow_to_allocations()
        print(f"Discrete primary facility allocation index array: {allocations}")
        # Expected primary allocations: [0, 2, 0]
        assert np.array_equal(allocations, [0, 2, 0])
        print("Assert passed: Discrete allocation mappings are correct!")
    except Exception as e:
        print(f"Test failed: {e}")
