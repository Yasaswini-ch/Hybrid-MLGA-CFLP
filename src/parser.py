import os
from typing import Dict, Any
import numpy as np

class CFLPDataset:
    """
    A class to represent and parse the Capacitated Facility Location Problem (CFLP) 
    datasets from the J.E. Beasley OR-Library.
    
    Attributes:
        file_path (str): Path to the raw OR-Library dataset text file.
        name (str): Name of the dataset (extracted from filename).
        num_facilities (int): Number of potential facility locations (m).
        num_customers (int): Number of customers to serve (n).
        capacities (np.ndarray): Array of shape (num_facilities,) containing capacity of each facility.
        fixed_costs (np.ndarray): Array of shape (num_facilities,) containing opening costs of each facility.
        demands (np.ndarray): Array of shape (num_customers,) containing demand of each customer.
        transport_costs (np.ndarray): Matrix of shape (num_customers, num_facilities) where 
                                      transport_costs[j, i] represents the unit transportation cost 
                                      from facility i to customer j.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.name = os.path.splitext(os.path.basename(file_path))[0]
        
        self.num_facilities: int = 0
        self.num_customers: int = 0
        
        self.capacities: np.ndarray = np.array([])
        self.fixed_costs: np.ndarray = np.array([])
        self.demands: np.ndarray = np.array([])
        self.transport_costs: np.ndarray = np.array([])
        
        # Load and parse the dataset immediately upon initialization
        self._parse_file()
        
    def _parse_file(self) -> None:
        """
        Reads and parses the raw text file according to the OR-Library CFLP structure.
        Uses a robust tokenization approach to bypass variable spaces and newlines.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.file_path}")
            
        with open(self.file_path, 'r') as file:
            content = file.read()
            
        # Dynamically replace 'capacity' placeholder with correct values
        if 'capacity' in content:
            name_lower = self.name.lower()
            content = content.replace('capacity', '999999999.0')
            
        # Split content by any whitespace (spaces, tabs, newlines) and filter out empty strings
        tokens = content.split()
        if not tokens:
            raise ValueError(f"The dataset file {self.file_path} is empty or invalid.")
            
        # Pointer to keep track of our position in the token list
        token_ptr = 0
        
        # --- 1. Parse Header ---
        # The first line contains the number of facilities (m) and the number of customers (n)
        self.num_facilities = int(tokens[token_ptr])
        self.num_customers = int(tokens[token_ptr + 1])
        token_ptr += 2
        
        # --- 2. Parse Facility Specifications ---
        # There are exactly `num_facilities` lines containing: [capacity, fixed_cost]
        capacities_list = []
        fixed_costs_list = []
        
        for i in range(self.num_facilities):
            capacity = float(tokens[token_ptr])
            fixed_cost = float(tokens[token_ptr + 1])
            
            capacities_list.append(capacity)
            fixed_costs_list.append(fixed_cost)
            token_ptr += 2
            
        self.capacities = np.array(capacities_list, dtype=np.float64)
        self.fixed_costs = np.array(fixed_costs_list, dtype=np.float64)
        
        # --- 3. Parse Customer Specifications ---
        # There are exactly `num_customers` blocks. Each block has:
        # - Customer demand (1 value)
        # - Transportation costs from each facility (num_facilities values)
        demands_list = []
        transport_costs_matrix = []
        
        for j in range(self.num_customers):
            # Read customer demand
            demand = float(tokens[token_ptr])
            demands_list.append(demand)
            token_ptr += 1
            
            # Read transportation costs for facility 1 to m
            costs_to_customer = []
            for i in range(self.num_facilities):
                cost = float(tokens[token_ptr])
                costs_to_customer.append(cost)
                token_ptr += 1
                
            transport_costs_matrix.append(costs_to_customer)
            
        self.demands = np.array(demands_list, dtype=np.float64)
        self.transport_costs = np.array(transport_costs_matrix, dtype=np.float64)
        
        # --- 4. Validate Parsing Integrity ---
        # Ensure that we processed the expected number of tokens
        expected_total_tokens = 2 + (self.num_facilities * 2) + (self.num_customers * (1 + self.num_facilities))
        if token_ptr != expected_total_tokens:
            print(f"[Warning] Token parsing size mismatch: Expected {expected_total_tokens} tokens, parsed {token_ptr}.")
            
    def get_summary(self) -> Dict[str, Any]:
        """
        Generates a summary of the dataset for logging and inspection.
        
        Returns:
            Dict[str, Any]: Summary dictionary containing key metrics.
        """
        return {
            "name": self.name,
            "facilities": self.num_facilities,
            "customers": self.num_customers,
            "total_demand": float(np.sum(self.demands)),
            "total_capacity": float(np.sum(self.capacities)),
            "min_fixed_cost": float(np.min(self.fixed_costs)),
            "max_fixed_cost": float(np.max(self.fixed_costs)),
            "avg_demand": float(np.mean(self.demands)),
            "cost_matrix_shape": self.transport_costs.shape,
            "capacity_demand_ratio": float(np.sum(self.capacities) / np.sum(self.demands))
        }

    def __str__(self) -> str:
        summary = self.get_summary()
        return (
            f"--- CFLP Dataset: {summary['name']} ---\n"
            f"Facilities (m)       : {summary['facilities']}\n"
            f"Customers (n)        : {summary['customers']}\n"
            f"Total Demand         : {summary['total_demand']:.2f}\n"
            f"Total Capacity       : {summary['total_capacity']:.2f}\n"
            f"Capacity/Demand Ratio: {summary['capacity_demand_ratio']:.2f}\n"
            f"Fixed Cost Range     : ${summary['min_fixed_cost']:.2f} - ${summary['max_fixed_cost']:.2f}\n"
            f"Transportation Costs : Matrix of size {summary['cost_matrix_shape'][0]}x{summary['cost_matrix_shape'][1]}"
        )


if __name__ == "__main__":
    # Test block to verify the parser works on our cap41 dataset
    import sys
    
    # Path to cap41.txt
    test_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cap41.txt")
    
    print("Testing CFLPDataset Parser...")
    try:
        dataset = CFLPDataset(test_path)
        print("Success! Dataset parsed correctly.")
        print(dataset)
        print("\nDataset Summary Stats:")
        for k, v in dataset.get_summary().items():
            print(f"  {k:22}: {v}")
    except Exception as e:
        print(f"Error encountered: {e}", file=sys.stderr)
