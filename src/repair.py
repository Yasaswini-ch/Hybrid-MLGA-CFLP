import numpy as np
from parser import CFLPDataset

class CFLPFeasibilityRepairer:
    """
    Implements a Lamarckian Feasibility Repair Operator for the CFLP.
    
    If an individual chromosome does not contain enough open facility capacity to satisfy 
    the total system customer demand, the repairer greedily opens closed facilities 
    sorted by their cost-to-capacity efficiency ratios (fixed_cost / capacity) 
    until physical feasibility is restored, writing the genetic changes directly back in-place.
    """
    
    def __init__(self, dataset: CFLPDataset):
        """
        Initializes the repairer.
        
        Args:
            dataset (CFLPDataset): Parsed CFLP instance.
        """
        self.dataset = dataset
        self.m = dataset.num_facilities
        self.total_demand = np.sum(dataset.demands)
        
        # Precompute cost-to-capacity efficiency ratios for all facilities
        # Smaller ratios indicate higher efficiency (more capacity per dollar spent)
        # Avoid division by zero by adding a tiny epsilon
        self.efficiency = self.dataset.fixed_costs / (self.dataset.capacities + 1e-9)
        
    def repair(self, individual: list) -> bool:
        """
        Audits and, if necessary, repairs an individual chromosome in-place.
        
        Args:
            individual (list): Binary facility status list representing a chromosome.
            
        Returns:
            bool: True if the individual was repaired; False if it was already feasible.
        """
        y = np.array(individual, dtype=np.int32)
        
        # 1. Audit active system capacity
        active_capacity = np.sum(self.dataset.capacities * y)
        
        # If capacity covers demand, the individual is physically feasible
        if active_capacity >= self.total_demand:
            return False
            
        # 2. Repair: Capacity is deficient
        # Identify all closed facility indices (where genes are 0)
        closed_indices = np.where(y == 0)[0]
        
        if len(closed_indices) == 0:
            # If all facilities are already open and yet capacity is insufficient,
            # this indicates a mathematically impossible problem instance (violates core assumptions)
            return False
            
        # Sort closed facilities by precomputed cost-to-capacity efficiency ratios
        sorted_closed = sorted(closed_indices.tolist(), key=lambda i: self.efficiency[i])
        
        # Greedily open facilities until total capacity satisfies system demand
        for idx in sorted_closed:
            individual[idx] = 1  # Open the facility in the DEAP list in-place
            active_capacity += self.dataset.capacities[idx]
            
            if active_capacity >= self.total_demand:
                break
                
        return True
