import random
from typing import List
import numpy as np
from parser import CFLPDataset
from chromosome import CFLPChromosome

class CFLPPopulationGenerator:
    """
    Responsible for generating the initial population of facility opening chromosomes.
    Supports random initialization and heuristic seeding.
    """
    
    def __init__(self, dataset: CFLPDataset):
        """
        Initializes the generator with the current problem instance.
        
        Args:
            dataset (CFLPDataset): Parsed CFLP instance.
        """
        self.dataset = dataset
        self.m = dataset.num_facilities
        self.total_demand = np.sum(dataset.demands)
        
    def generate_random_individual(self) -> List[int]:
        """
        Generates an individual using a purely random unbiased binary distribution.
        Note: The resulting individual may violate physical capacity bounds.
        
        Returns:
            List[int]: A list of binary integers of length m.
        """
        return [random.choice([0, 1]) for _ in range(self.m)]
        
    def generate_heuristic_seeded_individual(self) -> List[int]:
        """
        Generates a smart, physically feasible individual.
        Calculates the physical minimum facility limit needed to cover total demand,
        then randomly opens at least that many facilities to ensure high feasibility.
        
        Returns:
            List[int]: A list of binary integers of length m.
        """
        individual = [0] * self.m
        
        # Calculate the absolute minimum number of active facilities required
        # assuming the facility with the maximum capacity is opened repeatedly
        max_capacity = np.max(self.dataset.capacities)
        min_facilities_needed = int(np.ceil(self.total_demand / max_capacity))
        
        # Ensure we open at least the physical lower limit
        num_to_open = random.randint(min_facilities_needed, self.m)
        
        # Randomly choose indices to open
        open_indices = random.sample(range(self.m), num_to_open)
        for idx in open_indices:
            individual[idx] = 1
            
        return individual
        
    def create_population(self, pop_size: int, heuristic_ratio: float = 0.5) -> List[List[int]]:
        """
        Creates a list of binary chromosomes forming the initial population.
        
        Args:
            pop_size (int): Total size of the population.
            heuristic_ratio (float): The fraction of the population to seed with 
                                     heuristic-based individuals (0.0 to 1.0).
                                     
        Returns:
            List[List[int]]: A population list containing binary chromosomes.
        """
        population = []
        num_heuristic = int(pop_size * heuristic_ratio)
        num_random = pop_size - num_heuristic
        
        # 1. Seed heuristic-based individuals
        for _ in range(num_heuristic):
            population.append(self.generate_heuristic_seeded_individual())
            
        # 2. Add random individuals
        for _ in range(num_random):
            population.append(self.generate_random_individual())
            
        return population
