import random
from typing import Tuple

def single_point_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """
    Executes a Single-Point Crossover in-place between two binary chromosomes.
    
    Algorithm:
        Select a random crossover point. Swap all genes after this point.
        
    Args:
        ind1 (list): The first parent individual.
        ind2 (list): The second parent individual.
        
    Returns:
        Tuple[list, list]: The modified parent individuals.
    """
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    
    # Swap segments in-place
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    
    return ind1, ind2

def two_point_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """
    Executes a Two-Point Crossover in-place between two binary chromosomes.
    
    Algorithm:
        Select two random crossover points. Swap all genes in the middle segment.
        This preserves circular patterns and is less disruptive than single-point crossover.
        
    Args:
        ind1 (list): The first parent individual.
        ind2 (list): The second parent individual.
        
    Returns:
        Tuple[list, list]: The modified parent individuals.
    """
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size - 2)
    cxpoint2 = random.randint(cxpoint1 + 1, size - 1)
    
    # Swap the middle segment in-place
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    
    return ind1, ind2

def uniform_crossover(ind1: list, ind2: list, indpb: float = 0.5) -> Tuple[list, list]:
    """
    Executes a Uniform Crossover in-place between two binary chromosomes.
    
    Algorithm:
        For each gene index, swap genes between parents with probability indpb.
        This provides high exploratory recombination behavior.
        
    Args:
        ind1 (list): The first parent individual.
        ind2 (list): The second parent individual.
        indpb (float): The independent probability of swapping a gene (default 0.5).
        
    Returns:
        Tuple[list, list]: The modified parent individuals.
    """
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
            
    return ind1, ind2
