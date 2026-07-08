import random
from typing import List, Any
import numpy as np

def tournament_select(individuals: List[Any], k: int, tournsize: int) -> List[Any]:
    """
    Selects k individuals from the population using Tournament Selection.
    
    Algorithm:
        For each selection slot, select 'tournsize' individuals at random and 
        return the fittest one. This maintains high selection pressure.
        
    Args:
        individuals (List[Any]): List of candidate individuals (chromosomes with fitness).
        k (int): Number of individuals to select.
        tournsize (int): Size of each tournament.
        
    Returns:
        List[Any]: List of selected offspring.
    """
    selected = []
    for _ in range(k):
        # Sample tournament candidates randomly with replacement
        candidates = random.choices(individuals, k=tournsize)
        
        # In DEAP, individuals are minimized, so we select the one with the minimum cost
        # Fitness values are stored in individual.fitness.values (a 1-tuple)
        best_candidate = min(candidates, key=lambda ind: ind.fitness.values[0])
        selected.append(best_candidate)
        
    return selected

def roulette_select(individuals: List[Any], k: int) -> List[Any]:
    """
    Selects k individuals using Roulette Wheel (Fitness Proportionate) Selection.
    
    Since CFLP is a minimization problem, we map objective costs to maximization 
    fitness values: f'_i = Z_max - Z_i + epsilon, where epsilon = 1.0 to ensure 
    even the worst individual has a non-zero selection chance.
    
    Args:
        individuals (List[Any]): List of candidate individuals.
        k (int): Number of individuals to select.
        
    Returns:
        List[Any]: List of selected offspring.
    """
    # Extract fitness costs
    costs = np.array([ind.fitness.values[0] for ind in individuals])
    
    # Filter out dead-end penalty costs (1e11 or greater) to prevent scaling collapse
    feasible_costs = costs[costs < 1e11]
    
    if len(feasible_costs) == 0:
        # If the entire population is infeasible, selection becomes random
        return random.choices(individuals, k=k)
        
    max_cost = np.max(feasible_costs)
    
    # Map costs: lower cost = higher selection weight
    # If an individual has a penalty cost, give it a tiny epsilon weight
    weights = []
    for c in costs:
        if c >= 1e11:
            weights.append(1e-5)
        else:
            weights.append(max_cost - c + 1.0)
            
    weights = np.array(weights)
    total_weight = np.sum(weights)
    
    if total_weight <= 0.0:
        # Fallback to random if weights sum to 0
        probabilities = [1.0 / len(individuals)] * len(individuals)
    else:
        probabilities = weights / total_weight
        
    # Select k individuals based on calculated probabilities
    selected = random.choices(individuals, weights=probabilities, k=k)
    return selected

def apply_elitism(old_population: List[Any], offspring: List[Any], elite_count: int = 1) -> List[Any]:
    """
    Applies Elitism filter to the population.
    Replaces the worst 'elite_count' individuals in the offspring generation 
    with the best 'elite_count' individuals from the previous generation.
    This guarantees that our historical best solutions are never lost.
    
    Args:
        old_population (List[Any]): Fittest population from generation t.
        offspring (List[Any]): Unrefined offspring population for generation t+1.
        elite_count (int): Number of elite individuals to preserve.
        
    Returns:
        List[Any]: Elitism-filtered offspring population.
    """
    if elite_count <= 0:
        return offspring
        
    # Find the elite individuals from the old population (sort ascending by cost)
    elites = sorted(old_population, key=lambda ind: ind.fitness.values[0])[:elite_count]
    
    # Sort offspring descending by cost (worst first)
    offspring_sorted_indices = np.argsort([ind.fitness.values[0] for ind in offspring])[::-1]
    
    # Clone and write elites into the worst offspring slots
    # Using deep cloning to prevent reference leaks
    from deap import base
    toolbox = base.Toolbox()
    
    for idx, elite in zip(offspring_sorted_indices[:elite_count], elites):
        offspring[idx] = toolbox.clone(elite)
        
    return offspring
