import os
import sys
import time
import random
import numpy as np

# Setup DEAP creator classes globally
from deap import base, creator, tools

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def load_uflp_dataset(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    content = content.replace('capacity', '999999999.0')
    tokens = content.split()
    num_facilities = int(tokens[0])
    num_customers = int(tokens[1])
    token_ptr = 2
    capacities = []
    fixed_costs = []
    for _ in range(num_facilities):
        capacities.append(float(tokens[token_ptr]))
        fixed_costs.append(float(tokens[token_ptr+1]))
        token_ptr += 2
    demands = []
    transport_costs = []
    for _ in range(num_customers):
        demands.append(float(tokens[token_ptr]))
        token_ptr += 1
        costs = []
        for _ in range(num_facilities):
            costs.append(float(tokens[token_ptr]))
            token_ptr += 1
        transport_costs.append(costs)
    class UFLPDataset:
        def __init__(self):
            self.num_facilities = num_facilities
            self.num_customers = num_customers
            self.fixed_costs = np.array(fixed_costs)
            self.demands = np.array(demands)
            self.transport_costs = np.array(transport_costs)
            self.name = 'cap131'
    return UFLPDataset()

def evaluate_uflp(individual, fixed_costs, transport_costs, cache=None):
    if cache is not None:
        ind_tuple = tuple(individual)
        if ind_tuple in cache:
            return cache[ind_tuple]
            
    y = np.array(individual)
    if np.sum(y) == 0:
        val = (1e12,)
    else:
        open_indices = np.where(y == 1)[0]
        min_transport = np.min(transport_costs[:, open_indices], axis=1)
        total_transport = np.sum(min_transport)
        total_fixed = np.sum(fixed_costs * y)
        val = (total_fixed + total_transport,)
        
    if cache is not None:
        cache[ind_tuple] = val
    return val

def greedy_uflp_seed(fixed_costs, transport_costs, randomness=0.0):
    num_facilities = len(fixed_costs)
    y = np.zeros(num_facilities, dtype=np.int32)
    best_cost = float('inf')
    best_fac = -1
    for i in range(num_facilities):
        cost = fixed_costs[i] + np.sum(transport_costs[:, i])
        if cost < best_cost:
            best_cost = cost
            best_fac = i
    y[best_fac] = 1
    current_cost = best_cost
    while True:
        candidates = []
        open_indices = np.where(y == 1)[0]
        current_transports = np.min(transport_costs[:, open_indices], axis=1)
        for i in range(num_facilities):
            if y[i] == 1:
                continue
            new_transports = np.minimum(current_transports, transport_costs[:, i])
            new_cost = np.sum(new_transports) + np.sum(fixed_costs * y) + fixed_costs[i]
            if new_cost < current_cost:
                candidates.append((new_cost, i))
        if not candidates:
            break
        candidates.sort()
        if len(candidates) > 1 and random.random() < randomness:
            chosen_fac = candidates[random.randint(0, min(2, len(candidates)-1))][1]
        else:
            chosen_fac = candidates[0][1]
        y[chosen_fac] = 1
        open_indices = np.where(y == 1)[0]
        current_cost = np.sum(np.min(transport_costs[:, open_indices], axis=1)) + np.sum(fixed_costs * y)
    return list(y)

def run_ga_uflp(dataset, pop_size=200, n_gen=300, cx_pb=0.8, mut_pb=0.3, random_seed=42, use_cache=True):
    random.seed(random_seed)
    np.random.seed(random_seed)
    num_facilities = dataset.num_facilities
    
    cache = {} if use_cache else None
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", lambda: 1 if random.random() < 0.1 else 0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_facilities)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_uflp, fixed_costs=dataset.fixed_costs, transport_costs=dataset.transport_costs, cache=cache)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=2.0/num_facilities)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=pop_size)
    pop[0] = creator.Individual(greedy_uflp_seed(dataset.fixed_costs, dataset.transport_costs, randomness=0.0))
    for k in range(1, min(10, pop_size)):
        pop[k] = creator.Individual(greedy_uflp_seed(dataset.fixed_costs, dataset.transport_costs, randomness=0.5))
        
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        
    best_ind = tools.selBest(pop, 1)[0]
    best_cost = best_ind.fitness.values[0]
    
    for g in range(n_gen):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        for ind in offspring:
            if sum(ind) == 0:
                ind[random.randint(0, num_facilities - 1)] = 1
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        offspring_best = tools.selBest(offspring, 1)[0]
        if offspring_best.fitness.values[0] > best_cost:
            worst_idx = np.argmax([ind.fitness.values[0] for ind in offspring])
            offspring[worst_idx] = toolbox.clone(best_ind)
        else:
            best_ind = toolbox.clone(offspring_best)
            best_cost = best_ind.fitness.values[0]
            
        pop[:] = offspring
        
    cache_hits = len(cache) if cache is not None else 0
    return best_cost, cache_hits

def main():
    dataset = load_uflp_dataset('data/raw/cap131.txt')
    
    print("Running with cache...")
    t0 = time.time()
    costs_cached = []
    hits_total = 0
    for run in range(10):
        cost, hits = run_ga_uflp(dataset, pop_size=200, n_gen=300, random_seed=42+run, use_cache=True)
        costs_cached.append(cost)
        hits_total += hits
    t_cached = time.time() - t0
    print(f"Cached GA Time: {t_cached:.2f}s | Min cost: {min(costs_cached)} | Hits count (unique evaluations stored): {hits_total}")
    
    print("\nRunning without cache...")
    t0 = time.time()
    costs_uncached = []
    for run in range(10):
        cost, _ = run_ga_uflp(dataset, pop_size=200, n_gen=300, random_seed=42+run, use_cache=False)
        costs_uncached.append(cost)
    t_uncached = time.time() - t0
    print(f"Uncached GA Time: {t_uncached:.2f}s | Min cost: {min(costs_uncached)}")

if __name__ == "__main__":
    main()
