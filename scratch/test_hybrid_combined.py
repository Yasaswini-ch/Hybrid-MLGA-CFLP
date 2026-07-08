import sys
import os
import time
import random
import numpy as np

sys.path.append('src')
from benchmark_uflp import load_uflp_dataset, greedy_uflp_seed, evaluate_uflp, eval_cache
from deap import base, creator, tools

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def local_search_uflp(y_start, fixed_costs, transport_costs):
    y = list(y_start)
    current_cost = evaluate_uflp(y, fixed_costs, transport_costs)[0]
    num_fac = len(y)
    
    while True:
        improved = False
        best_neighbor = None
        best_neighbor_cost = current_cost
        
        for i in range(num_fac):
            y_new = list(y)
            y_new[i] = 1 - y_new[i]
            if sum(y_new) == 0:
                continue
            cost = evaluate_uflp(y_new, fixed_costs, transport_costs)[0]
            if cost < best_neighbor_cost:
                best_neighbor_cost = cost
                best_neighbor = y_new
                
        if best_neighbor is not None:
            y = best_neighbor
            current_cost = best_neighbor_cost
            improved = True
            
        if not improved:
            break
            
    return y, current_cost

def run_ga_uflp_combined(dataset, pop_size=120, n_gen=100, cx_pb=0.8, mut_pb=0.3, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    num_facilities = dataset.num_facilities
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", lambda: 1 if random.random() < 0.1 else 0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_facilities)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_uflp, fixed_costs=dataset.fixed_costs, transport_costs=dataset.transport_costs)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=2.0/num_facilities)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=pop_size)
    
    # 1. Deterministic Greedy Seed (Unoptimized)
    g0 = greedy_uflp_seed(dataset.fixed_costs, dataset.transport_costs, randomness=0.0)
    pop[0] = creator.Individual(g0)
    
    # 2. Deterministic Greedy Seed (Locally Optimized)
    pop[1] = creator.Individual(local_search_uflp(g0, dataset.fixed_costs, dataset.transport_costs)[0])
    
    # 3. 4 Randomized Greedy Seeds (Unoptimized)
    for k in range(2, 6):
        pop[k] = creator.Individual(greedy_uflp_seed(dataset.fixed_costs, dataset.transport_costs, randomness=0.5))
        
    # 4. 4 Randomized Greedy Seeds (Locally Optimized)
    for k in range(6, min(10, pop_size)):
        pop[k] = creator.Individual(local_search_uflp(greedy_uflp_seed(dataset.fixed_costs, dataset.transport_costs, randomness=0.5), dataset.fixed_costs, dataset.transport_costs)[0])
        
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
        
    # Local search ONLY at the end of the GA run
    opt_y, opt_cost = local_search_uflp(list(best_ind), dataset.fixed_costs, dataset.transport_costs)
    return opt_cost

def main():
    raw_dir = 'data/raw'
    
    # Check capb
    ds_b = load_uflp_dataset(os.path.join(raw_dir, 'capb.txt'))
    costs_b = []
    print("Benchmarking capb (30 runs)...")
    eval_cache.clear()
    t0 = time.time()
    for r in range(30):
        cost = run_ga_uflp_combined(ds_b, random_seed=42+r)
        costs_b.append(cost)
    t_elapsed = time.time() - t0
    print(f"capb Time: {t_elapsed:.2f}s | Min cost found: {min(costs_b)} | Target: 12979071.58")
    
    # Check capc
    ds_c = load_uflp_dataset(os.path.join(raw_dir, 'capc.txt'))
    costs_c = []
    print("Benchmarking capc (30 runs)...")
    eval_cache.clear()
    t0 = time.time()
    for r in range(30):
        cost = run_ga_uflp_combined(ds_c, random_seed=42+r)
        costs_c.append(cost)
    t_elapsed = time.time() - t0
    print(f"capc Time: {t_elapsed:.2f}s | Min cost found: {min(costs_c)} | Target: 11505594.33")

if __name__ == '__main__':
    main()
