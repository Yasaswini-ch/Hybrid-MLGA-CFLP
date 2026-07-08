import sys
import os
import time
import numpy as np

sys.path.append('src')
from benchmark_uflp import load_uflp_dataset, greedy_uflp_seed, evaluate_uflp

def local_search_uflp(y_start, fixed_costs, transport_costs):
    y = list(y_start)
    current_cost = evaluate_uflp(y, fixed_costs, transport_costs)[0]
    num_fac = len(y)
    steps = 0
    
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
            steps += 1
            
        if not improved:
            break
            
    return y, current_cost, steps

def main():
    raw_dir = 'data/raw'
    for name in ('capa', 'capb', 'capc'):
        ds = load_uflp_dataset(os.path.join(raw_dir, f'{name}.txt'))
        y = greedy_uflp_seed(ds.fixed_costs, ds.transport_costs, 0.0)
        t0 = time.time()
        y_opt, cost_opt, steps = local_search_uflp(y, ds.fixed_costs, ds.transport_costs)
        t_elapsed = time.time() - t0
        print(f"{name} | Greedy: {evaluate_uflp(y, ds.fixed_costs, ds.transport_costs)[0]:,.2f} | Local Search: {cost_opt:,.2f} | Steps: {steps} | Time: {t_elapsed:.4f}s")

if __name__ == '__main__':
    main()
