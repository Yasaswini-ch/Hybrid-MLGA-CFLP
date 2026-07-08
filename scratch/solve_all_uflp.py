import pulp
import numpy as np
import sys
sys.path.append("src")
from benchmark_uflp import load_uflp_dataset

for name in ["capa", "capb", "capc"]:
    d = load_uflp_dataset(f"data/raw/{name}.txt")
    
    prob = pulp.LpProblem("UFLP", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", range(d.num_facilities), cat=pulp.LpBinary)
    x = pulp.LpVariable.dicts("x", ((j, i) for j in range(d.num_customers) for i in range(d.num_facilities)), lowBound=0, cat=pulp.LpContinuous)
    
    prob += pulp.lpSum(d.fixed_costs[i] * y[i] for i in range(d.num_facilities)) + pulp.lpSum(d.transport_costs[j, i] * x[j, i] for j in range(d.num_customers) for i in range(d.num_facilities))
    
    for j in range(d.num_customers):
        prob += pulp.lpSum(x[j, i] for i in range(d.num_facilities)) == 1.0
    for i in range(d.num_facilities):
        prob += pulp.lpSum(x[j, i] for j in range(d.num_customers)) <= d.num_customers * y[i]
        
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
    prob.solve(solver)
    
    print(f"{name} solve value: {pulp.value(prob.objective)}")
