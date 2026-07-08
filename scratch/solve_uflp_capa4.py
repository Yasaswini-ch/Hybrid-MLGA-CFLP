import pulp
import numpy as np
import sys
sys.path.append("src")
from benchmark_uflp import load_uflp_dataset

d_capa4 = load_uflp_dataset("data/raw/capa4.txt")

prob = pulp.LpProblem("UFLP_Capa4", pulp.LpMinimize)
y = pulp.LpVariable.dicts("y", range(d_capa4.num_facilities), cat=pulp.LpBinary)
x = pulp.LpVariable.dicts("x", ((j, i) for j in range(d_capa4.num_customers) for i in range(d_capa4.num_facilities)), lowBound=0, cat=pulp.LpContinuous)

# Objective function
prob += pulp.lpSum(d_capa4.fixed_costs[i] * y[i] for i in range(d_capa4.num_facilities)) + pulp.lpSum(d_capa4.transport_costs[j, i] * x[j, i] for j in range(d_capa4.num_customers) for i in range(d_capa4.num_facilities))

# Constraints
for j in range(d_capa4.num_customers):
    prob += pulp.lpSum(x[j, i] for i in range(d_capa4.num_facilities)) == 1.0
for i in range(d_capa4.num_facilities):
    prob += pulp.lpSum(x[j, i] for j in range(d_capa4.num_customers)) <= d_capa4.num_customers * y[i]

solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
prob.solve(solver)

print("Status:", pulp.LpStatus[prob.status])
print("Objective value (UFLP on capa4.txt):", pulp.value(prob.objective))
print("Literature UFLP capa optimal cost:", 17156454.48)
