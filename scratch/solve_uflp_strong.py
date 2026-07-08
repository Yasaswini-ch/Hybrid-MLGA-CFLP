import pulp
import numpy as np
import sys
sys.path.append("src")
from benchmark_uflp import load_uflp_dataset

d_capa = load_uflp_dataset("data/raw/capa.txt")

prob = pulp.LpProblem("UFLP_Strong_Solve", pulp.LpMinimize)
y = pulp.LpVariable.dicts("y", range(d_capa.num_facilities), cat=pulp.LpBinary)
x = pulp.LpVariable.dicts("x", ((j, i) for j in range(d_capa.num_customers) for i in range(d_capa.num_facilities)), lowBound=0, upBound=1.0, cat=pulp.LpContinuous)

# Objective
prob += pulp.lpSum(d_capa.fixed_costs[i] * y[i] for i in range(d_capa.num_facilities)) + pulp.lpSum(d_capa.transport_costs[j, i] * x[j, i] for j in range(d_capa.num_customers) for i in range(d_capa.num_facilities))

# Constraints
for j in range(d_capa.num_customers):
    prob += pulp.lpSum(x[j, i] for i in range(d_capa.num_facilities)) == 1.0

for j in range(d_capa.num_customers):
    for i in range(d_capa.num_facilities):
        prob += x[j, i] <= y[i]

print("Solving with 30s limit...")
solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=30, threads=4)
prob.solve(solver)

print("Status:", pulp.LpStatus[prob.status])
print("Objective:", pulp.value(prob.objective))

y_sol = np.array([pulp.value(y[i]) for i in range(d_capa.num_facilities)])
open_indices = np.where(y_sol > 0.5)[0]
print("Open facilities:", open_indices)
print("Fixed cost sum:", np.sum(d_capa.fixed_costs * y_sol))

# Calculate the actual transport cost sum for these open facilities
min_transports = np.min(d_capa.transport_costs[:, open_indices], axis=1)
print("Actual transport sum:", min_transports.sum())
print("Actual total cost:", np.sum(d_capa.fixed_costs * y_sol) + min_transports.sum())
