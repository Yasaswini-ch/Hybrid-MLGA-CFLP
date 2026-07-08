import pulp
import numpy as np
import sys
sys.path.append("src")
from benchmark_uflp import load_uflp_dataset

d_capa = load_uflp_dataset("data/raw/capa.txt")

prob = pulp.LpProblem("UFLP_Scaled", pulp.LpMinimize)
y = pulp.LpVariable.dicts("y", range(d_capa.num_facilities), cat=pulp.LpBinary)
x = pulp.LpVariable.dicts("x", ((j, i) for j in range(d_capa.num_customers) for i in range(d_capa.num_facilities)), lowBound=0, cat=pulp.LpContinuous)

# Objective: sum(f_i * y_i) + sum( (c_ij / d_j) * x_ij )
prob += (
    pulp.lpSum(d_capa.fixed_costs[i] * y[i] for i in range(d_capa.num_facilities)) +
    pulp.lpSum((d_capa.transport_costs[j, i] / d_capa.demands[j]) * x[j, i] for j in range(d_capa.num_customers) for i in range(d_capa.num_facilities))
)

# Constraints
for j in range(d_capa.num_customers):
    prob += pulp.lpSum(x[j, i] for i in range(d_capa.num_facilities)) == 1.0
for i in range(d_capa.num_facilities):
    prob += pulp.lpSum(x[j, i] for j in range(d_capa.num_customers)) <= d_capa.num_customers * y[i]

solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
prob.solve(solver)

print("Status:", pulp.LpStatus[prob.status])
print("Objective value (Scaled Transport Cost):", pulp.value(prob.objective))
print("Hardcoded Ground Truth:", 17156454.48)
