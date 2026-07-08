import pulp
import numpy as np
import sys
sys.path.append("src")
from parser import CFLPDataset

dataset = CFLPDataset("data/raw/capa1.txt")

prob = pulp.LpProblem("True_CFLP", pulp.LpMinimize)
y = pulp.LpVariable.dicts("y", range(dataset.num_facilities), cat=pulp.LpBinary)
x = pulp.LpVariable.dicts("x", ((j, i) for j in range(dataset.num_customers) for i in range(dataset.num_facilities)), lowBound=0, cat=pulp.LpContinuous)

# Objective: sum(f_i * y_i) + sum( (c_ij / d_j) * x_ij )
# where x_ij is the flow (0 to d_j)
prob += (
    pulp.lpSum(dataset.fixed_costs[i] * y[i] for i in range(dataset.num_facilities)) +
    pulp.lpSum((dataset.transport_costs[j, i] / dataset.demands[j]) * x[j, i] for j in range(dataset.num_customers) for i in range(dataset.num_facilities))
)

# Constraints
for j in range(dataset.num_customers):
    prob += pulp.lpSum(x[j, i] for i in range(dataset.num_facilities)) == dataset.demands[j]

for i in range(dataset.num_facilities):
    prob += pulp.lpSum(x[j, i] for j in range(dataset.num_customers)) <= dataset.capacities[i] * y[i]

solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
prob.solve(solver)

print("Status:", pulp.LpStatus[prob.status])
print("Objective value (True CFLP Solve):", pulp.value(prob.objective))
print("Literature Ground Truth:", 19241056.93)
