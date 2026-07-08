import pulp
import numpy as np
import sys
sys.path.append("src")
from parser import CFLPDataset

dataset = CFLPDataset("data/raw/capa1.txt")

# Correct formulation
prob = pulp.LpProblem("Correct_CFLP", pulp.LpMinimize)
y = pulp.LpVariable.dicts("y", range(dataset.num_facilities), cat=pulp.LpBinary)
x = pulp.LpVariable.dicts("x", ((j, i) for j in range(dataset.num_customers) for i in range(dataset.num_facilities)), lowBound=0, upBound=1.0, cat=pulp.LpContinuous)

# Objective
prob += (
    pulp.lpSum(dataset.fixed_costs[i] * y[i] for i in range(dataset.num_facilities)) +
    pulp.lpSum(dataset.transport_costs[j, i] * x[j, i] for j in range(dataset.num_customers) for i in range(dataset.num_facilities))
)

# Constraints
for j in range(dataset.num_customers):
    prob += pulp.lpSum(x[j, i] for i in range(dataset.num_facilities)) == 1.0

for i in range(dataset.num_facilities):
    prob += pulp.lpSum(dataset.demands[j] * x[j, i] for j in range(dataset.num_customers)) <= dataset.capacities[i] * y[i]

solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
prob.solve(solver)

print("Status:", pulp.LpStatus[prob.status])
print("Objective:", pulp.value(prob.objective))

num_open = sum(1 for i in range(dataset.num_facilities) if pulp.value(y[i]) > 0.5)
fixed_sum = sum(dataset.fixed_costs[i] for i in range(dataset.num_facilities) if pulp.value(y[i]) > 0.5)
trans_sum = sum(dataset.transport_costs[j, i] * pulp.value(x[j, i]) for j in range(dataset.num_customers) for i in range(dataset.num_facilities))

print("Number of open facilities:", num_open)
print("Fixed cost sum:", fixed_sum)
print("Transport cost sum:", trans_sum)
print("Sum of fixed and transport:", fixed_sum + trans_sum)
