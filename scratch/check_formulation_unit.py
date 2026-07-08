import pulp
import numpy as np
import sys
sys.path.append("src")
from parser import CFLPDataset

dataset = CFLPDataset("data/raw/capa1.txt")

# Unit cost formulation
prob = pulp.LpProblem("Unit_CFLP", pulp.LpMinimize)
y = pulp.LpVariable.dicts("y", range(dataset.num_facilities), cat=pulp.LpBinary)
x = pulp.LpVariable.dicts("x", ((j, i) for j in range(dataset.num_customers) for i in range(dataset.num_facilities)), lowBound=0, upBound=1.0, cat=pulp.LpContinuous)

# Objective (Unit Cost * Demand * Fraction)
prob += (
    pulp.lpSum(dataset.fixed_costs[i] * y[i] for i in range(dataset.num_facilities)) +
    pulp.lpSum(dataset.transport_costs[j, i] * dataset.demands[j] * x[j, i] for j in range(dataset.num_customers) for i in range(dataset.num_facilities))
)

# Constraints
for j in range(dataset.num_customers):
    prob += pulp.lpSum(x[j, i] for i in range(dataset.num_facilities)) == 1.0

for i in range(dataset.num_facilities):
    prob += pulp.lpSum(dataset.demands[j] * x[j, i] for j in range(dataset.num_customers)) <= dataset.capacities[i] * y[i]

solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
prob.solve(solver)

print("Status:", pulp.LpStatus[prob.status])
print("Objective value (Unit Cost Formulation):", pulp.value(prob.objective))
print("Literature Ground Truth:", 19241056.93)
