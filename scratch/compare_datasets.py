import sys
sys.path.append("src")
from parser import CFLPDataset
import pulp
import numpy as np

dataset = CFLPDataset("data/raw/capa1.txt")
solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)

# Formulation A (Fractional: x in [0, 1])
probA = pulp.LpProblem("FormulationA", pulp.LpMinimize)
yA = pulp.LpVariable.dicts("y", range(dataset.num_facilities), cat=pulp.LpBinary)
xA = pulp.LpVariable.dicts("x", ((j, i) for j in range(dataset.num_customers) for i in range(dataset.num_facilities)), lowBound=0, upBound=1.0, cat=pulp.LpContinuous)

probA += (
    pulp.lpSum(dataset.fixed_costs[i] * yA[i] for i in range(dataset.num_facilities)) +
    pulp.lpSum(dataset.transport_costs[j, i] * xA[j, i] for j in range(dataset.num_customers) for i in range(dataset.num_facilities))
)

for j in range(dataset.num_customers):
    probA += pulp.lpSum(xA[j, i] for i in range(dataset.num_facilities)) == 1.0
for i in range(dataset.num_facilities):
    probA += pulp.lpSum(dataset.demands[j] * xA[j, i] for j in range(dataset.num_customers)) <= dataset.capacities[i] * yA[i]

probA.solve(solver)
print("Formulation A Status:", pulp.LpStatus[probA.status])
print("Objective:", pulp.value(probA.objective))

y_sol = np.array([pulp.value(yA[i]) for i in range(dataset.num_facilities)])
open_indices = np.where(y_sol > 0.5)[0]
print("Open facilities:", open_indices)
print("Fixed cost sum:", np.sum(dataset.fixed_costs * y_sol))
print("Transport cost sum:", pulp.value(probA.objective) - np.sum(dataset.fixed_costs * y_sol))
