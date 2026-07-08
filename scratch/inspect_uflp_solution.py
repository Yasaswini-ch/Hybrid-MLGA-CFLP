import sys
sys.path.append("src")
from parser import CFLPDataset
import pulp

for name, exp in [
    ("capa1.txt", 19240822.449),
    ("capa2.txt", 18438046.543),
    ("capb1.txt", 13656379.578),
    ("capc1.txt", 11646596.974)
]:
    dataset = CFLPDataset(f"data/raw/{name}")
    prob = pulp.LpProblem("CFLP", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", range(dataset.num_facilities), cat=pulp.LpBinary)
    x = pulp.LpVariable.dicts("x", ((j, i) for j in range(dataset.num_customers) for i in range(dataset.num_facilities)), lowBound=0, upBound=1.0, cat=pulp.LpContinuous)
    
    prob += (
        pulp.lpSum(dataset.fixed_costs[i] * y[i] for i in range(dataset.num_facilities)) +
        pulp.lpSum(dataset.transport_costs[j, i] * x[j, i] for j in range(dataset.num_customers) for i in range(dataset.num_facilities))
    )
    
    for j in range(dataset.num_customers):
        prob += pulp.lpSum(x[j, i] for i in range(dataset.num_facilities)) == 1.0
        
    for i in range(dataset.num_facilities):
        prob += pulp.lpSum(dataset.demands[j] * x[j, i] for j in range(dataset.num_customers)) <= dataset.capacities[i] * y[i]
        
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
    prob.solve(solver)
    
    val = pulp.value(prob.objective)
    print(f"{name}: Calculated: {val:.3f} | Expected: {exp:.3f} | Diff: {val - exp:.3f}")
