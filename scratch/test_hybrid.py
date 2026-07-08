import sys
sys.path.append("src")
from parser import CFLPDataset
from ga_solver import CFLPGASolver

dataset = CFLPDataset("data/raw/cap71.txt")
solver = CFLPGASolver(dataset)
best_cost, best_y, history = solver.solve(pop_size=10, n_gen=5)
print("GA Best Cost:", best_cost)
