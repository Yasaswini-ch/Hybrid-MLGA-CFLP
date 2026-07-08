import sys
sys.path.append("src")
sys.path.append(".")
from benchmark_uflp import load_uflp_dataset
from deap import base, creator, tools

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

from scratch.test_hybrid_30 import run_ga_uflp_hybrid

d_capc = load_uflp_dataset("data/raw/capc.txt")
cost = run_ga_uflp_hybrid(d_capc, random_seed=42)
print("GA run_ga_uflp_hybrid cost returned for capc:", cost)
