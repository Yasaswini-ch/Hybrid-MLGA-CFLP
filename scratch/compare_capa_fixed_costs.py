import sys
sys.path.append("src")
from benchmark_uflp import load_uflp_dataset
import numpy as np

d_capa = load_uflp_dataset("data/raw/capa.txt")
d_capa1 = load_uflp_dataset("data/raw/capa1.txt")
d_capa4 = load_uflp_dataset("data/raw/capa4.txt")

print("Capa vs Capa1 fixed costs equal:", np.array_equal(d_capa.fixed_costs, d_capa1.fixed_costs))
print("Capa vs Capa4 fixed costs equal:", np.array_equal(d_capa.fixed_costs, d_capa4.fixed_costs))
print("Capa1 vs Capa4 fixed costs equal:", np.array_equal(d_capa1.fixed_costs, d_capa4.fixed_costs))

print("Capa fixed costs (first 5):", d_capa.fixed_costs[:5])
print("Capa1 fixed costs (first 5):", d_capa1.fixed_costs[:5])
print("Capa4 fixed costs (first 5):", d_capa4.fixed_costs[:5])

print("Capa capacities (first 5):", d_capa.capacities[:5])
print("Capa1 capacities (first 5):", d_capa1.capacities[:5])
print("Capa4 capacities (first 5):", d_capa4.capacities[:5])
