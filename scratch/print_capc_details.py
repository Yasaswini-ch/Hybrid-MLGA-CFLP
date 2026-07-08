import sys
sys.path.append("src")
from benchmark_uflp import load_uflp_dataset
import numpy as np

d_capa = load_uflp_dataset("data/raw/capa.txt")
print("Facilities:", d_capa.num_facilities)
print("Customers:", d_capa.num_customers)
print("Fixed costs (first 5):", d_capa.fixed_costs[:5])
print("Fixed cost of all facilities:", d_capa.fixed_costs.sum())

# What is the minimum possible transport cost?
min_transports = np.min(d_capa.transport_costs, axis=1)
print("Min transport sum:", min_transports.sum())

# What is the cost of opening all facilities?
print("Total cost if all facilities open:", d_capa.fixed_costs.sum() + min_transports.sum())
