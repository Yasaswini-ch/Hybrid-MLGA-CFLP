import sys
sys.path.append("src")
from parser import CFLPDataset
import numpy as np

d_cap71 = CFLPDataset("data/raw/cap71.txt")
print("Cap71 demands:", set(d_cap71.demands))
print("Cap71 capacities:", set(d_cap71.capacities))
print("Cap71 fixed costs:", set(d_cap71.fixed_costs))
