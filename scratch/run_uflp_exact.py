import sys
sys.path.append("src")
from benchmark_uflp import load_uflp_dataset, solve_uflp_exact

d_capa = load_uflp_dataset("data/raw/capa.txt")
print("solve_uflp_exact value:", solve_uflp_exact(d_capa))
