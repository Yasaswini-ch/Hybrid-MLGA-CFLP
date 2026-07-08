import sys
sys.path.append("src")
from parser import CFLPDataset
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import time

dataset = CFLPDataset("data/raw/capa1.txt")
m = dataset.num_facilities
n = dataset.num_customers

# Mock open set (45 facilities)
open_indices = np.random.choice(m, 45, replace=False)
num_open = len(open_indices)

# 1. Objective coefficients
c = []
for j in range(n):
    for i in open_indices:
        c.append(dataset.transport_costs[j, i])
c = np.array(c)

# 2. Build sparse A_eq
# Sum_{k=0..num_open-1} x[j, k] == demand[j]
rows_eq = []
cols_eq = []
data_eq = []
for j in range(n):
    for k in range(num_open):
        rows_eq.append(j)
        cols_eq.append(j * num_open + k)
        data_eq.append(1.0)
A_eq_sparse = sp.coo_matrix((data_eq, (rows_eq, cols_eq)), shape=(n, n * num_open)).tocsr()
b_eq = dataset.demands

# 3. Build sparse A_ub
# Sum_{j=0..num_customers-1} x[j, k] <= capacities[open_indices[k]]
rows_ub = []
cols_ub = []
data_ub = []
for k in range(num_open):
    for j in range(n):
        rows_ub.append(k)
        cols_ub.append(j * num_open + k)
        data_ub.append(1.0)
A_ub_sparse = sp.coo_matrix((data_ub, (rows_ub, cols_ub)), shape=(num_open, n * num_open)).tocsr()
b_ub = dataset.capacities[open_indices]

bounds = [(0.0, None)] * len(c)

# Time sparse solve
t0 = time.time()
res = linprog(c, A_ub=A_ub_sparse, b_ub=b_ub, A_eq=A_eq_sparse, b_eq=b_eq, bounds=bounds, method='highs')
elapsed = time.time() - t0

print(f"Status: {res.status} | Objective: {res.fun:,.2f} | Time: {elapsed * 1000:.2f} ms")
