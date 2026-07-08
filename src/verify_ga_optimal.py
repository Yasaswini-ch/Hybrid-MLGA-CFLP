import os
import sys
import time
import random
import numpy as np

# Ensure src/ is on the path
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from parser import CFLPDataset
from ga_solver import CFLPGASolver

# Optimal values to verify against (matching the user's reference scores)
VERIFY_OPTIMAL = {
    "cap71": 932615.750,
    "cap101": 796648.437,
    "cap131": 793439.562,
    "capa": 17156454.48
}

def verify():
    print("================================================================================")
    print(" DOUBLE-CHECKING GA CONVERGENCE TO EXACT BENCHMARK OPTIMAL SCORES")
    print("================================================================================")
    
    raw_dir = os.path.join(_SRC_DIR, "..", "data", "raw")
    
    for name, opt_val in VERIFY_OPTIMAL.items():
        file_path = os.path.join(raw_dir, f"{name}.txt")
        if not os.path.exists(file_path):
            print(f"[SKIP] File not found for {name}")
            continue
            
        dataset = CFLPDataset(file_path)
        solver = CFLPGASolver(dataset)
        
        # Run 3 times with different seeds
        best_costs = []
        for run in range(3):
            random.seed(42 + run)
            np.random.seed(42 + run)
            solver.clear_cache()
            
            best_cost, _, _ = solver.solve(
                pop_size=120,
                n_gen=100,
                cx_pb=0.8,
                mut_pb=0.3
            )
            best_costs.append(best_cost)
            
        best_found = min(best_costs)
        gap = ((best_found - opt_val) / opt_val) * 100.0
        
        print(f"\nInstance: {name.upper()}")
        print(f"  Literature Optimal: ${opt_val:,.2f}")
        print(f"  GA Best Discovered: ${best_found:,.2f}")
        print(f"  Optimality Gap    : {gap:+.4f}%")
        print(f"  All 3 Runs Costs  : " + ", ".join([f"${c:,.2f}" for c in best_costs]))
        
        if abs(gap) < 1e-5:
            print(f"  --> [VERIFIED] Reached 100% exact optimal score!")
        else:
            print(f"  --> [WARNING] Optimality gap is {gap:+.4f}%")
            
    print("\n================================================================================")

if __name__ == "__main__":
    verify()
