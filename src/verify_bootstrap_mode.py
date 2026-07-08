"""
verify_bootstrap_mode.py
=========================
Independent verification that HybridMLGASolver can now run WITHOUT a
pre-trained surrogate (surrogate=None) and still produce a correctly
formatted training dataset. This is the cold-start path that removes
the circular dependency identified in review.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from parser import CFLPDataset
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
from dataset_generator import CFLPDatasetGenerator

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "..", "data", "raw")
instance_path = os.path.join(data_dir, "cap71.txt")

print("=" * 80)
print("VERIFICATION: Bootstrap Mode (No Pre-Trained Surrogate Required)")
print("=" * 80)

dataset = CFLPDataset(instance_path)
print(f"\nLoaded cap71: m={dataset.num_facilities}, n={dataset.num_customers}")

print("\n" + "-" * 80)
print("STEP 1: Instantiate HybridMLGASolver with surrogate=None")
print("-" * 80)
print("(No CFLPSurrogateModel object created or loaded anywhere in this script.)")

ga = HybridMLGASolver(
    dataset=dataset,
    surrogate=None,          # <-- THE FIX: no pre-trained model required
    pop_size=20,
    n_generations=15,        # small run for cold-start data harvesting
    cx_pb=0.8,
    mut_pb=0.2,
    random_seed=42
)

assert ga.bootstrap_mode is True, "bootstrap_mode flag not set correctly"
print("\n[OK] HybridMLGASolver instantiated successfully with surrogate=None")
print(f"[OK] ga.bootstrap_mode = {ga.bootstrap_mode}")

print("\n" + "-" * 80)
print("STEP 2: Run solve() -- must not touch self.surrogate anywhere")
print("-" * 80)

result = ga.solve()

print("\n[OK] solve() completed without AttributeError on NoneType surrogate")
print(f"  Exact evaluations: {result['exact_eval_count']}")
print(f"  Surrogate evaluations: {result['surrogate_eval_count']} (must be 0)")
print(f"  bootstrap_mode in result: {result['bootstrap_mode']}")

assert result["surrogate_eval_count"] == 0, \
    f"FAIL: surrogate was used ({result['surrogate_eval_count']} evals) despite surrogate=None"
assert result["exact_eval_count"] == 20 * 15, \
    f"FAIL: expected every individual in every generation to be exact-evaluated, " \
    f"got {result['exact_eval_count']} (expected {20*15})"
assert result["bootstrap_mode"] is True

print("\n[OK] Zero surrogate evaluations confirmed")
print("[OK] All 300 (pop_size x n_generations) evaluations were exact LP")

print("\n" + "-" * 80)
print("STEP 3: Extract training data (same extraction path as before)")
print("-" * 80)

X, y = extract_training_data_from_ga(result, dataset=dataset)

print(f"\nExtracted dataset shape: X={X.shape}, y={y.shape}")

assert X.shape[0] == 300, f"FAIL: expected 300 samples, got {X.shape[0]}"
assert X.shape[1] == dataset.num_facilities
assert not np.isnan(y).any(), "FAIL: NaN values in extracted costs"
assert not np.isinf(y).any(), "FAIL: Inf values in extracted costs"
assert (X >= 0).all() and (X <= 1).all(), "FAIL: chromosome values out of [0,1]"

print("[OK] Shape correct: (N, m) chromosomes, (N,) transport costs")
print("[OK] No NaN/Inf values")
print("[OK] Binary chromosome values valid")

print("\n" + "-" * 80)
print("STEP 4: Verify data format is TRANSPORT COST (not total cost)")
print("-" * 80)

# Spot-check: reconstruct total cost and compare against raw log
raw_log = result["exact_evaluations_log"]
chrom0, total_cost0 = raw_log[0]
fixed0 = np.dot(np.array(chrom0), dataset.fixed_costs)
expected_transport0 = total_cost0 - fixed0

print(f"\nRaw GA log[0]: total_cost = ${total_cost0:,.2f}")
print(f"Fixed cost for that chromosome: ${fixed0:,.2f}")
print(f"Expected transport cost: ${expected_transport0:,.2f}")
print(f"Extracted y[0]: ${y[0]:,.2f}")

assert abs(y[0] - expected_transport0) < 0.01, "FAIL: format correction not applied"
print("[OK] Format correction confirmed (transport cost only, matches previous fix)")

print("\n" + "-" * 80)
print("STEP 5: Save dataset -- ready for TrainingPipeline, zero manual steps")
print("-" * 80)

gen = CFLPDatasetGenerator(dataset)
out_path = os.path.join(base_dir, "..", "data", "processed", "training_data_bootstrap_cap71.npz")
gen.save(X, y, out_path)
print(f"[OK] Saved to: {out_path}")

print("\n" + "=" * 80)
print("[OK] CIRCULAR DEPENDENCY VERIFIED REMOVED")
print("=" * 80)
print("""
Confirmed:
  1. HybridMLGASolver runs with surrogate=None (no pre-trained model needed)
  2. Every generation used exact LP (bootstrap mode forces this)
  3. surrogate_eval_count == 0 (surrogate object never touched)
  4. Extracted data is correctly formatted (transport cost, matches prior fix)
  5. Dataset is saved and ready for TrainingPipeline with zero manual steps

The GA can now generate its own initial training data from a cold start.
""")
