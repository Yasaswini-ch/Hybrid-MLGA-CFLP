"""
Verify: Does Phase 1 produce data in the correct format for TrainingPipeline?

This is an independent verification, not a test from the Phase 1 implementation.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from parser import CFLPDataset
from fitness import CFLPFitnessEvaluator
from cost_calculator import calculate_total_cost, calculate_fixed_costs
from solution_representation import CFLPSolution

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "..", "data", "raw")
instance_path = os.path.join(data_dir, "cap71.txt")

print("=" * 80)
print("VERIFICATION: Data Format Compatibility")
print("=" * 80)

dataset = CFLPDataset(instance_path)
evaluator = CFLPFitnessEvaluator(dataset)

# Test with a simple chromosome
test_chromosome = [1] * 5 + [0] * 11  # Open first 5 facilities
print(f"\nTest chromosome: {test_chromosome}")

# Evaluate via fitness evaluator (what GA collects)
ga_cost_tuple = evaluator.evaluate(test_chromosome)
ga_cost = ga_cost_tuple[0]

print(f"\nGA collected cost from evaluator: ${ga_cost:,.2f}")

# Now calculate what TrainingPipeline expects
print("\n" + "-" * 80)
print("ISSUE VERIFICATION")
print("-" * 80)

# What does the cost include?
X_test = np.array([test_chromosome], dtype=np.int32)

# Calculate fixed cost (what TrainingPipeline adds)
fixed_cost = np.dot(X_test[0], dataset.fixed_costs)
print(f"\nFixed costs (would be added by TrainingPipeline): ${fixed_cost:,.2f}")

# The cost from GA should be decomposable
# If ga_cost = fixed + transport, then:
# transport = ga_cost - fixed
inferred_transport = ga_cost - fixed_cost

print(f"Inferred transport cost: ${inferred_transport:,.2f}")

# What would TrainingPipeline do if we feed it ga_cost as y?
print("\n" + "-" * 80)
print("SCENARIO: Using GA cost directly in TrainingPipeline")
print("-" * 80)

print(f"\n1. TrainingPipeline loads: y = {ga_cost:,.2f}")
print(f"2. TrainingPipeline adds fixed costs: y_total = {ga_cost:,.2f} + {fixed_cost:,.2f}")
print(f"3. Result: y_total = ${ga_cost + fixed_cost:,.2f}")
print(f"\nWARNING: This is WRONG. Fixed costs are counted TWICE!")
print(f"Expected total: ${ga_cost:,.2f}")
print(f"TrainingPipeline would produce: ${ga_cost + fixed_cost:,.2f}")
print(f"Difference: ${fixed_cost:,.2f} extra (ERROR)")

# What would TrainingPipeline do if we feed it transport-only cost?
print("\n" + "-" * 80)
print("SCENARIO: Using transport cost only (correct format)")
print("-" * 80)

print(f"\n1. TrainingPipeline loads: y = {inferred_transport:,.2f}")
print(f"2. TrainingPipeline adds fixed costs: y_total = {inferred_transport:,.2f} + {fixed_cost:,.2f}")
print(f"3. Result: y_total = ${inferred_transport + fixed_cost:,.2f}")
print(f"\nCORRECT: Matches GA cost of ${ga_cost:,.2f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"""
Phase 1 extracts: y = {ga_cost:,.2f} (TOTAL COST)
TrainingPipeline expects: y = {inferred_transport:,.2f} (TRANSPORT ONLY)

Phase 1 data format is INCOMPATIBLE with TrainingPipeline.

To fix: Subtract fixed costs from extracted data before saving.
y_corrected = y_ga - (X_ga @ dataset.fixed_costs)
""")
