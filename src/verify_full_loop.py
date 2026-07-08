"""
verify_full_loop.py
====================
Runs the ENTIRE cold-start loop end to end, with no manual intervention
and no pre-existing model:

    HybridMLGASolver(surrogate=None)   [bootstrap, exact LP only]
        -> exact_evaluations_log
        -> extract_training_data_from_ga(dataset=dataset)   [format-corrected]
        -> CFLPDatasetGenerator.save()                       [.npz corpus]
        -> SurrogateTrainingPipeline.run()                   [trains real model]
        -> HybridMLGASolver(surrogate=<trained model>)       [second, informed run]

If this script completes, the circular dependency is proven removed:
the GA bootstrapped its own training data and a real surrogate now exists
without the user having supplied one.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from parser import CFLPDataset
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
from dataset_generator import CFLPDatasetGenerator
from training_pipeline import SurrogateTrainingPipeline

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "..", "data", "raw")
processed_dir = os.path.join(base_dir, "..", "data", "processed")
instance_path = os.path.join(data_dir, "cap71.txt")

print("=" * 80)
print("FULL COLD-START LOOP: GA -> Data -> Training -> Informed GA")
print("=" * 80)

dataset = CFLPDataset(instance_path)

# ---------------------------------------------------------------------------
print("\n[1/5] Running bootstrap GA (surrogate=None)...")
# ---------------------------------------------------------------------------
bootstrap_ga = HybridMLGASolver(
    dataset=dataset,
    surrogate=None,
    pop_size=25,
    n_generations=20,
    random_seed=42
)
result = bootstrap_ga.solve()
print(f"  -> {result['exact_eval_count']} exact evaluations collected, "
      f"{result['surrogate_eval_count']} surrogate evaluations (must be 0)")
assert result["surrogate_eval_count"] == 0

# ---------------------------------------------------------------------------
print("\n[2/5] Extracting + correcting training data...")
# ---------------------------------------------------------------------------
X, y = extract_training_data_from_ga(result, dataset=dataset)
print(f"  -> X={X.shape}, y={y.shape}")

# ---------------------------------------------------------------------------
print("\n[3/5] Saving corpus...")
# ---------------------------------------------------------------------------
gen = CFLPDatasetGenerator(dataset)
corpus_path = os.path.join(processed_dir, "training_data_bootstrap_loop_cap71.npz")
gen.save(X, y, corpus_path)

# ---------------------------------------------------------------------------
print("\n[4/5] Training surrogate model on bootstrap-derived corpus...")
# ---------------------------------------------------------------------------
pipeline = SurrogateTrainingPipeline(
    dataset=dataset,
    corpus_path=corpus_path,
    model_save_dir=processed_dir,
)
train_results = pipeline.run(model_types=("random_forest",))
trained_surrogate = train_results["best_model"]
print(f"  -> Trained model type: {train_results.get('best_model_type')}")
print(f"  -> Test R2: {train_results['random_forest']['metrics']['r2']:.4f}")

# ---------------------------------------------------------------------------
print("\n[5/5] Running SECOND hybrid GA using the model just trained...")
# ---------------------------------------------------------------------------
informed_ga = HybridMLGASolver(
    dataset=dataset,
    surrogate=trained_surrogate,
    pop_size=25,
    n_generations=20,
    mode="confidence_aware",
    warmup_fraction=0.1,
    random_seed=43
)
result2 = informed_ga.solve()
print(f"  -> Best cost: ${result2['best_cost']:,.2f}")
print(f"  -> Surrogate evaluations used: {result2['surrogate_eval_count']}")
assert result2["surrogate_eval_count"] > 0, \
    "FAIL: second run should have used the surrogate at least once"

print("\n" + "=" * 80)
print("[OK] FULL LOOP CLOSED WITHOUT ANY PRE-EXISTING MODEL")
print("=" * 80)
print("""
No CFLPSurrogateModel was created, loaded, or referenced anywhere in this
script before step 4. Step 1's GA ran entirely on exact LP evaluation
(surrogate=None), step 4 trained the FIRST model to ever exist for this
run purely from data the GA itself produced, and step 5 shows that model
being consumed by a second GA run. The circular dependency identified in
review (HybridMLGASolver requiring a surrogate to produce the data that
would train a surrogate) is resolved.
""")
