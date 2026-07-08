"""
test_ga_derived_sampling.py
===========================
Test script for Phase 1: GA-Derived Sampling implementation.

Demonstrates the workflow:
  1. Run hybrid GA with warmup period (collects exact evaluations)
  2. Extract training data from collected evaluations
  3. Verify data quality
  4. Save as training corpus for future ML training
"""

import os
import sys
import numpy as np
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, os.path.dirname(__file__))

from parser import CFLPDataset
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
from dataset_generator import CFLPDatasetGenerator
from surrogate_model import CFLPSurrogateModel
from feature_engineering import CFLPFeatureEngineer


def test_ga_derived_sampling(instance_name="cap71"):
    """
    Test GA-derived sampling workflow on a small instance.

    Args:
        instance_name (str): Instance name to test (default: cap71 — smallest CFLP instance)
    """
    print("\n" + "=" * 70)
    print(f"  TEST: GA-Derived Sampling on {instance_name}")
    print("=" * 70)

    # --- Load instance ---
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "data", "raw")
    instance_path = os.path.join(data_dir, f"{instance_name}.txt")

    if not os.path.exists(instance_path):
        print(f"ERROR: Instance file not found: {instance_path}")
        return False

    dataset = CFLPDataset(instance_path)
    print(f"\nLoaded {instance_name}: m={dataset.num_facilities} facilities, "
          f"n={dataset.num_customers} customers")

    # --- Load pre-trained ML model (required for hybrid GA) ---
    model_dir = os.path.join(base_dir, "..", "data", "processed")
    model_path = os.path.join(model_dir, "surrogate_xgboost.pkl")

    if not os.path.exists(model_path):
        print(f"ERROR: Pre-trained model not found: {model_path}")
        print("       Please run training_pipeline.py first to generate ML models.")
        return False

    surrogate = CFLPSurrogateModel(model_type="xgboost")
    surrogate.load(model_path)
    print(f"Loaded ML surrogate: {model_path}")

    # --- Initialize feature engineer ---
    feature_engineer = CFLPFeatureEngineer(dataset)

    # --- Run Hybrid GA with warmup to collect exact evaluations ---
    print("\n[Step 1] Running Hybrid GA with warmup period...")
    print("  (Warmup ensures initial generations use exact LP, collecting training samples)")

    hybrid_ga = HybridMLGASolver(
        dataset=dataset,
        surrogate=surrogate,
        feature_engineer=feature_engineer,
        pop_size=30,
        n_generations=50,  # Keep small for testing
        cx_pb=0.8,
        mut_pb=0.2,
        mode="confidence_aware",
        uncertainty_threshold_pct=5.0,
        warmup_fraction=0.4,  # 40% of generations (20 gens) use exact LP
        random_seed=42
    )

    result = hybrid_ga.solve()

    print(f"\n  GA completed:")
    print(f"    Best cost (exact): ${result['best_cost']:,.2f}")
    print(f"    Total exact evaluations: {result['exact_eval_count']}")
    print(f"    Total surrogate evaluations: {result['surrogate_eval_count']}")

    # --- Extract training data from collected evaluations ---
    print("\n[Step 2] Extracting training data from GA-collected evaluations...")
    print("  (Correcting format: GA returns TOTAL COST, converting to TRANSPORT COST)")

    try:
        X_ga, y_ga = extract_training_data_from_ga(result, dataset=dataset)
        print(f"  Extracted {len(X_ga):,} training samples from GA run")
        print(f"  Feature matrix shape: {X_ga.shape}")
        print(f"  Cost array shape: {y_ga.shape}")
    except (KeyError, ValueError) as e:
        print(f"ERROR during data extraction: {e}")
        return False

    # --- Verify data quality ---
    print("\n[Step 3] Verifying extracted data quality...")

    print(f"  Chromosomes:")
    print(f"    Data type: {X_ga.dtype}")
    print(f"    Min value: {X_ga.min()}")
    print(f"    Max value: {X_ga.max()}")
    print(f"    Unique rows: {len(np.unique(X_ga, axis=0))}")

    print(f"  Costs:")
    print(f"    Data type: {y_ga.dtype}")
    print(f"    Min cost: ${y_ga.min():,.2f}")
    print(f"    Max cost: ${y_ga.max():,.2f}")
    print(f"    Mean cost: ${y_ga.mean():,.2f}")
    print(f"    Std dev: ${y_ga.std():,.2f}")

    # Check for duplicates
    unique_X = np.unique(X_ga, axis=0)
    if len(unique_X) < len(X_ga):
        dupes = len(X_ga) - len(unique_X)
        print(f"  WARNING: {dupes} duplicate chromosomes found")

    # Check for NaN or inf
    if np.isnan(y_ga).any() or np.isinf(y_ga).any():
        print(f"  ERROR: NaN or Inf values in costs!")
        return False
    else:
        print(f"  ✓ No NaN or Inf values")

    # --- Save extracted training data ---
    print("\n[Step 4] Saving extracted training data...")

    output_dir = os.path.join(base_dir, "..", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"training_data_ga_derived_{instance_name}.npz")

    gen = CFLPDatasetGenerator(dataset)
    gen.save(X_ga, y_ga, output_path)
    print(f"  Saved to: {output_path}")

    # --- Compare with full enumeration (for small instances only) ---
    if dataset.num_facilities <= 15:
        print("\n[Step 5] Comparing GA-derived data with full enumeration...")

        X_enum, y_enum = gen.generate_full_enumeration()

        print(f"  Full enumeration: {len(X_enum):,} samples")
        print(f"  GA-derived: {len(X_ga):,} samples")
        print(f"  GA coverage: {len(X_ga) / len(X_enum) * 100:.1f}% of solution space")

        # Check cost range
        enum_min, enum_max = y_enum.min(), y_enum.max()
        ga_min, ga_max = y_ga.min(), y_ga.max()

        print(f"\n  Cost range comparison:")
        print(f"    Enumeration: ${enum_min:,.2f} to ${enum_max:,.2f}")
        print(f"    GA-derived:  ${ga_min:,.2f} to ${ga_max:,.2f}")

        if ga_min <= enum_min and ga_max >= enum_max * 0.9:
            print(f"  ✓ GA-derived data covers good range of solution space")
        else:
            print(f"  ⚠ GA-derived data may miss extreme regions")
    else:
        print(f"\n[Step 5] Skipping full enumeration comparison (instance too large)")

    print("\n" + "=" * 70)
    print(f"  ✓ PHASE 1 TEST COMPLETE: GA-Derived Sampling Working")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Use {output_path}")
    print(f"     as training data for ML model retraining")
    print(f"  2. Train new surrogate on GA-derived data")
    print(f"  3. Run hybrid GA with new surrogate in subsequent iterations")
    print()

    return True


if __name__ == "__main__":
    success = test_ga_derived_sampling(instance_name="cap71")
    sys.exit(0 if success else 1)
