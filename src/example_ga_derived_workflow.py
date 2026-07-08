"""
example_ga_derived_workflow.py
==============================
Example showing the complete GA-derived sampling workflow:
  1. Run hybrid GA with warmup (collect training data)
  2. Extract training data
  3. Train new ML model on GA-derived data
  4. Run hybrid GA again with new model

This is a reference implementation for the intended mentor's objective.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from parser import CFLPDataset
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
from dataset_generator import CFLPDatasetGenerator
from surrogate_model import CFLPSurrogateModel
from feature_engineering import CFLPFeatureEngineer
from training_pipeline import TrainingPipeline


def example_ga_derived_workflow():
    """
    Complete example: GA-derived sampling workflow.

    This demonstrates the intended approach from mentor's objective:
    "Initial generations of GA would generate training data for ML model.
     Trained ML model would then predict fitness. Only compute exact when
     prediction indicates potential to beat current best."
    """

    print("\n" + "=" * 80)
    print("  EXAMPLE: GA-Derived Sampling Workflow (Phase 1 & 3 Integration)")
    print("=" * 80)

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "data", "raw")
    model_dir = os.path.join(base_dir, "..", "data", "processed")

    instance_name = "cap71"
    instance_path = os.path.join(data_dir, f"{instance_name}.txt")

    # ========================================================================
    # STAGE 1: Generate Initial Training Data with Full Enumeration
    # ========================================================================
    print("\n[STAGE 1] Generate Initial Training Data (Full Enumeration)")
    print("-" * 80)

    dataset = CFLPDataset(instance_path)
    print(f"Loaded {instance_name}: m={dataset.num_facilities}, n={dataset.num_customers}")

    # For this example, we use pre-trained enumeration-based model
    initial_model_path = os.path.join(model_dir, "surrogate_xgboost.pkl")
    surrogate_v1 = CFLPSurrogateModel(dataset=dataset, model_type="xgboost")
    surrogate_v1.load(initial_model_path)
    print(f"Loaded initial surrogate (trained on enumerated data)")

    feature_engineer = CFLPFeatureEngineer(dataset)

    # ========================================================================
    # STAGE 2: Run Hybrid GA with Warmup (Collect GA-Derived Training Data)
    # ========================================================================
    print("\n[STAGE 2] Run Hybrid GA with Warmup (Collect Training Data)")
    print("-" * 80)
    print("Running GA with warmup_fraction=0.4 (first 40% of generations)")
    print("This collects exact LP evaluations for training data generation.")

    hybrid_ga_v1 = HybridMLGASolver(
        dataset=dataset,
        surrogate=surrogate_v1,
        feature_engineer=feature_engineer,
        pop_size=30,
        n_generations=50,
        cx_pb=0.8,
        mut_pb=0.2,
        mode="confidence_aware",
        uncertainty_threshold_pct=5.0,
        warmup_fraction=0.4,  # Collect data during first 40% of gens
        random_seed=42
    )

    result_v1 = hybrid_ga_v1.solve()

    print(f"\nGA Round 1 Results:")
    print(f"  Best cost (exact verification): ${result_v1['best_cost']:,.2f}")
    print(f"  Exact evaluations (warmup + fallback): {result_v1['exact_eval_count']}")
    print(f"  Surrogate evaluations: {result_v1['surrogate_eval_count']}")

    # ========================================================================
    # STAGE 3: Extract Training Data from GA-Collected Evaluations
    # ========================================================================
    print("\n[STAGE 3] Extract Training Data from GA Run")
    print("-" * 80)

    X_ga, y_ga = extract_training_data_from_ga(result_v1, dataset=dataset)

    print(f"Extracted {len(X_ga):,} training samples from GA-collected evaluations")
    print(f"  Cost range: ${y_ga.min():,.2f} to ${y_ga.max():,.2f}")
    print(f"  Cost mean: ${y_ga.mean():,.2f}")
    print(f"  Cost std: ${y_ga.std():,.2f}")

    # Save for reference
    gen = CFLPDatasetGenerator(dataset)
    ga_data_path = os.path.join(model_dir, f"training_data_ga_derived_{instance_name}_v1.npz")
    gen.save(X_ga, y_ga, ga_data_path)
    print(f"Saved to: {ga_data_path}")

    # ========================================================================
    # STAGE 4: Train New ML Model on GA-Derived Data
    # ========================================================================
    print("\n[STAGE 4] Train New ML Model on GA-Derived Data")
    print("-" * 80)
    print("Training new surrogate on GA-derived samples...")
    print("(In production, this would be integrated into TrainingPipeline)")

    # For this example, we simulate the retraining
    # In real workflow, this would call TrainingPipeline.train(corpus_path=ga_data_path)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    # Scale features (feature engineering already applied)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ga)

    # Train new model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_scaled, y_ga)

    # Evaluate on training set (simple metric for demo)
    train_pred = rf_model.predict(X_scaled)
    train_mape = np.mean(np.abs((y_ga - train_pred) / y_ga)) * 100
    print(f"New model training MAPE: {train_mape:.4f}%")
    print(f"Model trained with GA-derived data: {len(X_ga)} samples")

    # ========================================================================
    # STAGE 5: Prepare for Next GA Round (Would Use New Model)
    # ========================================================================
    print("\n[STAGE 5] Next Steps with GA-Derived Model")
    print("-" * 80)
    print("In Phase 2, the new model would be integrated:")
    print("  • Load new surrogate trained on GA-derived data")
    print("  • Run hybrid GA again with better initial predictions")
    print("  • Collect more samples in next warmup phase")
    print("  • Iterate until convergence")

    print("\n[ITERATION CYCLE]")
    print("  Round 1: GA run → collect data → train model → save")
    print("  Round 2: GA run → collect more data → retrain → save")
    print("  Round 3: Iterate with increasingly better models")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("  ✓ EXAMPLE COMPLETE: GA-Derived Sampling Workflow")
    print("=" * 80)

    print("\nKey Takeaways:")
    print("  1. GA collects exact evaluations during warmup")
    print("  2. Training data extracted automatically from these evaluations")
    print("  3. New ML models trained on GA-derived data (not full enumeration)")
    print("  4. Cycle repeats with increasingly better surrogates")
    print("\nThis aligns with mentor's intended approach:")
    print("  'Initial GA generations generate training data for ML model'")

    print(f"\nGenerated files:")
    print(f"  • {ga_data_path}")
    print(f"\nNext phase: Implement competitive fitness check (Phase 2)")
    print()


if __name__ == "__main__":
    try:
        example_ga_derived_workflow()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nMake sure you have:")
        print("  1. Pre-trained model at: data/processed/surrogate_xgboost.pkl")
        print("  2. CFLP instance at: data/raw/cap71.txt")
        print("  3. Run training_pipeline.py first to generate models")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
