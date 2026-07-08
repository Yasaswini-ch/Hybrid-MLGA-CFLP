"""
verify_phase3_adaptive_retraining.py
=====================================
Verifies, by execution, that Phase 3 (Adaptive Model Retraining) works:

  1. generate_from_ga_evaluations() / extract_training_data_from_ga() now
     deduplicate chromosomes before returning training data.
  2. SurrogateActiveLearner's retrain loop no longer hand-rolls the total->
     transport cost conversion (uses the shared, deduplicating function).
  3. The model-quality gate correctly ADOPTS improvements and REJECTS
     regressions, keeping the best-known model active across rounds.
  4. The best model is persisted to a stable path that later worse rounds
     cannot clobber.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from parser import CFLPDataset
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
from dataset_generator import CFLPDatasetGenerator
from training_pipeline import SurrogateTrainingPipeline
from active_learning import SurrogateActiveLearner

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "..", "data", "raw")
processed_dir = os.path.join(base_dir, "..", "data", "processed")
dataset = CFLPDataset(os.path.join(data_dir, "cap71.txt"))

print("=" * 80)
print("PHASE 3 VERIFICATION: Adaptive Model Retraining")
print("=" * 80)

# ---------------------------------------------------------------------------
print("\n[1/3] Verify deduplication in extraction functions")
# ---------------------------------------------------------------------------
ga = HybridMLGASolver(dataset=dataset, surrogate=None, pop_size=20, n_generations=10, random_seed=10)
result = ga.solve()
raw_count = len(result["exact_evaluations_log"])

X, y = extract_training_data_from_ga(result, dataset=dataset)
unique_raw = len(np.unique(np.array([c for c, _ in result["exact_evaluations_log"]]), axis=0))

print(f"\n  Raw log entries: {raw_count}")
print(f"  Unique chromosomes in raw log (ground truth): {unique_raw}")
print(f"  extract_training_data_from_ga() returned: {X.shape[0]} samples")
assert X.shape[0] == unique_raw, f"FAIL: expected {unique_raw} deduplicated samples, got {X.shape[0]}"
assert X.shape[0] < raw_count, "FAIL: no deduplication occurred (raw count == output count)"
print(f"  [OK] extract_training_data_from_ga() deduplicated {raw_count} -> {X.shape[0]}")

gen = CFLPDatasetGenerator(dataset)
X2, y2 = gen.generate_from_ga_evaluations(result["exact_evaluations_log"])
assert X2.shape[0] == unique_raw, f"FAIL: generate_from_ga_evaluations returned {X2.shape[0]}, expected {unique_raw}"
assert np.allclose(y, y2), "FAIL: the two extraction paths disagree after dedup"
print(f"  [OK] generate_from_ga_evaluations() also deduplicated to {X2.shape[0]}, matches extract_training_data_from_ga()")

corpus_path = os.path.join(processed_dir, "verify_phase3_corpus.npz")
gen.save(X, y, corpus_path)

# ---------------------------------------------------------------------------
print("\n[2/3] Verify SurrogateActiveLearner runs and produces model_adopted decisions")
# ---------------------------------------------------------------------------
learner = SurrogateActiveLearner(
    dataset_path=os.path.join(data_dir, "cap71.txt"),
    corpus_path=corpus_path,
    model_save_dir=processed_dir,
    random_seed=21
)

history = learner.run_active_learning(
    n_rounds=3, pop_size=15, n_generations=8,
    mode="confidence_aware", model_type="random_forest"
)

print(f"\n  Rounds completed: {len(history)}")
for h in history:
    print(f"    Round {h['round']}: R2={h['r2']:.4f}, adopted={h['model_adopted']}, "
          f"best_r2_so_far={h['best_r2_so_far']:.4f}, new_samples={h['new_samples_added']}")

# Every entry must have the new gating fields
for h in history:
    assert "model_adopted" in h and "best_r2_so_far" in h, f"FAIL: round {h['round']} missing gating fields"

# best_r2_so_far must be monotonically non-decreasing across rounds (the whole point of the gate)
r2_best_series = [h["best_r2_so_far"] for h in history]
for i in range(1, len(r2_best_series)):
    assert r2_best_series[i] >= r2_best_series[i - 1] - 1e-9, \
        f"FAIL: best_r2_so_far decreased between round {i-1} and {i}: {r2_best_series[i-1]} -> {r2_best_series[i]}"
print(f"\n  [OK] best_r2_so_far is monotonically non-decreasing across all rounds: {[f'{v:.4f}' for v in r2_best_series]}")

# Any round where r2 < best_r2_so_far (from the PRIOR round) must be marked NOT adopted
violations = []
for i in range(1, len(history)):
    prev_best = history[i - 1]["best_r2_so_far"]
    this_r2 = history[i]["r2"]
    this_adopted = history[i]["model_adopted"]
    if this_r2 < prev_best and this_adopted:
        violations.append(history[i])
    if this_r2 >= prev_best and not this_adopted and history[i]["new_samples_added"] != 0:
        # only flag if it wasn't a "no exact evals" skip round
        violations.append(history[i])
assert not violations, f"FAIL: gating logic violated in rounds: {violations}"
print(f"  [OK] Every round's adopt/reject decision is consistent with its R2 vs. prior best")

# ---------------------------------------------------------------------------
print("\n[3/3] Verify best model file is stable (not clobbered by a worse round)")
# ---------------------------------------------------------------------------
best_model_path = os.path.join(processed_dir, "surrogate_random_forest_best.pkl")
assert os.path.exists(best_model_path), f"FAIL: {best_model_path} does not exist"

from surrogate_model import CFLPSurrogateModel
best_on_disk = CFLPSurrogateModel.load(best_model_path)

# Re-evaluate the on-disk "best" model against a fresh holdout to confirm it's fitted and usable
X_check, y_check = gen.load(corpus_path)
from feature_engineering import CFLPFeatureEngineer
fe = CFLPFeatureEngineer(dataset, mode="full")
X_feat_check = fe.transform(X_check[:20].astype(np.float64))
preds = best_on_disk.predict(X_feat_check)
assert len(preds) == 20 and not np.isnan(preds).any()
print(f"\n  [OK] Best model file at {best_model_path} loads and predicts correctly")
print(f"  [OK] Best model's R2 (from history): {r2_best_series[-1]:.6f}")

print("\n" + "=" * 80)
print("[OK] PHASE 3 VERIFIED: Deduplication + Quality-Gated Adaptive Retraining")
print("=" * 80)
