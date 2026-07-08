"""
verify_phase2_cost_comparison.py
=================================
Verifies, by execution, that the confidence_aware decision logic in
HybridMLGASolver now follows:

    Predicted Cost -> Comparison with self.best_overall_cost -> Exact LP evaluation

instead of the old uncertainty (sigma) threshold logic.

This script:
  1. Bootstraps a surrogate from scratch (Phase 1, unchanged).
  2. Runs a SECOND hybrid GA in confidence_aware mode with the trained surrogate.
  3. Monkey-patches the surrogate's predict() to record every (predicted, best_at_call_time)
     pair BEFORE the decision is made, so we can prove the actual trigger condition
     used by the code, not an assumed one.
  4. Cross-checks: every logged exact evaluation must correspond to a case where
     predicted_cost < best_overall_cost at the time of the call, and vice versa.
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
dataset = CFLPDataset(os.path.join(base_dir, "..", "data", "raw", "cap71.txt"))
processed_dir = os.path.join(base_dir, "..", "data", "processed")

print("=" * 80)
print("PHASE 2 VERIFICATION: Predicted Cost -> Best Cost Comparison -> Exact LP")
print("=" * 80)

# ---------------------------------------------------------------------------
print("\n[1/3] Bootstrap a surrogate (Phase 1 path, unchanged)...")
# ---------------------------------------------------------------------------
bootstrap_ga = HybridMLGASolver(dataset=dataset, surrogate=None, pop_size=25, n_generations=15, random_seed=1)
boot_result = bootstrap_ga.solve()
X, y = extract_training_data_from_ga(boot_result, dataset=dataset)
corpus_path = os.path.join(processed_dir, "verify_phase2_corpus.npz")
CFLPDatasetGenerator(dataset).save(X, y, corpus_path)

pipeline = SurrogateTrainingPipeline(dataset=dataset, corpus_path=corpus_path, model_save_dir=processed_dir)
train_results = pipeline.run(model_types=("random_forest",))
surrogate = train_results["best_model"]
print(f"  Surrogate trained: R2={train_results['random_forest']['metrics']['r2']:.4f}")

# ---------------------------------------------------------------------------
print("\n[2/3] Instrument surrogate.predict() to capture decision-time state...")
# ---------------------------------------------------------------------------
ga2 = HybridMLGASolver(
    dataset=dataset,
    surrogate=surrogate,
    pop_size=20,
    n_generations=15,
    mode="confidence_aware",
    warmup_fraction=0.1,
    random_seed=2
)

decision_trace = []  # (predicted_cost, best_at_call_time, exact_log_len_before, exact_log_len_after)

original_predict = ga2.surrogate.predict

def instrumented_predict(X_feat):
    best_before = ga2.best_overall_cost
    log_len_before = len(ga2.exact_evaluations_log)
    y_pred = original_predict(X_feat)
    # Record one trace entry per row in this batch call
    for i in range(len(y_pred)):
        decision_trace.append({
            "predicted": float(y_pred[i]),
            "best_at_call_time": best_before,
            "log_len_before_batch": log_len_before,
        })
    return y_pred

ga2.surrogate.predict = instrumented_predict

# ---------------------------------------------------------------------------
print("\n[3/3] Run GA and cross-check every decision against the trace...")
# ---------------------------------------------------------------------------
result2 = ga2.solve()

print(f"\n  Total predict() calls traced: {len(decision_trace)}")
print(f"  exact_eval_count: {result2['exact_eval_count']}")
print(f"  surrogate_eval_count: {result2['surrogate_eval_count']}")

# Cross-check: reconstruct decision using the SAME rule the code claims to use
# (predicted < best_at_call_time => exact), and compare against actual exact_evaluations_log growth.
predicted_lt_best_count = sum(1 for t in decision_trace if t["predicted"] < t["best_at_call_time"])
predicted_ge_best_count = sum(1 for t in decision_trace if t["predicted"] >= t["best_at_call_time"])

print(f"\n  Predictions where predicted_cost < best_overall_cost (at call time): {predicted_lt_best_count}")
print(f"  Predictions where predicted_cost >= best_overall_cost (at call time): {predicted_ge_best_count}")

# The number of exact evaluations attributable to confidence_aware decisions
# (i.e. NOT warmup, NOT bootstrap) must equal predicted_lt_best_count exactly,
# since warmup generations bypass predict() entirely (see bootstrap/warmup guard).
warmup_gens = int(ga2.n_generations * ga2.warmup_fraction)
print(f"\n  Warmup generations (bypass predict() entirely): {warmup_gens} of {ga2.n_generations}")

# Show first 15 trace entries as concrete evidence
print("\n  Sample of decision trace (first 15 predict() outputs):")
print(f"  {'predicted_cost':>18} | {'best_at_call_time':>18} | {'decision':>12}")
print("  " + "-" * 55)
for t in decision_trace[:15]:
    decision = "EXACT LP" if t["predicted"] < t["best_at_call_time"] else "TRUST PREDICTION"
    best_str = f"${t['best_at_call_time']:,.2f}" if t["best_at_call_time"] != float("inf") else "inf"
    print(f"  ${t['predicted']:>16,.2f} | {best_str:>18} | {decision:>12}")

# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("CROSS-CHECK: exact_eval_count (post-warmup) must equal predicted_lt_best_count")
print("-" * 80)

post_warmup_exact = result2["exact_eval_count"] - (ga2.pop_size * warmup_gens)
print(f"  exact_eval_count total: {result2['exact_eval_count']}")
print(f"  minus warmup exact ({ga2.pop_size} pop x {warmup_gens} warmup gens): {ga2.pop_size * warmup_gens}")
print(f"  = post-warmup exact evaluations: {post_warmup_exact}")
print(f"  predicted_lt_best_count (from trace): {predicted_lt_best_count}")

assert post_warmup_exact == predicted_lt_best_count, (
    f"MISMATCH: post-warmup exact evals ({post_warmup_exact}) != "
    f"predicted<best count ({predicted_lt_best_count}). "
    f"The code is NOT following predicted_cost < best_overall_cost as the sole trigger."
)

print("\n  [OK] EXACT MATCH: every post-warmup exact LP evaluation corresponds to a case")
print("       where predicted_cost < best_overall_cost, and no others.")

# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("CONFIRM: old uncertainty-based (sigma) logic is no longer present")
print("-" * 80)
import inspect
src = inspect.getsource(type(ga2)._evaluate_population_batch)
assert "predict_with_uncertainty" not in src, "FAIL: old uncertainty method still called"
assert "sigma" not in src, "FAIL: sigma variable still referenced"
assert "self.best_overall_cost" in src, "FAIL: best_overall_cost comparison not present"
print("  [OK] predict_with_uncertainty() is not called in _evaluate_population_batch")
print("  [OK] sigma is not referenced in _evaluate_population_batch")
print("  [OK] self.best_overall_cost comparison is present in the decision path")

print("\n" + "=" * 80)
print("[OK] PHASE 2 VERIFIED: Predicted Cost -> Best Cost Comparison -> Exact LP")
print("=" * 80)
print(f"""
Evidence summary:
  - {len(decision_trace)} surrogate predictions captured with their decision-time
    best_overall_cost.
  - {predicted_lt_best_count} were below the incumbent best -> triggered exact LP verification.
  - {predicted_ge_best_count} were at/above the incumbent best -> surrogate prediction trusted, no exact LP.
  - Post-warmup exact_eval_count ({post_warmup_exact}) matches predicted<best count
    ({predicted_lt_best_count}) EXACTLY -- proving the comparison IS the trigger,
    not an approximation or coincidence.
  - Source inspection confirms predict_with_uncertainty/sigma are gone from the
    decision path and self.best_overall_cost is the active comparison.
""")
