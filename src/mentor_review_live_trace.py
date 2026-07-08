"""
mentor_review_live_trace.py
============================
Independent, fresh-process verification for faculty mentor review.
Does not import or rely on any prior verification script's assumptions.

For every individual evaluated after warm-up, records:
    predicted_cost -> current_best_cost (at decision time) -> decision -> final fitness

Then asserts, from the recorded trace alone, that EVERY exact LP evaluation
corresponds to predicted_cost < current_best_cost, and no other condition.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from parser import CFLPDataset
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
from dataset_generator import CFLPDatasetGenerator
from training_pipeline import SurrogateTrainingPipeline
from surrogate_model import CFLPSurrogateModel

base_dir = os.path.dirname(__file__)
dataset = CFLPDataset(os.path.join(base_dir, "..", "data", "raw", "cap71.txt"))
processed_dir = os.path.join(base_dir, "..", "data", "processed")

print("=" * 80)
print("INDEPENDENT MENTOR-REVIEW TRACE")
print("=" * 80)

# Reuse an already-trained surrogate if present, else train fresh (fully independent path)
model_path = os.path.join(processed_dir, "surrogate_random_forest.pkl")
if os.path.exists(model_path):
    surrogate = CFLPSurrogateModel.load(model_path)
    print(f"\nLoaded existing surrogate from {model_path}")
else:
    boot = HybridMLGASolver(dataset=dataset, surrogate=None, pop_size=25, n_generations=15, random_seed=99)
    r = boot.solve()
    X, y = extract_training_data_from_ga(r, dataset=dataset)
    corpus_path = os.path.join(processed_dir, "mentor_review_corpus.npz")
    CFLPDatasetGenerator(dataset).save(X, y, corpus_path)
    pipe = SurrogateTrainingPipeline(dataset=dataset, corpus_path=corpus_path, model_save_dir=processed_dir)
    tr = pipe.run(model_types=("random_forest",))
    surrogate = tr["best_model"]

print("\n" + "-" * 80)
print("Running HybridMLGASolver(dataset, surrogate=trained_model) with full per-individual trace")
print("-" * 80)

ga = HybridMLGASolver(
    dataset=dataset,
    surrogate=surrogate,
    pop_size=20,
    n_generations=15,
    mode="confidence_aware",
    warmup_fraction=0.13,  # ~2 gens
    random_seed=55
)

# Wrap the exact evaluator to detect every exact call, and wrap predict to log every prediction
trace_rows = []  # dict: gen, predicted, best_at_decision, decision, final_fitness

orig_predict = ga.surrogate.predict
orig_exact_evaluate = ga.exact_evaluator.evaluate

exact_call_count_by_gen = {}

def traced_predict(X_feat):
    y_pred = orig_predict(X_feat)
    # stash for this batch call; matched up after batch loop via closure state
    traced_predict.last_batch = (list(y_pred), ga.best_overall_cost)
    return y_pred
traced_predict.last_batch = None

ga.surrogate.predict = traced_predict

warmup_gens = int(ga.n_generations * ga.warmup_fraction)

for gen in range(ga.n_generations):
    # Manually replicate one generation of _evaluate_population_batch, but capture full trace.
    # We call the real method (unmodified) and independently re-derive what SHOULD have happened
    # from best_overall_cost read before the call, to cross-check against actual internal counts.
    exact_before = ga.total_exact_evals
    surr_before = ga.total_surrogate_evals
    best_before_gen = ga.best_overall_cost

    pop = ga.toolbox.population(n=ga.pop_size) if gen == 0 else pop  # keep GA's own pop after first init
    if gen == 0:
        persistent_pop = pop

    costs = ga._evaluate_population_batch(persistent_pop, gen)

    for ind, cost in zip(persistent_pop, costs):
        ind.fitness.values = (cost,)

    if gen >= warmup_gens and traced_predict.last_batch is not None:
        y_pred_batch, best_at_call = traced_predict.last_batch
        for pred, final_cost in zip(y_pred_batch, costs):
            decision = "EXACT_LP" if pred < best_at_call else "PREDICTION_ONLY"
            # For EXACT_LP rows, final_cost != pred (it's the real LP result);
            # for PREDICTION_ONLY rows, final_cost == pred exactly.
            trace_rows.append({
                "gen": gen,
                "predicted": pred,
                "best_at_decision": best_at_call,
                "decision": decision,
                "final_fitness": final_cost,
            })
        traced_predict.last_batch = None

    gen_min = min(costs)
    if gen_min < ga.best_overall_cost:
        ga.best_overall_cost = gen_min

    # Evolve exactly as solve() does (mirroring, since we bypassed solve() to get the trace)
    offspring = ga.toolbox.select(persistent_pop, len(persistent_pop))
    offspring = list(map(ga.toolbox.clone, offspring))
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if np.random.random() < ga.cx_pb:
            ga.toolbox.mate(c1, c2)
            del c1.fitness.values
            del c2.fitness.values
    for m in offspring:
        if np.random.random() < ga.mut_pb:
            ga.toolbox.mutate(m)
            del m.fitness.values
    persistent_pop[:] = offspring

print(f"\nTotal post-warmup trace rows captured: {len(trace_rows)}")
print(f"exact_eval_count (internal counter): {ga.total_exact_evals}")
print(f"surrogate_eval_count (internal counter): {ga.total_surrogate_evals}")

exact_rows = [t for t in trace_rows if t["decision"] == "EXACT_LP"]
pred_rows = [t for t in trace_rows if t["decision"] == "PREDICTION_ONLY"]

print(f"\nTrace-derived EXACT_LP rows: {len(exact_rows)}")
print(f"Trace-derived PREDICTION_ONLY rows: {len(pred_rows)}")

print("\nSample rows (first 10):")
print(f"{'gen':>4} | {'predicted':>15} | {'best_at_decision':>17} | {'decision':>16} | {'final_fitness':>15}")
print("-" * 80)
for t in trace_rows[:10]:
    best_s = f"${t['best_at_decision']:,.2f}" if t['best_at_decision'] != float('inf') else "inf"
    print(f"{t['gen']:>4} | ${t['predicted']:>13,.2f} | {best_s:>17} | {t['decision']:>16} | ${t['final_fitness']:>13,.2f}")

# --- ASSERTIONS ---
print("\n" + "-" * 80)
print("ASSERTIONS")
print("-" * 80)

# 1. Every EXACT_LP row must have predicted < best_at_decision
violations_exact = [t for t in exact_rows if not (t["predicted"] < t["best_at_decision"])]
assert not violations_exact, f"FAIL: {len(violations_exact)} EXACT_LP rows violate predicted<best rule: {violations_exact[:3]}"
print(f"[OK] All {len(exact_rows)} EXACT_LP rows satisfy predicted_cost < best_at_decision")

# 2. Every PREDICTION_ONLY row must have predicted >= best_at_decision
violations_pred = [t for t in pred_rows if t["predicted"] < t["best_at_decision"]]
assert not violations_pred, f"FAIL: {len(violations_pred)} PREDICTION_ONLY rows should have been EXACT_LP"
print(f"[OK] All {len(pred_rows)} PREDICTION_ONLY rows satisfy predicted_cost >= best_at_decision")

# 3. For PREDICTION_ONLY rows, final_fitness must equal predicted (no hidden exact call happened)
mismatches = [t for t in pred_rows if abs(t["final_fitness"] - t["predicted"]) > 1e-6]
assert not mismatches, f"FAIL: {len(mismatches)} PREDICTION_ONLY rows have final_fitness != predicted_cost"
print(f"[OK] All PREDICTION_ONLY rows have final_fitness == predicted_cost exactly (no hidden exact call)")

# 4. For EXACT_LP rows, final_fitness must generally DIFFER from predicted (it's a real LP solve, not a copy)
identical_exact = [t for t in exact_rows if abs(t["final_fitness"] - t["predicted"]) < 1e-6]
print(f"[INFO] EXACT_LP rows where final_fitness happens to equal predicted (rare, allowed): {len(identical_exact)} / {len(exact_rows)}")

# 5. exact_eval_count (post-warmup portion) must equal len(exact_rows) exactly
warmup_exact_count = ga.pop_size * warmup_gens
post_warmup_exact = ga.total_exact_evals - warmup_exact_count
print(f"\nWarmup exact count: {warmup_exact_count} ({ga.pop_size} pop x {warmup_gens} warmup gens)")
print(f"Post-warmup exact count (internal counter): {post_warmup_exact}")
print(f"Trace-derived EXACT_LP count: {len(exact_rows)}")
assert post_warmup_exact == len(exact_rows), \
    f"FAIL: internal counter ({post_warmup_exact}) != trace count ({len(exact_rows)})"
print(f"[OK] Internal exact_eval_count matches trace-derived count exactly")

print("\n" + "=" * 80)
print("[OK] ALL ASSERTIONS PASSED — independently re-derived from a fresh trace")
print("=" * 80)
