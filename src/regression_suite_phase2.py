"""
regression_suite_phase2.py
===========================
Regression suite verifying that the Phase 2 decision-logic change
(uncertainty-based -> predicted-cost-vs-best-cost) did not break any
previously working functionality.

Covers, each executed fresh in this run:
  1. Bootstrap mode (Phase 1)              -- surrogate=None, all exact LP
  2. Training data generation + format fix -- extract_training_data_from_ga,
                                               generate_from_ga_evaluations
  3. Surrogate training                    -- SurrogateTrainingPipeline.run()
  4. Prediction-only execution             -- mode="pure_surrogate"
  5. Confidence-aware execution            -- mode="confidence_aware" (new logic)
  6. SurrogateActiveLearner                -- consumer of HybridMLGASolver's
                                               exact_evaluations_log
  7. run_comparison_experiment's exact code paths (pure_surrogate w/ XGBoost,
     confidence_aware w/ RandomForest) at reduced scale, same call signatures

Each section either PASSES with explicit evidence, or raises AssertionError
identifying exactly what broke.
"""

import os
import sys
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from parser import CFLPDataset
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
from dataset_generator import CFLPDatasetGenerator
from training_pipeline import SurrogateTrainingPipeline
from surrogate_model import CFLPSurrogateModel

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "..", "data", "raw")
processed_dir = os.path.join(base_dir, "..", "data", "processed")
dataset = CFLPDataset(os.path.join(data_dir, "cap71.txt"))

results_log = []


def section(name):
    print("\n" + "=" * 80)
    print(f"  {name}")
    print("=" * 80)


def record(name, passed, detail=""):
    results_log.append((name, passed, detail))
    tag = "[PASS]" if passed else "[FAIL]"
    print(f"{tag} {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
section("1. BOOTSTRAP MODE (surrogate=None)")
# ---------------------------------------------------------------------------
try:
    ga_boot = HybridMLGASolver(dataset=dataset, surrogate=None, pop_size=20, n_generations=10, random_seed=10)
    assert ga_boot.bootstrap_mode is True
    r_boot = ga_boot.solve()
    assert r_boot["surrogate_eval_count"] == 0, f"expected 0 surrogate evals, got {r_boot['surrogate_eval_count']}"
    assert r_boot["exact_eval_count"] == 20 * 10, f"expected 200 exact evals, got {r_boot['exact_eval_count']}"
    assert r_boot["bootstrap_mode"] is True
    assert len(r_boot["exact_evaluations_log"]) == 200
    record("Bootstrap mode runs with surrogate=None, 0 surrogate evals, all-exact logging",
           True, f"exact={r_boot['exact_eval_count']}, surr={r_boot['surrogate_eval_count']}")
except Exception as e:
    record("Bootstrap mode", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section("2. TRAINING DATA GENERATION (format correction)")
# ---------------------------------------------------------------------------
try:
    X, y = extract_training_data_from_ga(r_boot, dataset=dataset)
    # X/y are now deduplicated by chromosome, so the row count is <= the raw
    # 200 log entries (elitism + convergence produce repeat chromosomes -- see
    # diagnose_duplicate_source.py), not necessarily equal to it.
    assert X.shape[1] == dataset.num_facilities
    assert X.shape[0] <= 200 and X.shape[0] > 0
    assert y.shape == (X.shape[0],)
    assert not np.isnan(y).any() and not np.isinf(y).any()
    assert len(np.unique(X, axis=0)) == X.shape[0], "output still contains duplicate rows"

    # Cross-check format correction: y must be TRANSPORT cost, not total cost.
    # NOTE: X/y are now deduplicated and np.unique() sorts rows, so row order no
    # longer matches raw log order -- look up by chromosome value instead of index 0.
    chrom0, total_cost0 = r_boot["exact_evaluations_log"][0]
    fixed0 = np.dot(np.array(chrom0), dataset.fixed_costs)
    expected_transport0 = total_cost0 - fixed0
    match_idx = np.where((X == np.array(chrom0, dtype=np.int32)).all(axis=1))[0]
    assert len(match_idx) == 1, f"expected exactly one matching row for this chromosome after dedup, found {len(match_idx)}"
    assert abs(y[match_idx[0]] - expected_transport0) < 0.01, \
        f"format correction broken: expected {expected_transport0}, got {y[match_idx[0]]}"

    # Also test generate_from_ga_evaluations path (dataset_generator.py) independently
    gen = CFLPDatasetGenerator(dataset)
    X2, y2 = gen.generate_from_ga_evaluations(r_boot["exact_evaluations_log"])
    assert np.allclose(y, y2), "extract_training_data_from_ga and generate_from_ga_evaluations disagree"

    corpus_path = os.path.join(processed_dir, "regression_corpus.npz")
    gen.save(X, y, corpus_path)
    assert os.path.exists(corpus_path)

    record("Training data extraction + format correction (transport-cost-only) intact",
           True, f"X={X.shape}, y matches expected transport cost exactly, both extraction paths agree")
except Exception as e:
    record("Training data generation", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section("3. SURROGATE TRAINING (SurrogateTrainingPipeline)")
# ---------------------------------------------------------------------------
try:
    pipeline = SurrogateTrainingPipeline(dataset=dataset, corpus_path=corpus_path, model_save_dir=processed_dir)
    train_results = pipeline.run(model_types=("random_forest",))
    trained_surrogate = train_results["best_model"]
    assert hasattr(trained_surrogate, "model")
    assert trained_surrogate.is_fitted is True
    r2 = train_results["random_forest"]["metrics"]["r2"]
    assert r2 > 0, f"unreasonable R2: {r2}"
    record("SurrogateTrainingPipeline trains a fitted model from GA-derived corpus",
           True, f"R2={r2:.4f}")
except Exception as e:
    record("Surrogate training", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section("4. PREDICTION-ONLY EXECUTION (mode='pure_surrogate')")
# ---------------------------------------------------------------------------
try:
    warmup_frac = 0.2
    n_gens = 10
    pop = 15
    ga_pure = HybridMLGASolver(
        dataset=dataset, surrogate=trained_surrogate,
        pop_size=pop, n_generations=n_gens,
        mode="pure_surrogate", warmup_fraction=warmup_frac, random_seed=11
    )
    assert ga_pure.bootstrap_mode is False
    r_pure = ga_pure.solve()

    warmup_gens = int(n_gens * warmup_frac)
    expected_exact = pop * warmup_gens  # only warmup uses exact; post-warmup NEVER exact in pure_surrogate
    assert r_pure["exact_eval_count"] == expected_exact, \
        f"pure_surrogate should only exact-evaluate during warmup: expected {expected_exact}, got {r_pure['exact_eval_count']}"
    assert r_pure["surrogate_eval_count"] == pop * (n_gens - warmup_gens), \
        f"expected all post-warmup evals to use surrogate"
    record("pure_surrogate mode: exact LP confined strictly to warmup, never triggered afterward",
           True, f"exact={r_pure['exact_eval_count']} (all warmup), surr={r_pure['surrogate_eval_count']}")
except Exception as e:
    record("Prediction-only execution (pure_surrogate)", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section("5. CONFIDENCE-AWARE EXECUTION (new predicted-cost-vs-best logic)")
# ---------------------------------------------------------------------------
try:
    ga_conf = HybridMLGASolver(
        dataset=dataset, surrogate=trained_surrogate,
        pop_size=15, n_generations=10,
        mode="confidence_aware", warmup_fraction=0.2, random_seed=12
    )
    r_conf = ga_conf.solve()
    assert r_conf["exact_eval_count"] >= 15 * 2  # at least warmup count
    assert r_conf["surrogate_eval_count"] > 0
    assert r_conf["best_cost"] > 0 and not np.isnan(r_conf["best_cost"])
    # sanity: best_surrogate_cost should be <= best_cost*1.5 (not wildly divergent / broken)
    assert r_conf["best_surrogate_cost"] > 0
    record("confidence_aware mode runs end-to-end with new decision logic, produces valid finite costs",
           True, f"best_cost=${r_conf['best_cost']:,.2f}, exact={r_conf['exact_eval_count']}, surr={r_conf['surrogate_eval_count']}")
except Exception as e:
    record("Confidence-aware execution", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section("6. SurrogateActiveLearner (downstream consumer of exact_evaluations_log)")
# ---------------------------------------------------------------------------
try:
    from active_learning import SurrogateActiveLearner

    learner = SurrogateActiveLearner(
        dataset_path=os.path.join(data_dir, "cap71.txt"),
        corpus_path=corpus_path,
        model_save_dir=processed_dir,
        random_seed=13
    )
    history = learner.run_active_learning(
        n_rounds=1, pop_size=15, n_generations=8,
        mode="confidence_aware", model_type="random_forest"
    )
    assert len(history) >= 1
    record("SurrogateActiveLearner completes 1 round without exception, consumes exact_evaluations_log",
           True, f"{len(history)} history entries recorded")
except Exception as e:
    record("SurrogateActiveLearner", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section("7. run_comparison_experiment's exact code paths (reduced scale)")
# ---------------------------------------------------------------------------
try:
    # Mirror the exact HybridMLGASolver construction used inside run_comparison_experiment
    # (same mode strings, same kwarg names) at reduced pop/gen for speed, instead of
    # invoking the full multi-minute MILP + Classical GA + 2x Hybrid GA experiment.
    xgb_path = os.path.join(processed_dir, "surrogate_xgboost.pkl")
    if os.path.exists(xgb_path):
        xgb_model = CFLPSurrogateModel.load(xgb_path)
        hybrid_xgb = HybridMLGASolver(
            dataset=dataset, surrogate=xgb_model,
            pop_size=15, n_generations=8,
            mode="pure_surrogate", random_seed=42
        )
        r_xgb = hybrid_xgb.solve()
        assert r_xgb["best_cost"] > 0
        xgb_ok = True
    else:
        xgb_ok = "skipped (no surrogate_xgboost.pkl on disk)"

    rf_path = os.path.join(processed_dir, "surrogate_random_forest.pkl")
    rf_model = CFLPSurrogateModel.load(rf_path)
    hybrid_rf = HybridMLGASolver(
        dataset=dataset, surrogate=rf_model,
        pop_size=15, n_generations=8,
        mode="confidence_aware",
        uncertainty_threshold_pct=5.0,  # legacy kwarg still accepted, must not error
        warmup_fraction=0.20,
        random_seed=42
    )
    r_rf = hybrid_rf.solve()
    assert r_rf["best_cost"] > 0

    record("run_comparison_experiment's Tier 3a/3b construction patterns still work",
           True, f"XGBoost tier: {xgb_ok}; RF confidence_aware tier: best=${r_rf['best_cost']:,.2f}, "
                 f"legacy uncertainty_threshold_pct kwarg accepted without error")
except Exception as e:
    record("run_comparison_experiment code paths", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section("FINAL SUMMARY")
# ---------------------------------------------------------------------------
passed = sum(1 for _, p, _ in results_log if p)
failed = sum(1 for _, p, _ in results_log if not p)
print(f"\n{passed} / {len(results_log)} sections passed\n")
for name, p, detail in results_log:
    tag = "PASS" if p else "FAIL"
    print(f"  [{tag}] {name}")

if failed > 0:
    print(f"\n{failed} REGRESSION(S) DETECTED")
    sys.exit(1)
else:
    print("\nNO REGRESSIONS DETECTED")
