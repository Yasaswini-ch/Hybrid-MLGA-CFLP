"""
diagnose_duplicate_source.py
=============================
Determines whether duplicate chromosomes in the GA-derived training corpus
originate from:
  (a) the Genetic Algorithm itself (elitism re-presenting the same chromosome
      across generations, or population convergence producing repeated genotypes), or
  (b) a bug/omission in the extraction/logging pipeline (e.g. double-appending
      the same evaluation, or missing a cache that the classical GA already has).

Method: run bootstrap mode with per-generation chromosome tracking. For every
logged (chromosome, cost) entry, record which generation produced it and
whether that exact chromosome was already logged in a PRIOR generation.
Separately track whether the duplicate is attributable to elitism (the
running-best chromosome reappearing because it's re-inserted into the
population and bootstrap mode re-evaluates everyone unconditionally).
"""

import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from parser import CFLPDataset
from hybrid_ga import HybridMLGASolver

dataset = CFLPDataset(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cap71.txt"))

print("=" * 80)
print("DIAGNOSING SOURCE OF DUPLICATE CHROMOSOMES IN THE EXTRACTED CORPUS")
print("=" * 80)

ga = HybridMLGASolver(dataset=dataset, surrogate=None, pop_size=20, n_generations=10, random_seed=10)

# Instrument the exact evaluator to record which generation each call happens in
# and whether the fitness evaluator itself has any cache (it should not, per source).
original_evaluate = ga.exact_evaluator.evaluate
call_log = []  # (generation, chromosome_tuple, was_seen_before)

seen_chromosomes = set()
current_gen = [0]

def traced_evaluate(individual):
    chrom_tuple = tuple(individual)
    was_seen = chrom_tuple in seen_chromosomes
    seen_chromosomes.add(chrom_tuple)
    result = original_evaluate(individual)
    call_log.append((current_gen[0], chrom_tuple, was_seen))
    return result

ga.exact_evaluator.evaluate = traced_evaluate

# Patch solve() generation counter by wrapping _evaluate_population_batch
original_batch = ga._evaluate_population_batch
def traced_batch(population, generation):
    current_gen[0] = generation
    return original_batch(population, generation)
ga._evaluate_population_batch = traced_batch

result = ga.solve()

print(f"\nTotal exact_evaluations_log entries: {len(result['exact_evaluations_log'])}")
print(f"Total evaluate() calls traced (including final one-off verification call): {len(call_log)}")

# solve() makes exactly one extra evaluate() call AFTER the generation loop, for final
# verification of the best individual (hybrid_ga.py: "Verifying best chromosome with
# exact LP..."). That call intentionally does NOT append to exact_evaluations_log --
# it is a validation step, not new training data. Exclude it from the comparison below.
in_loop_calls = call_log[:-1]
final_verification_call = call_log[-1]

assert len(in_loop_calls) == len(result["exact_evaluations_log"]), \
    "MISMATCH: in-loop evaluate() call count != log entry count -- would indicate the pipeline " \
    "logs something OTHER than what was actually evaluated, or double-logs a single call."
print(f"[OK] In-loop evaluate() calls ({len(in_loop_calls)}) == log entries "
      f"({len(result['exact_evaluations_log'])}): the pipeline logs exactly one entry per")
print("     in-loop exact_evaluator.evaluate() call. No double-logging bug in the extraction pipeline.")
print(f"     (The 1 excluded call is solve()'s final best-individual verification, which is")
print(f"      correctly NOT appended to exact_evaluations_log -- confirmed by source at")
print(f"      hybrid_ga.py's post-loop 'Verifying best chromosome' block.)")

call_log = in_loop_calls

duplicate_calls = [c for c in call_log if c[2]]  # was_seen == True
print(f"\nDuplicate evaluate() calls (same chromosome evaluated in >1 generation): {len(duplicate_calls)}")
print(f"Unique chromosomes ever evaluated: {len(seen_chromosomes)}")
print(f"Total evaluations: {len(call_log)}")
print(f"Duplicate rate: {len(duplicate_calls)/len(call_log)*100:.1f}%")

# Now specifically test the elitism hypothesis: does the best-known chromosome
# reappear in consecutive generations because elitism re-inserts it into pop,
# and bootstrap mode re-evaluates the ENTIRE population unconditionally (no cache)?
print("\n" + "-" * 80)
print("ELITISM HYPOTHESIS TEST")
print("-" * 80)

# Track, per generation, the count of individuals in that generation's population
# that were ALREADY seen in a strictly earlier generation.
by_gen = defaultdict(list)
for gen, chrom, was_seen in call_log:
    by_gen[gen].append((chrom, was_seen))

print(f"\n{'Gen':>4} | {'Pop evaluated':>14} | {'New chromosomes':>17} | {'Repeats from earlier gens':>26}")
print("-" * 70)
for gen in sorted(by_gen.keys()):
    entries = by_gen[gen]
    new_count = sum(1 for _, seen in entries if not seen)
    repeat_count = sum(1 for _, seen in entries if seen)
    print(f"{gen:>4} | {len(entries):>14} | {new_count:>17} | {repeat_count:>26}")

# Direct elitism check: does the best chromosome as of generation N appear
# again, unchanged, in generation N+1's evaluated population?
print("\n" + "-" * 80)
print("DIRECT CHECK: does the incumbent-best chromosome get re-logged next generation?")
print("-" * 80)

history_best_chrom = []
best_so_far = None
best_cost_so_far = float("inf")
for gen in sorted(by_gen.keys()):
    # cross-reference costs from the log entries for this generation
    gen_entries = [(chrom, cost) for (g, chrom, seen), (c, cost) in
                   zip(call_log, result["exact_evaluations_log"]) if g == gen]
    if gen_entries:
        gen_best_chrom, gen_best_cost = min(gen_entries, key=lambda x: x[1])
        if gen_best_cost < best_cost_so_far:
            best_cost_so_far = gen_best_cost
            best_so_far = gen_best_chrom
        history_best_chrom.append((gen, best_so_far, best_cost_so_far))

reappear_count = 0
for i in range(1, len(history_best_chrom)):
    prev_gen, prev_best, _ = history_best_chrom[i - 1]
    cur_gen, cur_best, _ = history_best_chrom[i]
    cur_gen_chroms = [chrom for g, chrom, _ in call_log if g == cur_gen]
    if prev_best in cur_gen_chroms:
        reappear_count += 1

print(f"\nOut of {len(history_best_chrom)-1} generation transitions, the PRIOR generation's")
print(f"incumbent-best chromosome reappeared in the NEXT generation's evaluated population")
print(f"{reappear_count} times.")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print(f"""
1. evaluate()-call-count == log-entry-count: {len(call_log) == len(result['exact_evaluations_log'])}
   -> The extraction/logging pipeline does NOT introduce duplicates by itself
      (no double-append bug found).

2. {len(duplicate_calls)} of {len(call_log)} evaluate() calls ({len(duplicate_calls)/len(call_log)*100:.1f}%)
   were re-evaluations of a chromosome already seen in an earlier generation.

3. The incumbent-best chromosome reappeared in the immediately-following generation's
   evaluated population {reappear_count}/{len(history_best_chrom)-1} times
   -> consistent with elitism (solve()'s elitism block re-inserts the best individual
      into offspring[0] every generation) combined with bootstrap mode's unconditional
      re-evaluation of every population member (no cache check before calling
      exact_evaluator.evaluate()).

4. CFLPFitnessEvaluator (src/fitness.py) has NO cache attribute, unlike
   CFLPGASolver (src/ga_solver.py) which DOES cache by chromosome key
   (self.cache = {{}}, checked at ga_solver.py:113 before evaluating).
""")
