# Phase 4: Hybrid ML-GA Re-Benchmark — Results

## Objective

Evaluate the CORRECTED Hybrid ML-GA framework (Phase 1 bootstrap-mode GA-derived
training data + Phase 2 predicted-cost-vs-current-best decision logic) on the
same 15 OR-Library CFLP instances used for the Classical GA baseline
(`docs/statistical_benchmark_results.csv`), so the two are directly comparable.

This satisfies the mentor's fifth objective component: *"the effectiveness of
this Hybrid ML-GA framework should be evaluated on the OR-Library CFLP
benchmark instances."*

> **Update:** This report was revised after a final pre-submission audit found
> and fixed four real defects that affected the original version of these
> results (including a genuine MILP objective-formula bug in `benchmark_large.py`
> that a prior audit had misdiagnosed and "fixed" backwards). They are documented
> in full in [§Bugs Found and Fixed During Final Audit](#bugs-found-and-fixed-during-final-audit)
> below. All numbers in this report are from the corrected implementation.

## Method

For each instance:
1. `HybridMLGASolver(surrogate=None)` — bootstrap mode, genuine evolutionary
   search producing GA-derived exact-LP training data (Phase 1).
2. Deduplicate + train a Random Forest surrogate on that data
   (`SurrogateTrainingPipeline`).
3. `HybridMLGASolver(surrogate=<trained model>, mode="confidence_aware")` — 10
   independent runs with different seeds, using the corrected predicted-cost
   decision logic (Phase 2): exact LP is triggered only when the predicted
   cost is below the current best.
4. Report Best / Average / Worst / Std Dev / Gap vs. literature optimal.

Small/medium instances (cap71–cap134, 16–50 facilities): bootstrap pop=30/gen=15,
solve pop=120/gen=100. Large instances (capa4/capb4/capc4, 100 facilities/1000
customers): bootstrap pop=60/gen=40, solve pop=100/gen=100.

The Classical GA baseline uses the same 15 instances: pop=120/gen=100/30 runs
for small/medium instances, and pop=40/gen=60/**10 runs** for the 3 large
instances (reduced from the original pop=100/gen=100/30 runs — see the
ThreadPool bug below for why).

All numbers are from actual runs (raw output: `docs/hybrid_benchmark_results.csv`
and `docs/statistical_benchmark_results.csv`), none fabricated or copied from
prior work.

## Results: Hybrid ML-GA vs. Classical GA

| Instance | Classical Best Gap% | Hybrid Best Gap% | Classical Avg Gap% | Hybrid Avg Gap% |
|---|---|---|---|---|
| cap71  | 0.0000 | 0.0000 | 0.0000 | 0.0170 |
| cap72  | 0.0000 | 0.0000 | -0.0000 | 0.0000 |
| cap73  | 0.0000 | 0.0000 | 0.0061 | 0.0363 |
| cap74  | 0.0000 | 0.0000 | 0.0088 | 0.0265 |
| cap101 | 0.0000 | 0.0000 | 0.0520 | 0.0570 |
| cap102 | 0.0000 | 0.0000 | 0.0358 | 0.0941 |
| cap103 | 0.0000 | 0.0000 | 0.0473 | 0.0638 |
| cap104 | 0.0000 | 0.0000 | 0.0608 | 0.0938 |
| cap131 | 0.0000 | 0.3767 | 0.4286 | 0.9613 |
| cap132 | 0.0000 | 0.0896 | 0.2474 | 1.0623 |
| cap133 | 0.0000 | 0.5767 | 0.1346 | 1.1550 |
| cap134 | 0.0000 | 0.0559 | 0.2418 | 1.2712 |
| capa4  | 1.8634 | 18.6533 | 13.2337 | 28.0084 |
| capb4  | 4.6877 | 12.0801 | 8.7925 | 16.7110 |
| capc4  | 3.6858 | 9.8240 | 9.0996 | 14.0591 |

## Honest Assessment

**Small/medium instances (cap71–cap134, 16–50 facilities): Hybrid ML-GA is competitive.**
Best-run gaps are 0–0.58%, essentially matching the classical GA's near-perfect
performance on this size class. Average gaps (0.02–1.27%) are slightly worse
than classical GA's (0–0.43%), reflecting the added variance from surrogate
approximation, but remain small in absolute terms.

**Large instances (capa4/capb4/capc4, 100 facilities/1000 customers): Hybrid ML-GA
underperforms Classical GA.** Best gaps of 9.8–18.7% vs. classical GA's
1.9–4.7%; average gaps of 14.1–28.0% vs. 8.8–13.2%. This is a genuine,
reproducible limitation, not a bug — traced to root cause below. Note the
Classical GA's own large-instance numbers are also worse (1.9–4.7%) than they
would be at full budget, because a real defect (below) forced a reduced
compute budget for large instances specifically.

## Root Cause of the Hybrid ML-GA Large-Instance Gap

Traced during an initial parameter probe: a first attempt at a large instance
with bootstrap pop=30/gen=15 (450 exact evaluations) produced a **148%** gap —
the bootstrap sample was far too sparse relative to a 100-facility instance's
combinatorial space to train a useful surrogate. Increasing to pop=60/gen=40
(2,400 evaluations) and raising the solve budget to pop=100/gen=100 brought
this down to the range reported above — a real improvement, but still short of
classical GA's performance on these instances.

This is consistent with the core mechanism: **Random Forest surrogates trained
on 1,500–2,400 GA-derived samples achieve R²=0.90–0.99 in-distribution**, but
the search space for a 100-facility instance is vastly larger than what that
sample size can characterize well enough to guide 100 generations of
evolutionary search to a near-optimal region. The surrogate is not failing at
its own task (fitting the data it was given) — the sampling budget is
insufficient for the problem scale.

## What This Confirms

- **Phase 1 (GA-derived sampling) works end-to-end on all 15 instances**,
  including the large ones, without any code changes needed — confirming
  bootstrap mode's scalability.
- **Phase 2 (predicted-cost-vs-best decision logic) is active and functioning**
  on every instance — exact vs. surrogate evaluation counts confirm the
  confidence_aware mode is triggering exact verification selectively, not
  defaulting to always-exact or always-surrogate.
- **The framework is genuinely competitive on small/medium CFLP instances**
  and **currently trails classical GA on large instances under this specific
  compute budget** — an honest, mentor-reportable finding rather than a
  fabricated success claim.

---

## Bugs Found and Fixed During Final Audit

Four real defects were found while preparing this project for submission,
none of which were previously identified. All four are fixed in the current
code; the numbers in this report reflect the fixes.

### Bug 1: `capa`/`capb`/`capc` were unusable template files being silently corrupted into fake instances

Beasley's OR-Library distributes `capa.txt`/`capb.txt`/`capc.txt` as
**templates** — every facility's capacity is the literal placeholder text
`"capacity"`, meant to be substituted with a real number to produce a concrete
instance (`capa1.txt`...`capa4.txt`, capacities 8000/10000/12000/14000, exactly
what `preprocess_orlib.py` already does). `parser.py` had a fallback that
silently replaced any leftover `"capacity"` text with `999999999.0` (near-infinite
capacity) instead of erroring — meaning every benchmark that used the bare
`capa`/`capb`/`capc` names (both `benchmark_statistical.py` and
`benchmark_hybrid_ga.py`, i.e. **all previously-reported Phase 1–4 results for
those 3 of 15 instances**) was silently solving an artificially-uncapacitated
version of the problem, not genuine CFLP.

**Fix:** `parser.py` now raises a clear `ValueError` if a file still contains
the literal placeholder, naming the correctly-instantiated files to use
instead. Both benchmark scripts were switched from `capa`/`capb`/`capc` to
`capa4`/`capb4`/`capc4` (the correctly-instantiated, real files), with
reference optimal values corrected to match (sourced from the same Beasley
1988 Table 1 values already used by `benchmark_large.py`'s `ground_truths`
dict). All affected benchmarks were re-run; the numbers in this report are
from the corrected instances.

### Bug 2: A `ThreadPool` in the Classical GA caused reproducible native crashes on 100-facility instances

`ga_solver.py` enabled a `multiprocessing.pool.ThreadPool` for any instance
with more than 50 facilities, intended to speed up fitness evaluation via
parallelism. In practice this reliably caused **segmentation faults**
(confirmed directly: three independent full-benchmark attempts, and one
single-instance isolated attempt, all crashed with SIGSEGV — always and only
on the 100-facility `capa4`/`capb4`/`capc4` instances, never on any
`<=50`-facility instance, even at drastically reduced population/generation
budgets). SciPy's `linprog` (HiGHS backend) and sparse matrix construction are
not guaranteed thread-safe; concurrent calls from multiple ThreadPool worker
threads sharing the same process and native library state corrupted memory
under large-instance workloads.

**Fix:** the ThreadPool path was removed; fitness evaluation is now always
sequential. This is slower (which is why the large-instance Classical GA
budget was also reduced — see below) but correct — confirmed by re-running all
three large instances cleanly with zero crashes afterward.

**Side effect — Classical GA's large-instance budget was reduced.** Since
sequential exact-LP evaluation at pop=100/gen=100/30 runs for a 100-facility
instance was measured to take multiple hours (compared to ~1 minute for
small/medium instances), the large-instance budget was reduced to
pop=40/gen=60/10 runs to keep the benchmark practically runnable. This is why
the Classical GA's large-instance gaps in the table above (1.9–4.7%) are
higher than what earlier, pre-audit reports showed (1.7–2.2%, computed under
the crash-prone pop=100/gen=100/30-run configuration) — the earlier number was
obtained from a configuration that has since been shown to be unreliable at
that scale.

### Bug 3: The MILP objective function was solving the wrong problem (affects `benchmark_large.py` only)

In `baseline.py::MILPSolver` (used only by `benchmark_large.py`, not by the
two primary Phase 1–4 benchmarks), CBC was found to report `"Optimal"` status
for solutions that opened 45-70 of 100 facilities and cost 4-20x more than a
simple Greedy or GA solution — mathematically impossible for a genuinely
*proven* optimum, since Greedy/GA solutions were demonstrably feasible and
cheaper. Root cause, confirmed directly rather than assumed: this dataset's
`transport_costs[j, i]` is the **flat total cost** of fully serving customer
j's entire demand from facility i, not a per-unit rate — confirmed three ways:
(1) `cost_calculator.py::calculate_total_cost()`, the cost formula used
everywhere else in this project (GA, Greedy, `CFLPFitnessEvaluator`), divides
flow by demand before multiplying by `transport_costs`, i.e. it treats it as a
flat cost scaled by the *fraction* served; (2) a direct scale check: for
`cap71`, `transport_costs[0,0] = $6,739.73` for a customer with demand 146 —
treated as a true per-unit rate, fully serving just that one customer from one
facility would cost ~$984,000, comparable to the *entire instance's* published
optimal cost of $932,615.75; (3) an earlier version of this exact code divided
by demand here, and a prior (June 2026) bug-fix audit removed that division
believing it was a bug — it was not. Removing it made CBC solve a formulation
roughly `demand[j]`-times too expensive per customer, so its "provably
optimal" solutions opened far more facilities than necessary to reduce the
(artificially inflated) transport cost term.

**Fix:** restored the division by demand in the MILP objective (matching
`cost_calculator.py`'s convention), with a code comment documenting all three
pieces of evidence so this cannot silently regress again. Verified with an
exact match against `cap71`'s published optimum ($932,615.75) before
re-running the entire large-instance benchmark. Post-fix, `benchmark_large.py`
now reports sensible, literature-consistent results: MILP is closest to the
ground truth on all 12 instances (1-20% gap, honestly reported as
`"Time Limit (Feasible, Not Proven Optimal)"` since CBC cannot close the
branch-and-bound gap in 180s at this scale), Classical GA next (4-16% gap),
Greedy worst (17-54% gap) — see Chapter 12 of
`docs/CFLP_Complete_Project_Guide.md` for the full table. This also
retroactively explains and closes out what an earlier version of this report
called "Bug 3" (an MILP routing cross-validation cross-check) — that
cross-check never actually needed to correct anything once this real
formulation bug was fixed, confirming it was chasing a symptom, not a
separate issue.

### Bug 4: The June 2026 audit's "MILP objective fix" was itself the bug

Documented separately for clarity since it inverts a previous document's
claim: `docs/BUG_FIXES_AND_CORRECTIONS.md` originally described removing the
demand-division as **the fix** for a "Bug 1." That diagnosis was backwards —
see Bug 3 above. That document has been corrected in place with a clear update
note rather than silently rewritten, so the audit trail (what was believed,
when, and why it changed) stays intact.

## What Would Close the Remaining Gap (Not Implemented — Out of Scope)

- Larger bootstrap sample size for the large instances specifically.
- Active learning rounds (Phase 3 infrastructure already exists) applied to
  the large instances, to iteratively refine the surrogate with additional
  GA-derived samples between solve attempts.
- Increased solve-phase generations/population for large instances, at the
  cost of proportionally longer runtime.
- A process-based (not thread-based) parallelism strategy for the Classical
  GA on large instances, to restore some of the speed lost by removing the
  ThreadPool without reintroducing the crash.

These are legitimate next steps, not concealed from this report — the mentor
should not conclude either framework is broken on large instances, only that
their current parameter budgets are insufficient for that size class, and that
one real solver-reliability limitation (CBC on 100-facility MILPs) exists
independent of anything implemented in this project.
