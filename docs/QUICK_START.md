# Quick Start Guide

**TL;DR**: Run benchmarks and understand results in 5 minutes

---

## 🚀 I Just Want to Run a Benchmark

```bash
# Setup (one time only)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# Run standard benchmark (8 minutes)
python src/benchmark_statistical.py

# Check results
head docs/statistical_benchmark_results.csv
```

**What you got**:
- `docs/statistical_benchmark_results.csv` — Table with GA performance stats
- `docs/statistical_benchmark_results.png` — Convergence graph

---

## 📚 I Want to Understand the Code

**Start here**, in order:

1. **What problem are we solving?**
   - Read: [IMPLEMENTATION_ARCHITECTURE.md](IMPLEMENTATION_ARCHITECTURE.md) → Problem Formulation section

2. **What approaches are implemented?**
   - Read: [IMPLEMENTATION_ARCHITECTURE.md](IMPLEMENTATION_ARCHITECTURE.md) → Solution Approaches section
   - Summary: MILP (exact), Greedy (fast), GA (balanced, PRIMARY), Modular GA (experimental), Hybrid ML-GA (requires ML validation)

3. **How do I run each?**
   - Read: [IMPLEMENTATION_ARCHITECTURE.md](IMPLEMENTATION_ARCHITECTURE.md) → Running Benchmarks section
   - Or jump to next section below

4. **What's the file structure?**
   - Read: [IMPLEMENTATION_ARCHITECTURE.md](IMPLEMENTATION_ARCHITECTURE.md) → Module Reference section
   - Each file explained with purpose and key functions

---

## 🔧 I Want to Run Specific Solvers

### Run Exact MILP Solver on cap41

```python
from src.parser import CFLPDataset
from src.baseline import MILPSolver

dataset = CFLPDataset("data/raw/cap41.txt")
milp = MILPSolver(dataset)
cost, y, x, status = milp.solve(timeout_sec=60)

print(f"Optimal cost: ${cost:,.2f}")
print(f"Facilities opened: {sum(y)}/{len(y)}")
print(f"Status: {status}")
```

### Run Greedy Heuristic

```python
from src.baseline import GreedySolver

greedy = GreedySolver(dataset)
cost, y, x = greedy.solve()

print(f"Greedy cost: ${cost:,.2f}")
```

### Run Classical GA (PRIMARY)

```python
from src.ga_solver import CFLPGASolver

ga = CFLPGASolver(dataset)
best_cost, best_y, history = ga.solve(
    pop_size=120,    # Population size
    n_gen=100,       # Generations
    cx_pb=0.8,       # Crossover probability
    mut_pb=0.2       # Mutation probability
)

print(f"GA best cost: ${best_cost:,.2f}")
print(f"Convergence history: {history['min_cost']}")
```

---

## 🧪 I'm Getting Weird Results

### Issue 1: Results seem cached or wrong

**Check**: Are standard deviations zero?
```bash
awk -F',' 'NR > 1 {print $1, "std=" $7}' docs/statistical_benchmark_results.csv | head -5
```

**Expected**: All std > 0 (e.g., `cap71 std=X.XX` where X.XX > 0)  
**If zero**: Bug #2 may still exist; check [BUG_FIXES_AND_CORRECTIONS.md](BUG_FIXES_AND_CORRECTIONS.md)

### Issue 2: MILP costs look identical or cached

**Check**: Are MILP costs unique per instance?
```bash
awk -F',' 'NR > 1 {print $1, $3}' docs/large_benchmark_results.csv | sort -u | wc -l
```

**Expected**: 12 different costs  
**If <= 3**: Bug #1 may still exist; check [BUG_FIXES_AND_CORRECTIONS.md](BUG_FIXES_AND_CORRECTIONS.md)

### Issue 3: Code compilation errors

**Verify all modules compile**:
```bash
python -m py_compile src/baseline.py src/ga_solver.py src/benchmark_statistical.py
```

If you get errors, see [REPRODUCIBILITY_AND_VERIFICATION.md](REPRODUCIBILITY_AND_VERIFICATION.md) → Troubleshooting

---

## ✅ I'm Skeptical About Results — How Do I Verify?

Read [BUG_FIXES_AND_CORRECTIONS.md](BUG_FIXES_AND_CORRECTIONS.md) for:
- What bugs existed and how they were fixed
- Evidence from original data that bugs existed
- How to verify each fix

Then run [REPRODUCIBILITY_AND_VERIFICATION.md](REPRODUCIBILITY_AND_VERIFICATION.md) → Verification Tests section:

```bash
# TEST 1: Code compiles
python -m py_compile src/baseline.py

# TEST 2: MILP costs are unique (not cached)
python src/benchmark_large.py
awk -F',' 'NR > 1 {print $3}' docs/large_benchmark_results.csv | sort -u | wc -l
# Expected: 12

# TEST 3: GA has variation (std dev > 0, not cached)
python src/benchmark_statistical.py
awk -F',' 'NR > 1 && $7 > 0' docs/statistical_benchmark_results.csv | wc -l
# Expected: 15

# TEST 4: Different seeds produce different results
sed -i 's/BASE_SEED = 42/BASE_SEED = 99/g' src/benchmark_statistical.py
python src/benchmark_statistical.py
# Results should differ from previous run
```

---

## 📖 I Want to Modify the Code

### Before you start:
1. Understand what the current code does
   - Read: [IMPLEMENTATION_ARCHITECTURE.md](IMPLEMENTATION_ARCHITECTURE.md)
   
2. Understand what's been fixed
   - Read: [BUG_FIXES_AND_CORRECTIONS.md](BUG_FIXES_AND_CORRECTIONS.md)

3. Know which implementation is primary
   - Use: `src/ga_solver.py` (NOT `genetic_algorithm.py`)
   - Use: `src/baseline.py` for exact/greedy solvers
   - Use: `src/fitness.py` for fitness evaluation

### Common modifications:

**Change GA parameters**:
```python
# src/benchmark_statistical.py, around line 48-54
SMALL_POP = 150      # Increase population
SMALL_GEN = 200      # More generations
SMALL_MUT = 0.3      # Mutation rate (probability per individual)
```

**Add convergence early termination**:
- Already implemented! See `src/ga_solver.py` line ~235
- Edit `STAGNATION_LIMIT = 10` to change sensitivity

**Switch mutation probability**:
- Already adaptive! `src/ga_solver.py` line 74: `indpb=(1.0 / self.num_facilities)`
- No hardcoding

**Test on single instance**:
```python
# Quick test on cap41 instead of all 15
python -c "
from src.parser import CFLPDataset
from src.ga_solver import CFLPGASolver

dataset = CFLPDataset('data/raw/cap41.txt')
ga = CFLPGASolver(dataset)
cost, y, hist = ga.solve(pop_size=120, n_gen=100)
print(f'Cap41 GA cost: {cost:.2f}')
"
```

---

## 🤔 Frequently Asked Questions

**Q: Which solver should I use?**  
A: Depends on your needs:
- Need provably optimal? → MILP (small instances only)
- Need speed? → Greedy
- Need good balance? → GA (recommended, PRIMARY)
- Need to experiment? → Modular GA
- Need speedup via ML? → Hybrid ML-GA (`python src/benchmark_hybrid_ga.py` — bootstraps its own training data and surrogate automatically, no pre-trained model needed)

**Q: How long does a benchmark take?**
- Statistical GA: ~8 minutes (30 runs × 15 instances)
- Large-scale: ~10 minutes (3 solvers × 12 instances)
- Hybrid ML-GA: ~15-20 minutes (bootstrap + train + 10 solve runs × 15 instances)

**Q: What's the difference between ga_solver.py and genetic_algorithm.py?**  
See [IMPLEMENTATION_ARCHITECTURE.md](IMPLEMENTATION_ARCHITECTURE.md) → Module Reference:
- `ga_solver.py`: Primary GA, uses DEAP framework, benchmarked, production-ready
- `genetic_algorithm.py`: Experimental modular GA, not benchmarked, research only

**Q: Can I train my own surrogate model?**  
A: Yes! `python src/training_pipeline.py`  
But note: Surrogate quality directly affects hybrid GA performance

**Q: My results differ from the CSV file — is something wrong?**  
A: No! Random algorithms produce different results. Expected:
- Same seed → same results (reproducible)
- Different seed → different results (randomness)
- Same hardware → same results
- Different hardware → slightly different results (floating point differences)

See [REPRODUCIBILITY_AND_VERIFICATION.md](REPRODUCIBILITY_AND_VERIFICATION.md) for verification tests

---

## 📚 Full Documentation Map

| Document | When to Read | What It Covers |
|----------|----------|----------|
| [README.md](../README.md) | First | Project overview, setup |
| **[QUICK_START.md](QUICK_START.md)** | **→ You are here** | **TL;DR version of everything** |
| [BUG_FIXES_AND_CORRECTIONS.md](BUG_FIXES_AND_CORRECTIONS.md) | Skeptical about results | 6 bugs found & fixed, verification tests |
| [IMPLEMENTATION_ARCHITECTURE.md](IMPLEMENTATION_ARCHITECTURE.md) | Want to understand code | Detailed algorithm explanations, all modules |
| [REPRODUCIBILITY_AND_VERIFICATION.md](REPRODUCIBILITY_AND_VERIFICATION.md) | Want to verify reproducibility | Step-by-step reproduction guide, 5 verification tests |
| [PHASE_4_HYBRID_BENCHMARK_REPORT.md](PHASE_4_HYBRID_BENCHMARK_REPORT.md) | Want full results & honest analysis | Hybrid ML-GA vs. Classical GA on all 15 instances, large-instance MILP table, root-cause writeups |

---

## 🎯 Next Steps

1. **Run a benchmark**: `python src/benchmark_statistical.py`
2. **Read [IMPLEMENTATION_ARCHITECTURE.md](IMPLEMENTATION_ARCHITECTURE.md)** to understand the code
3. **Read [BUG_FIXES_AND_CORRECTIONS.md](BUG_FIXES_AND_CORRECTIONS.md)** to understand what was fixed
4. **Modify and experiment**: Change parameters, test different solvers
5. **Verify your results**: Use tests in [REPRODUCIBILITY_AND_VERIFICATION.md](REPRODUCIBILITY_AND_VERIFICATION.md)

---

## ✨ Key Takeaways

- **This project is research-grade and reproducible**
- **5 complementary solvers**: MILP, Greedy, GA (primary), Modular GA, Hybrid ML-GA
- **All critical bugs have been fixed** (see [BUG_FIXES_AND_CORRECTIONS.md](BUG_FIXES_AND_CORRECTIONS.md))
- **All benchmarks can be independently verified** (see [REPRODUCIBILITY_AND_VERIFICATION.md](REPRODUCIBILITY_AND_VERIFICATION.md))
- **Code is well-documented and modular** (see [IMPLEMENTATION_ARCHITECTURE.md](IMPLEMENTATION_ARCHITECTURE.md))

**Questions?** Check the full documentation guides above.

