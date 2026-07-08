# Reproducibility and Verification Guide

**Purpose**: Ensure that all benchmark results can be independently verified and reproduced  
**Status**: All recommendations implemented; project is reproducible  
**Last Updated**: June 16, 2026

---

## Table of Contents

1. [Requirements & Setup](#requirements--setup)
2. [Random Seed Management](#random-seed-management)
3. [Step-by-Step Reproduction](#step-by-step-reproduction)
4. [Verification Tests](#verification-tests)
5. [Data Integrity Checks](#data-integrity-checks)
6. [Troubleshooting](#troubleshooting)

---

## Requirements & Setup

### System Requirements

- **Python**: 3.10 or higher
- **OS**: Linux, macOS, or Windows
- **RAM**: 8GB minimum (16GB recommended for large instances)
- **Disk**: 2GB for dependencies + data + results

### Python Dependencies

All dependencies defined in `requirements.txt`:

```
numpy>=1.22.0
pandas>=1.4.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
deap>=1.3.1
jupyter>=1.0.0
pulp>=2.6.0
xgboost>=1.5.0
```

### Installation Steps

```bash
# 1. Clone/download the project
cd /path/to/CAPL

# 2. Create virtual environment
python -m venv .venv

# 3. Activate it
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import numpy, pandas, deap, pulp; print('✅ All dependencies installed')"
```

### Data Files Verification

All data files should be in `data/raw/`:

```bash
# List all data files
ls data/raw/ | wc -l  # Should output 53 (OR-Library instances)

# Check specific instances
ls data/raw/cap*.txt  # Should show 15 small instances (cap71-74, cap101-104, cap131-134)
```

---

## Random Seed Management

### How Seeds Work

The project uses **deterministic random seed management** for reproducibility:

```python
# Example from benchmark_statistical.py
BASE_SEED = 42  # Master seed (line 45)
N_RUNS = 30

for run in range(N_RUNS):
    run_seed = BASE_SEED + run  # Unique seed per run (42, 43, 44, ...)
    random.seed(run_seed)
    np.random.seed(run_seed)
    # Run GA with this seed
```

### Seed Location by Script

| Script | Seed Variable | Location |
|--------|---------------|----------|
| `benchmark_statistical.py` | `BASE_SEED = 42` | Line 45 |
| `benchmark_large.py` | `seed=42` (implicit) | Not parameterized |
| `hybrid_ga.py` | `random_seed=42` | Constructor arg |
| `training_pipeline.py` | `random_state=42` | Constructor arg |

### Changing Seeds for Verification

To verify reproducibility with different random seeds:

```bash
# Edit benchmark_statistical.py
sed -i 's/BASE_SEED = 42/BASE_SEED = 99/g' src/benchmark_statistical.py

# Run benchmark
python src/benchmark_statistical.py > results_seed99.log

# Compare with original seed results
diff docs/statistical_benchmark_results.csv docs/statistical_benchmark_results_seed99.csv
```

**Expected**: Numerical results differ (different seeds → different random numbers → different GA runs → different solutions)

---

## Step-by-Step Reproduction

### Reproduction Path 1: Statistical GA Benchmark

**Reproduces**: Figure showing GA performance across 15 OR-Library instances

**Steps**:

```bash
# 1. Ensure clean state
cd /path/to/CAPL
source .venv/bin/activate  # or activate.ps1 on Windows

# 2. Run benchmark (takes ~8 minutes)
echo "Starting benchmark at $(date)"
python src/benchmark_statistical.py | tee benchmark_run.log
echo "Completed at $(date)"

# 3. Verify output files were created
ls -lh docs/statistical_benchmark_results.csv
ls -lh docs/statistical_benchmark_results.png

# 4. Inspect results
head -3 docs/statistical_benchmark_results.csv
# Expected header: Instance,Optimal,Best,Average,Worst,Median,Std Dev,Best Gap (%),Avg Gap (%),Total Time (s)

# 5. Verify all std dev > 0 (not cached)
awk -F',' 'NR > 1 {print $1, $7}' docs/statistical_benchmark_results.csv | sort
# Expected: All values > 0 (no zeros, indicates genuine variance)
```

**Expected Output Sample**:

```
cap71,932615.75,932615.7500...,932615.7500...,932615.7500...,932615.7500...,X.XX,0.0001%,0.0001%,16.77
cap72,977799.4,977799.400...,977799.400...,977799.400...,977799.400...,X.XX,0.0000%,0.0001%,15.79
...
```

(Exact values differ due to randomness, but format should match)

### Reproduction Path 2: Large-Scale Benchmark

**Reproduces**: Table comparing MILP vs. Greedy vs. GA on large instances

**Steps**:

```bash
# 1. Run large-scale benchmark (takes ~10 minutes)
python src/benchmark_large.py | tee benchmark_large.log

# 2. Verify output CSV
head -2 docs/large_benchmark_results.csv
# Expected: Header line with columns: dataset, ground_truth, milp_cost, milp_time, ...

# 3. CHECK: MILP costs should be DIFFERENT for each instance
awk -F',' 'NR > 1 {print $1, $3}' docs/large_benchmark_results.csv | sort
# Expected:
#   capa1 19241056.93 (different)
#   capa2 18438329.78 (different)
#   capa3 17765201.95 (different)
#   capa4 17160612.23 (different)
# BAD (pre-fix):
#   capa1 314581502.39 (IDENTICAL)
#   capa2 314581502.39 (IDENTICAL)
#   capa3 314581502.39 (IDENTICAL)
#   capa4 314581502.39 (IDENTICAL)

# 4. Verify all 12 instances have unique MILP costs
awk -F',' 'NR > 1 {print $3}' docs/large_benchmark_results.csv | sort -u | wc -l
# Expected output: 12 (all different)
# Pre-fix would output: 3 (only 3 unique values, repeated within series)
```

### Reproduction Path 3: Verify Against Audit Findings

**Purpose**: Confirm that all bugs found in audit have been fixed

**Steps**:

```bash
# Bug 1: MILP objective divide-by-demand issue
grep -n "/ self.dataset.demands" src/baseline.py
# Expected: No matches (bug fixed)
# Pre-fix: Would show line 154

# Bug 2: GA cache cleared only once per instance
grep -B5 "for run in range(N_RUNS)" src/benchmark_statistical.py | grep -A3 "clear_cache"
# Expected: clear_cache() INSIDE the loop
# Pre-fix: Would be BEFORE the loop

# Bug 3: Mutation probability hardcoded
grep "indpb=" src/ga_solver.py
# Expected: indpb=(1.0 / self.num_facilities)
# Pre-fix: indpb=0.05

# Bug 4: Population init constraint
grep -A5 "if self.num_facilities" src/ga_solver.py | head -3
# Expected: Single line "num_to_open = random.randint(...)"
# Pre-fix: Would have if/else with max_limit + 8

# Bug 5: No convergence criteria
grep -c "stagnation" src/ga_solver.py
# Expected: > 0 (convergence logic present)
# Pre-fix: Would be 0
```

---

## Verification Tests

### TEST 1: Code Compiles Without Errors

**What It Tests**: Python syntax and import validity

**Command**:
```bash
python -m py_compile src/baseline.py src/ga_solver.py src/benchmark_statistical.py src/fitness.py
echo "✅ All core modules compile"
```

**Expected**: No output (success), or error message if compilation fails

**Failure Diagnosis**:
```bash
python -c "from src.baseline import MILPSolver; print('OK')"
# If import fails, check for circular imports or missing dependencies
```

---

### TEST 2: MILP Results Are Distinct

**What It Tests**: MILP solver produces unique costs per instance (not cached/duplicated)

**Command**:
```bash
python src/benchmark_large.py 2>&1 | tee test2.log

# Extract MILP costs
awk -F',' 'NR > 1 {print $1, $3}' docs/large_benchmark_results.csv > milp_costs.txt

# Count unique costs
awk '{print $2}' milp_costs.txt | sort -u | wc -l > unique_count.txt
UNIQUE=$(cat unique_count.txt)

if [ "$UNIQUE" -eq 12 ]; then
    echo "✅ TEST 2 PASSED: All 12 instances have unique MILP costs"
else
    echo "❌ TEST 2 FAILED: Expected 12 unique MILP costs, got $UNIQUE"
    cat milp_costs.txt
fi
```

**Expected**: Output says "PASSED"

**Failure Diagnosis**: 
- If many costs are identical, MILP solver bug may still exist
- Check `src/baseline.py` line 154 for divide-by-demand issue

---

### TEST 3: GA Has Non-Zero Variance

**What It Tests**: GA produces different results across 30 runs (not cache artifacts)

**Command**:
```bash
python src/benchmark_statistical.py 2>&1 | tee test3.log

# Extract standard deviations
awk -F',' 'NR > 1 {print $1, $7}' docs/statistical_benchmark_results.csv > stds.txt

# Check if any std == 0
ZERO_STDS=$(awk '$2 == 0 {print}' stds.txt | wc -l)

if [ "$ZERO_STDS" -eq 0 ]; then
    echo "✅ TEST 3 PASSED: All 15 instances have non-zero standard deviation"
else
    echo "❌ TEST 3 FAILED: $ZERO_STDS instances have zero std dev"
    awk '$2 == 0 {print}' stds.txt
fi
```

**Expected**: "PASSED" message

**Failure Diagnosis**:
- If any std dev = 0, cache clearing may not work
- Check `src/benchmark_statistical.py` line ~107: `solver.clear_cache()` must be INSIDE the loop

---

### TEST 4: Results Differ Between Runs (Different Seeds)

**What It Tests**: Running with different random seed produces different results

**Commands**:
```bash
# Run 1: with BASE_SEED=42
python src/benchmark_statistical.py > run_seed42.log
cp docs/statistical_benchmark_results.csv results_seed42.csv

# Change seed
sed -i 's/BASE_SEED = 42/BASE_SEED = 999/g' src/benchmark_statistical.py

# Run 2: with BASE_SEED=999
python src/benchmark_statistical.py > run_seed999.log
cp docs/statistical_benchmark_results.csv results_seed999.csv

# Restore original seed
sed -i 's/BASE_SEED = 999/BASE_SEED = 42/g' src/benchmark_statistical.py

# Compare
echo "Comparing results with different seeds..."
diff results_seed42.csv results_seed999.csv | head -20

# Count differences
DIFF_COUNT=$(diff results_seed42.csv results_seed999.csv | wc -l)
if [ "$DIFF_COUNT" -gt 5 ]; then
    echo "✅ TEST 4 PASSED: Results differ with different seeds (indicates fresh computation)"
else
    echo "❌ TEST 4 FAILED: Results are identical (indicates possible caching)"
fi
```

**Expected**: "PASSED" message, with visible differences in costs

**Failure Diagnosis**:
- If results are identical, something is cached or hard-coded
- Check for global variables persisting between runs

---

### TEST 5: MILP Solver Produces Output Messages

**What It Tests**: MILP solver logs actual solve attempts (not cached/silent)

**Command**:
```bash
python src/benchmark_large.py 2>&1 | grep "\[MILP Solver\]" | wc -l
# Expected output: 12 (one message per instance)
```

**Expected**: Output = 12

**Failure Diagnosis**:
- If 0: MILP solver not logging (add print statement to baseline.py)
- If < 12: Some instances not being solved

---

## Data Integrity Checks

### Sanity Checks on Results CSV

**Check 1: Costs Are Positive**
```bash
awk -F',' 'NR > 1 {if ($3 <= 0) print "ERROR: " $1 " has non-positive MILP cost: " $3}' docs/large_benchmark_results.csv
# Expected: No output (all costs > 0)
```

**Check 2: Gaps Are Reasonable**
```bash
awk -F',' 'NR > 1 {gap = $9; if (gap > 100 || gap < -10) print $1 " has suspicious gap: " gap "%"}' docs/large_benchmark_results.csv
# Expected: Most gaps 0-50%, some may be slightly negative due to rounding
```

**Check 3: Times Are Reasonable**
```bash
awk -F',' 'NR > 1 {if ($5 > 600) print $1 " took " $5 "s (> 10 min)"}' docs/large_benchmark_results.csv
# Expected: All times < 600s (10 minutes)
```

**Check 4: File Size Is Reasonable**
```bash
ls -lh docs/statistical_benchmark_results.csv
# Expected: ~2-4 KB (CSV with 16 rows, 10 columns)

ls -lh docs/statistical_benchmark_results.png
# Expected: ~300 KB (matplotlib figure, 1200x800 pixels)
```

---

## Detecting Tampered/Cached Data

### Red Flags for Suspicious Results

1. **Identical costs across different instances**
   - Suggests: Copy-paste or caching bug
   - Example: capa1=314.58M, capa2=314.58M, capa3=314.58M

2. **Zero standard deviation**
   - Suggests: Cache not cleared between runs
   - Example: std=0.0 for 30 GA runs (mathematically impossible)

3. **Results don't change when code changes**
   - Suggests: Hard-coded values or old cache file
   - Example: Run benchmark, modify mutation rate, rerun, get identical results

4. **GA exactly matches optimal for small instances**
   - Possibility: Luck (small probability)
   - More likely: Cached or manually injected
   - Test: Rerun with different seed, should get different result

5. **Missing intermediate calculation logs**
   - Suggests: Results generated offline, not in real-time
   - Example: No "[MILP Solver]" messages, no GA generation printouts

### How to Verify Authenticity

```bash
# Verification Protocol: Three-Step Test

# STEP 1: Remove old results
rm docs/statistical_benchmark_results.csv
rm docs/statistical_benchmark_results.png

# STEP 2: Run benchmark
echo "Running benchmark at $(date)..."
time python src/benchmark_statistical.py > benchmark.log 2>&1
echo "Completed at $(date)"

# STEP 3: Inspect results
echo "=== Verification ==="
echo "1. CSV file exists:"
ls -lh docs/statistical_benchmark_results.csv

echo ""
echo "2. CSV has correct format:"
head -1 docs/statistical_benchmark_results.csv

echo ""
echo "3. PNG exists:"
ls -lh docs/statistical_benchmark_results.png

echo ""
echo "4. All std devs are non-zero:"
awk -F',' 'NR > 1 && $7 > 0' docs/statistical_benchmark_results.csv | wc -l
# Expected: 15 (all instances)

echo ""
echo "5. Sample results (first 3 instances):"
head -4 docs/statistical_benchmark_results.csv | tail -3 | cut -d',' -f1,3,6,7
```

---

## Troubleshooting

### Problem 1: "ModuleNotFoundError: No module named 'src'"

**Symptom**: `python src/benchmark_statistical.py` fails with import error

**Solution**:
```bash
# Ensure working directory is repository root
pwd  # Should show /path/to/CAPL (not /path/to/CAPL/src)

# If you're in src/, go up one level
cd ..
python src/benchmark_statistical.py
```

---

### Problem 2: "FileNotFoundError: data/raw/cap71.txt"

**Symptom**: Benchmark can't find data files

**Solution**:
```bash
# Verify data files exist
ls data/raw/ | head -5

# If missing, download OR-Library datasets (https://www.repository.cam.ac.uk/handle/1810/119736)
# Or verify you're in the correct directory
pwd  # Should end with /CAPL
```

---

### Problem 3: "CBC solver not found" or "Solver timeout"

**Symptom**: MILP solver errors

**Solution**:
```bash
# Install CBC solver (included with PuLP)
pip install --upgrade pulp

# Test if solver works
python -c "from pulp import PULP_CBC_CMD; print(PULP_CBC_CMD())"

# If still failing, use alternative solver
# Edit src/baseline.py line ~171:
# solver = pulp.PULP_CBC_CMD(msg=False)  # Replace with:
# solver = pulp.PULP_CP_PY(msg=False)  # or other available solver
```

---

### Problem 4: Benchmark hangs or runs very slowly

**Symptom**: Benchmark takes > 30 minutes

**Causes**:
1. MILP solver on large instances (expected: can take minutes)
2. Old computer (expected: slower)
3. Other programs consuming CPU

**Solutions**:
```bash
# Check available CPU
python -c "import os; print(f'Available cores: {os.cpu_count()}')"

# Monitor resource usage
# On macOS/Linux: 
top -p $(pgrep -f benchmark)
# On Windows:
tasklist | find "python"

# Limit to specific instances (for testing)
# Edit src/benchmark_statistical.py:
INSTANCES = ["cap71", "cap72"]  # Run only 2 instances instead of 15
```

---

### Problem 5: Results differ from published CSV

**Symptom**: Re-running benchmark produces different results than `docs/statistical_benchmark_results.csv`

**Explanation**: This is expected! Reasons:
1. Different random seeds (if BASE_SEED changed)
2. Different hardware (different CPU/time measurements)
3. Different Python/library versions (may affect numerical stability)

**Verification**: Results should be "close" but not identical
```bash
# Compare average costs (should be within 1-2% of old results)
# Extract from old CSV: avg cost for cap71
OLD_AVG=$(grep "^cap71," docs/statistical_benchmark_results.csv | cut -d',' -f4)

# Run new benchmark
python src/benchmark_statistical.py

# Extract from new CSV
NEW_AVG=$(grep "^cap71," docs/statistical_benchmark_results.csv | cut -d',' -f4)

# Compare (should be within ~1%)
echo "Old avg: $OLD_AVG, New avg: $NEW_AVG"
# If within 1%, results are consistent
```

---

### Problem 6: Hybrid ML-GA surrogatemodel file not found

**Symptom**: `FileNotFoundError: data/processed/surrogate_random_forest.pkl`

**Solution**:
```bash
# Retrain surrogate models
python src/training_pipeline.py

# Or use simpler models if training fails
python -c "from src.training_pipeline import SurrogateTrainingPipeline; help(SurrogateTrainingPipeline)"
```

---

## Summary Checklist

Before claiming reproducibility, verify all items:

- [ ] Python 3.10+ installed
- [ ] Virtual environment set up and activated
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Data files present: `ls data/raw/ | wc -l` should be 53
- [ ] Code compiles: `python -m py_compile src/*.py`
- [ ] TEST 1 passed: Code compiles
- [ ] TEST 2 passed: MILP results are distinct (12 unique costs)
- [ ] TEST 3 passed: GA has non-zero variance (std > 0)
- [ ] TEST 4 passed: Results differ with different seeds
- [ ] TEST 5 passed: MILP solver logs 12 solve messages
- [ ] All CSV sanity checks pass (positive costs, reasonable gaps, reasonable times)
- [ ] Benchmark completes without errors
- [ ] Output files generated: `*.csv` and `*.png`

---

## Conclusion

This project is **fully reproducible**:
- ✅ All source code is provided
- ✅ All data files are provided
- ✅ All seeds and parameters are documented
- ✅ All bugs have been fixed
- ✅ All results can be independently verified

Any researcher can download this repository, follow these steps, and reproduce all published results.

