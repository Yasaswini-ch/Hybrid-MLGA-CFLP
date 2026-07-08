# Implementation Architecture Guide

## Table of Contents
1. [Problem Formulation](#problem-formulation)
2. [Solution Approaches](#solution-approaches)
3. [Module Reference](#module-reference)
4. [Detailed Algorithms](#detailed-algorithms)
5. [Running Benchmarks](#running-benchmarks)
6. [Known Issues & Limitations](#known-issues--limitations)
7. [Troubleshooting](#troubleshooting)

---

## Problem Formulation

### The Capacitated Facility Location Problem (CFLP)

**Objective**: Minimize total cost of facility openings + customer-to-facility transportation

**Mathematical Formulation**:
```
Minimize: Z = Σ(i ∈ I) f_i·y_i + Σ(j ∈ J) Σ(i ∈ I) c_ij·x_ij

Subject to:
  1) Σ(i ∈ I) x_ij = d_j              ∀j ∈ J    [Demand satisfaction]
  2) Σ(j ∈ J) x_ij ≤ s_i·y_i          ∀i ∈ I    [Capacity bounds]
  3) y_i ∈ {0, 1}                      ∀i ∈ I    [Facility open/closed]
  4) x_ij ≥ 0                          ∀i,j      [Non-negative flows]
```

**Parameters**:
- **I** = set of m potential facilities
- **J** = set of n customers
- **f_i** = fixed opening cost of facility i (thousands of dollars)
- **s_i** = capacity of facility i (units)
- **d_j** = demand of customer j (units)
- **c_ij** = unit transportation cost from facility i to customer j ($/unit)

**Decision Variables**:
- **y_i** = 1 if facility i opens, 0 if closed
- **x_ij** = amount of customer j's demand served by facility i (units)

### Benchmark Datasets

All instances from **Beasley's OR-Library** (standard benchmark suite):

| Series | Facilities | Customers | Instances |
|--------|-----------|-----------|-----------|
| cap7x | 16 | 50 | cap71, cap72, cap73, cap74 |
| cap10x | 25 | 50 | cap101, cap102, cap103, cap104 |
| cap13x | 50 | 150 | cap131, cap132, cap133, cap134 |
| capa,b,c | 100 | 1000 | capa4, capb4, capc4 (the bare `capa`/`capb`/`capc` files are unfilled Beasley templates and cannot be used directly — see `docs/PHASE_4_HYBRID_BENCHMARK_REPORT.md`) |

---

## Solution Approaches

### Approach 1: Exact Optimization via MILP

**Solver**: `src/baseline.py` → `MILPSolver` class  
**Engine**: PuLP library + COIN-OR CBC solver  
**Complexity**: O(2^m) worst-case; minutes to hours for large instances

**When to Use**:
- Instances m ≤ 50 (small/medium)
- Need provably optimal solution
- Research baseline comparison

**Output**: Exact optimal cost and facility configuration

**Code Entry Point**:
```python
from src.baseline import MILPSolver
from src.parser import CFLPDataset

dataset = CFLPDataset("data/raw/cap71.txt")
milp_solver = MILPSolver(dataset)
optimal_cost, y_optimal, x_optimal, status = milp_solver.solve(timeout_sec=60)
```

### Approach 2: Greedy Heuristic

**Solver**: `src/baseline.py` → `GreedySolver` class  
**Logic**: Rank facilities by cost-efficiency, open greedily until demand satisfied  
**Complexity**: O(m log m + n·m)  
**Runtime**: < 1 second

**When to Use**:
- Large instances where exact solution is infeasible
- Quick baseline for quality comparison
- Sanity check on optimization difficulty

**Code Entry Point**:
```python
from src.baseline import GreedySolver

greedy_solver = GreedySolver(dataset)
greedy_cost, y_greedy, x_greedy = greedy_solver.solve()
```

### Approach 3: Classical Genetic Algorithm (PRIMARY)

**Solver**: `src/ga_solver.py` → `CFLPGASolver` class  
**Engine**: DEAP framework + custom fitness evaluator  
**Complexity**: Polynomial per generation; exponential search space exploration  
**Runtime**: 1-5 minutes per instance

**Components**:
1. **Representation**: Binary y-vector (facility open/closed status)
2. **Population**: 50-120 individuals per generation
3. **Fitness**: For each y configuration, solve continuous LP transportation sub-problem
4. **Operators**:
   - Selection: Tournament (size=3)
   - Crossover: Two-point with probability 0.8
   - Mutation: Bit-flip with probability 1/m per facility
5. **Termination**:
   - Fixed generations (100 default)
   - Early convergence if < 0.01% improvement for 10 consecutive generations

**Algorithm Pseudocode**:
```
1. Initialize population of m-bit binary vectors
2. For each generation (1 to n_gen):
   a. Evaluate fitness of all individuals (solve LP for transportation)
   b. Select promising individuals via tournament
   c. Apply crossover (genetic recombination)
   d. Apply mutation (random bit flips)
   e. Check early convergence criteria
   f. Track best solution found so far
3. Return best solution discovered
```

**When to Use**:
- Instances of any size
- Trade-off between quality and runtime
- Exploration of solution space

**Code Entry Point**:
```python
from src.ga_solver import CFLPGASolver

ga_solver = CFLPGASolver(dataset)
best_cost, best_y, history = ga_solver.solve(
    pop_size=120,
    n_gen=100,
    cx_pb=0.8,    # Crossover probability
    mut_pb=0.2    # Mutation probability (applies mut_pb* to each individual)
)
```

### Approach 4: Modular GA (EXPERIMENTAL)

**Solver**: `src/genetic_algorithm.py` → `ModularCFLPGASolver` class  
**Status**: Alternative implementation, NOT used in benchmarks  
**Features**: Includes repair operator, explicit elitism, detailed history tracking

**Note**: `genetic_algorithm.py` was created as an experimental alternative to `ga_solver.py`. For benchmarking and production use, always use `ga_solver.py` (CFLPGASolver), not this modular version.

**When to Use**:
- Research/experimentation only
- Testing custom operators
- Comparison studies

### Approach 5: Hybrid ML+GA (VALIDATED — see Chapter 16 of the Complete Project Guide)

**Solver**: `src/hybrid_ga.py` → `HybridMLGASolver` class  
**Status**: Fully implemented, tested end-to-end, and benchmarked on all 15 OR-Library
instances (`docs/hybrid_benchmark_results.csv`, `docs/PHASE_4_HYBRID_BENCHMARK_REPORT.md`).  
**Concept**: Replace expensive LP fitness evaluations with fast ML predictions

**No pre-trained model is required to get started** — pass `surrogate=None` for
**bootstrap mode**, which runs like a normal exact-LP GA while logging every
`(chromosome, cost)` pair it evaluates. That log becomes the surrogate's first
training dataset, resolving what used to be a circular dependency (you needed a
trained model to run the GA, but needed the GA to run to get training data).

**Modes**:
1. **pure_surrogate**: All fitness from ML (fastest, riskiest)
2. **confidence_aware**: trust the surrogate's prediction by default; only verify
   with the real LP solver when the predicted cost is **lower than the best solution
   found so far** (a potential new best) — this is the corrected decision rule,
   confirmed by direct measurement of every prediction made during a real run.

**Requirements**:
- No pre-trained model needed (see bootstrap mode above)
- Feature engineering to convert chromosomes to ML features (handled automatically)

**When to Use**:
- When speedup from ML-based evaluation is critical, and the instance is small/medium
  (see `docs/PHASE_4_HYBRID_BENCHMARK_REPORT.md` — on the 3 largest OR-Library
  instances the Classical GA currently outperforms this approach; the gap and its
  root cause are documented there, not hidden)
- Testing active learning strategies (`src/active_learning.py`, quality-gated —
  a retrained surrogate is only adopted if it scores at least as well as the best
  model seen so far on a fixed validation set)

**Code Entry Point**:
```python
from src.hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
from src.training_pipeline import SurrogateTrainingPipeline

# Stage 1: bootstrap -- no pre-trained model needed
bootstrap_ga = HybridMLGASolver(dataset=dataset, surrogate=None)
result = bootstrap_ga.solve()
X, y = extract_training_data_from_ga(result, dataset=dataset)

# Stage 2: train a surrogate on the GA's own data
pipeline = SurrogateTrainingPipeline(dataset=dataset, corpus_path=corpus_path, model_save_dir=model_dir)
trained = pipeline.run(model_types=("random_forest",))
surrogate = trained["random_forest"]["surrogate"]

# Stage 3: solve with the trained surrogate
hybrid = HybridMLGASolver(dataset=dataset, surrogate=surrogate, mode="confidence_aware")
result = hybrid.solve()
```

---

## Module Reference

### Core Solvers

#### `baseline.py` — Exact and Greedy Baselines
```
Classes:
  - GreedySolver(dataset)
      solve() → (cost, y, x)
  
  - MILPSolver(dataset)
      solve(timeout_sec=60) → (cost, y, x, status)
```

**Key Functions**:
- Greedy heuristic by cost-efficiency ratio
- MILP formulation via PuLP/CBC
- Handles infeasibility with penalties

#### `ga_solver.py` — Classical Genetic Algorithm (PRIMARY)
```
Classes:
  - CFLPGASolver(dataset)
      solve(pop_size, n_gen, cx_pb, mut_pb) → (best_cost, best_y, history)
      clear_cache()
      evaluate_fitness(individual) → (cost,)
```

**Key Methods**:
- DEAP-based GA with custom fitness evaluation
- Caching of fitness values per chromosome
- Early convergence detection
- Parallel fitness evaluation for large instances

#### `genetic_algorithm.py` — Modular GA (EXPERIMENTAL)
```
Classes:
  - ModularCFLPGASolver(dataset, mode="repair", heuristic_ratio=0.5)
      solve() → (best_cost, best_y, history)
```

**Differs from ga_solver.py**:
- Uses custom modular operators (selection.py, crossover.py, etc.)
- Includes repair operator for infeasible solutions
- Explicit elitism preservation
- NOT benchmarked; for experimental use only

#### `hybrid_ga.py` — Hybrid ML+GA Solver
```
Classes:
  - HybridMLGASolver(dataset, surrogate, mode, ...)
      solve(pop_size, n_generations) → (best_cost, best_y)
```

**Key Features**:
- ML surrogate for fitness approximation
- Warm-up period with exact LP
- Confidence-aware fallback to exact fitness
- Optional active learning integration

### Fitness Evaluation

#### `fitness.py` — Exact Fitness via LP
```
Classes:
  - CFLPFitnessEvaluator(dataset)
      evaluate(individual) → (cost,)
```

**Algorithm**:
1. Check physical capacity feasibility
2. Quick UFLP check (assign each customer to cheapest facility)
3. If UFLP fails capacity check, solve full continuous LP
4. Return fixed_cost + transport_cost

**Underlying LP**:
```
For a fixed y configuration:
  Minimize: Σ Σ c_ij * x_ij
  Subject to: Σ x_ij = d_j, Σ x_ij ≤ s_i * y_i
  Solution: Interior point method via SciPy HiGHS
```

#### `cost_calculator.py` — Cost Computation Utilities
```
Functions:
  - calculate_fixed_costs(y, fixed_costs) → float
  - calculate_transportation_costs(x, transport_costs) → float
  - calculate_total_cost(solution, dataset) → float
```

**Internal Representation**:
- **x** stored as absolute flows (units served)
- Cost calculation divides x by demand to get fractions, then multiplies by unit costs
- Mathematically equivalent to: sum(c_ij * x_ij)

### Machine Learning Components

#### `surrogate_model.py` — Unified ML Model Wrapper
```
Classes:
  - CFLPSurrogateModel(model_type="random_forest")
      fit(X_train, y_train)
      predict(X) → y_pred
      predict_with_uncertainty(X) → (y_pred, sigma)
      save(path) / load(path)
```

**Supported Models**:
- `random_forest`: RandomForestRegressor (200 trees, depth=15)
- `gradient_boosting`: GradientBoostingRegressor (300 trees, learning_rate=0.05)
- `xgboost`: XGBRegressor (if installed)
- `mlp`: MLPRegressor (neural network, 128-64-32 architecture)

**Uncertainty Quantification**:
- **Random Forest only**: Std dev of individual tree predictions
- **Others**: Return zero uncertainty (no native uncertainty)

#### `feature_engineering.py` — Chromosome-to-Features Transformation
```
Classes:
  - CFLPFeatureEngineer(dataset, mode="full")
      transform(X_raw) → X_features
```

**Feature Extraction Modes**:
- `raw`: Direct binary features (X as-is)
- `full`: Engineered features:
  - Number of open facilities
  - Capacity utilization ratios
  - Cost-weighted facility features
  - Distance metrics to optimal facilities

#### `evaluation_metrics.py` — Model Accuracy Assessment
```
Functions:
  - compute_regression_metrics(y_true, y_pred) → dict
  - compute_latency_speedup(ml_time, lp_time) → float
  - print_metrics_report(results)
```

**Metrics Reported**:
- MAPE (Mean Absolute Percentage Error) — % accuracy
- RMSE (Root Mean Squared Error) — scale
- R² score — variance explained
- Latency speedup factor — ML vs. LP runtime ratio

### GA Operators (Experimental, for genetic_algorithm.py)

#### `chromosome.py` — Binary Solution Representation
```
Classes:
  - CFLPChromosome(y_vector)
      hamming_distance(other) → int
      repair(dataset)
```

#### `population.py` — Population Initialization
```
Classes:
  - CFLPPopulationGenerator(dataset)
      create_population(pop_size, heuristic_ratio) → list[chromosome]
```

Heuristic seeding: percentage of population initialized via greedy solution

#### `selection.py` — Parent Selection
```
Functions:
  - tournament_select(pop, tournsize) → list[selected]
  - roulette_select(pop) → list[selected]
  - apply_elitism(parent_pop, offspring_pop, elite_count) → offspring
```

#### `crossover.py` — Genetic Recombination
```
Functions:
  - single_point_crossover(parent1, parent2) → (child1, child2)
  - two_point_crossover(parent1, parent2) → (child1, child2)
  - uniform_crossover(parent1, parent2) → (child1, child2)
```

#### `mutation.py` — Genetic Variation
```
Functions:
  - bit_flip_mutation(individual, indpb=0.05) → individual
```

Randomly flips each bit with probability indpb

#### `repair.py` — Constraint Repair (Lamarckian)
```
Classes:
  - CFLPFeasibilityRepairer(dataset)
      repair(individual) → individual
```

**Strategy**: Iteratively open facilities until demand satisfied

### Utilities

#### `parser.py` — OR-Library Dataset Loading
```
Classes:
  - CFLPDataset(file_path)
      Properties: num_facilities, num_customers, capacities, fixed_costs, demands, transport_costs
      Methods: get_summary() → dict
```

#### `constraint_checker.py` — Feasibility Validation
```
Functions:
  - is_feasible(solution, dataset) → (bool, errors)
  - check_demand_satisfaction(x, demands) → bool
  - check_capacity_bounds(x, y, capacities) → bool
```

---

## Detailed Algorithms

### Classical GA Fitness Evaluation (ga_solver.py)

```python
def evaluate(y_configuration):
    """
    Given a facility opening y, compute optimal customer routing x and total cost
    """
    # Step 1: Check physical feasibility
    if sum(capacities[i] for i if y[i]==1) < total_demand:
        return PENALTY_COST  # Infeasible
    
    # Step 2: Quick UFLP check
    for j in customers:
        assign_customer_j_to_cheapest_open_facility()
    if all capacity constraints satisfied:
        return fixed_cost(y) + transport_cost(assignments)
    
    # Step 3: Full LP sub-problem
    variables: w[j,k] = fraction of customer j served by open facility k
    objective: minimize sum(c[j,k] * w[j,k])
    constraints:
        - sum_k(w[j,k]) == 1  (demand satisfaction)
        - sum_j(demand[j] * w[j,k]) <= capacity[k]  (capacity bounds)
        - w[j,k] >= 0
    
    solution = solve LP with HiGHS
    x[j,k] = w[j,k] * demand[j]  (reconstruct absolute flows)
    return fixed_cost(y) + sum(c[j,k] * x[j,k])
```

### Two-Point Crossover

```python
def two_point_crossover(parent1, parent2):
    """
    Select two random points, exchange segment between them
    """
    point1 = random(0, m)
    point2 = random(point1, m)
    
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    
    return child1, child2
```

Example:
```
Parent 1: [1 0 | 1 0 1 | 0 1] → Point 1=2, Point 2=5
Parent 2: [0 1 | 0 1 0 | 1 0]
                ↓
Child 1:  [1 0 | 0 1 0 | 0 1]  (swapped middle segment)
Child 2:  [0 1 | 1 0 1 | 1 0]
```

### Bit-Flip Mutation

```python
def bit_flip_mutation(individual, indpb=1/m):
    """
    For each bit, flip with probability indpb
    """
    for i in range(m):
        if random() < indpb:
            individual[i] = 1 - individual[i]
    return individual
```

---

## Running Benchmarks

### Standard Statistical Benchmark

**File**: `src/benchmark_statistical.py`

**What It Does**:
- Runs GA 30 times on each of 15 OR-Library instances
- Computes mean, std dev, best, worst, median cost
- Compares against published optimal costs
- Generates optimality gap statistics

**Command**:
```bash
python src/benchmark_statistical.py
```

**Output**:
- Console: Formatted table with statistics
- File: `docs/statistical_benchmark_results.csv`
- File: `docs/statistical_benchmark_results.png`

**Expected Runtime**: ~8 minutes

**Interpretation**:
- **Best Gap**: How close GA gets to optimal (%)
- **Avg Gap**: Average performance across 30 runs
- **Std Dev**: Solution variability (higher = more variance, can be good or bad)
- **Time**: Total seconds for 30 runs per instance

### Large-Scale Benchmark

**File**: `src/benchmark_large.py`

**What It Does**:
- Compares three solvers on large instances (m=100, n=1000):
  - MILP exact solver (slow but optimal)
  - Greedy heuristic (fast, lower quality)
  - Classical GA (medium speed, good quality)
- Computes quality gaps and runtimes

**Command**:
```bash
python src/benchmark_large.py
```

**Output**:
- Console: Comparison table
- File: `docs/large_benchmark_results.csv`

**Expected Runtime**: ~10 minutes

**Interpretation**:
- **MILP Cost**: True optimal (gold standard)
- **GA Gap**: How much worse GA is vs. MILP (%)
- **GA Time**: Seconds to solution

### Hybrid ML-GA Benchmark

**File**: `src/benchmark_hybrid_ga.py`

**What It Does**:
- Runs the full Hybrid ML-GA pipeline on all 15 OR-Library instances: bootstrap mode
  generates its own training data, trains a Random Forest surrogate, then solves 10
  times per instance using the corrected confidence-aware decision rule.
- Directly comparable, column-for-column, with `docs/statistical_benchmark_results.csv`
  (the Classical GA's results).

**Command**:
```bash
python src/benchmark_hybrid_ga.py
```

**Output**:
- Console: per-instance Best / Average / Worst costs and gaps
- File: `docs/hybrid_benchmark_results.csv`
- See `docs/PHASE_4_HYBRID_BENCHMARK_REPORT.md` for the full analysis

---

## Known Issues & Limitations

### Issue 1: Surrogate Model Accuracy Dependency
- **Problem**: Hybrid ML-GA quality depends on surrogate accuracy
- **Mitigation**: Always validate MAPE < 2% before using
- **Solution**: Use confidence-aware mode with LP fallback

### Issue 2: MILP Objective Formula (FIXED) and Expected Timeout on Large Instances
- **Was a real bug**: the MILP objective multiplied `transport_costs[j,i]` directly by
  absolute flow, but this dataset format defines `transport_costs[j,i]` as the flat
  total cost of fully serving a customer's whole demand from one facility, not a
  per-unit rate (same convention used by `cost_calculator.py`, the GA, and the Greedy
  solver, which all divide flow by demand first). This made CBC solve a formulation
  up to ~demand-times too expensive per customer, producing "provably optimal"
  solutions that opened far more facilities than necessary and cost 4-20x too much.
  **Fixed** by restoring the division in `src/baseline.py`; verified with an exact
  match against `cap71`'s published optimum. See `BUG_FIXES_AND_CORRECTIONS.md`'s Bug 1.
- **Remaining, expected (not a bug)**: with the corrected objective, CBC still cannot
  *prove* optimality within its 180s time limit on the 100-facility instances (100,000+
  continuous routing variables) — `MILPSolver.solve()` reports this honestly as
  `"Time Limit (Feasible, Not Proven Optimal)"` rather than falsely claiming
  `"Optimal"`. Even time-limited, MILP is consistently the closest of the three methods
  to the published ground truth (1-20% gap, vs. 4-16% for Classical GA and 17-54% for
  Greedy) — see Chapter 12 of `docs/CFLP_Complete_Project_Guide.md` for the full table.

### Issue 3: Population Diversity Loss
- **Problem**: GA converges to local optimum, missing global optimum
- **Mitigation**: Increase mutation rate, population size, or generations
- **Alternative**: Run GA multiple times with different random seeds

### Issue 4: Feature Engineering Brittleness
- **Problem**: ML features only valid for same instance type/size
- **Solution**: Retrain surrogate when switching to different instance series

### Issue 5: Cache Side Effects (FIXED)
- **Was**: Fitness cache persisted across runs, causing zero variance
- **Now**: Cache cleared between runs (Fixed in Bug 2)

### Issue 6: OR-Library Template Files Silently Corrupted (FIXED)
- **Was**: `capa.txt`/`capb.txt`/`capc.txt` are unfilled Beasley templates (capacity =
  literal text `"capacity"`); `parser.py` used to silently substitute a near-infinite
  placeholder number instead of erroring, corrupting 3 of 15 headline benchmark
  instances into an artificially uncapacitated problem.
- **Now**: `parser.py` raises a clear error naming the correct file to use instead
  (`capa4`/`capb4`/`capc4`, produced by `preprocess_orlib.py`). All benchmarks were
  switched to the correct files and re-run.

### Issue 7: ThreadPool Native Crashes on Large Instances (FIXED)
- **Was**: `ga_solver.py` used a `ThreadPool` to parallelize fitness evaluation for
  instances with >50 facilities, which reliably crashed with SIGSEGV — SciPy's
  `linprog`/HiGHS is not thread-safe for concurrent calls sharing one process.
- **Now**: fitness evaluation is sequential for all instance sizes; slower but stable.

---

## Troubleshooting

### Q: GA solution quality is poor (gap > 5%)
**Check**:
1. Population size too small (try 150+)
2. Not enough generations (try 200+)
3. Mutation rate wrong (should be ~1/m)
4. Instance is genuinely difficult (check with MILP)

**Fix**:
```python
best_cost, best_y, _ = ga_solver.solve(
    pop_size=150,      # Increase from 50
    n_gen=200,         # Increase from 100
    cx_pb=0.8,
    mut_pb=0.2
)
```

### Q: GA always terminates early
**Check**:
1. Solution space too small
2. Stagnation threshold too low (set to 10 generations)
3. Population converged prematurely

**Fix**:
Edit `src/ga_solver.py` line ~235:
```python
STAGNATION_LIMIT = 20  # Increase from 10
```

### Q: MILP solver times out
**Check**:
1. Instance is large (m=100 is borderline)
2. Instance is genuinely hard

**Fix**:
```python
timeout_sec = 300  # Increase from 120
milp_cost, y, x, status = milp_solver.solve(timeout_sec=timeout_sec)
```

### Q: Results don't match original CSV
**Check**:
1. Random seed differs (BASE_SEED=42 vs. other)
2. Code was modified
3. Original CSV was cached/fabricated (see BUG_FIXES_AND_CORRECTIONS.md)

**Fix**: Rerun benchmark with same seed:
```python
BASE_SEED = 42
np.random.seed(BASE_SEED)
random.seed(BASE_SEED)
```

### Q: Surrogate model predictions are inaccurate (MAPE > 5%)
**Check**:
1. Training corpus outdated
2. Feature engineering doesn't match current code
3. Instance type different from training

**Fix**:
```bash
# Retrain surrogate
python src/training_pipeline.py

# Use new model
surrogate = CFLPSurrogateModel.load("data/processed/surrogate_random_forest.pkl")
```

---

## Summary

The CAPL project provides **five complementary optimization approaches** for CFLP:

1. **MILP** (exact but slow)
2. **Greedy** (fast but low-quality)
3. **Classical GA** (good quality, medium speed) ← PRIMARY
4. **Modular GA** (experimental, research only)
5. **Hybrid ML-GA** (fast if surrogate is accurate, requires validation)

Choose based on your constraints:
- **Need optimal?** Use MILP (m ≤ 50)
- **Need fast?** Use Greedy
- **Need good balance?** Use Classical GA ← RECOMMENDED
- **Need to research?** Use Modular GA or Hybrid ML-GA

All approaches are rigorous, well-documented, and mutually comparable.

