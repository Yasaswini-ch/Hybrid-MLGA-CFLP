# Implementation Log: Technical Progress

This log documents the granular technical progress of the software engineering aspects of this project.

---

## Log Entries

### Step 1: Environment and Core Structure Setup (2026-05-23)
- **Directory Layout:** Created directories for modular structure:
  - `data/raw/` for raw OR-Library files.
  - `data/processed/` for serialized python datasets.
  - `src/` for python modules.
  - `docs/` for research documentation.
  - `notebooks/` for exploratory data analysis.
- **File Relocation:** Moved original Beasley benchmark datasets:
  - `cap41.txt` -> `data/raw/cap41.txt`
  - `cap42.txt` -> `data/raw/cap42.txt`
  - `cap43.txt` -> `data/raw/cap43.txt`
  - `cap44.txt` -> `data/raw/cap44.txt`
- **Dependencies:** Created `requirements.txt` containing `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `deap`, and `jupyter`.
- **Environment:** Created `.venv` virtual environment using Python 3.11.9 and initiated package installation.

### Step 2: Baseline Solver Implementation (2026-05-23)
- **Library Dependency:** Added `pulp>=2.6.0` to `requirements.txt` and successfully installed it in the `.venv` virtual environment.
- **Solver Module:** Designed and implemented `src/baseline.py`:
  - `GreedySolver`: Focuses on sorted cost-to-capacity efficiency ratios and greedy customer-facility flows.
  - `MILPSolver`: Modelled the Mixed-Integer Linear Program using `PuLP` variables and constraints, solved via Coin-OR CBC.
- **Batch Evaluation:** Created verification runner `run_benchmarks()` to test and compare solvers over all 4 datasets.

### Step 3: Genetic Algorithm Solver (2026-05-23)
- **Solver Module:** Coded `src/ga_solver.py` using `deap` framework.
- **Performance Optimization:** Replaced slow PuLP external solver subprocess calls inside fitness evaluation with `scipy.optimize.linprog(method='highs')` in-memory LP. This bypassed process creation and disk I/O, yielding a **500x speedup** per evaluation.
- **GA Configurations:** Enabled Tournament selection (size 3), Two-point crossover (prob 0.8), Flip-bit mutation (prob 0.2, bit indpb 0.05), and Elitism (carrying over best individual).
- **Visualization:** Integrated convergence plotting and saved output to `docs/cap41_ga_convergence.png`.

### Step 4: Machine Learning Surrogate & Hybrid Solver (2026-05-23)
- **Surrogate Training:** Developed `src/train_surrogate.py` to generate 100% of all possible feasible configurations (exactly 2,517 combinations) and evaluate their true costs using our optimized SciPy LP. Trained a Scikit-Learn `RandomForestRegressor` with $R^2 = 0.9299$ and $\text{MAPE} = 0.770\%$. Serialized model to `data/processed/surrogate_rf.pkl`.
- **Hybrid Solver:** Developed `src/hybrid_ga.py` replacing LP solves with Random Forest regressor predictions in-memory.
- **Comparative Analysis:** Configured `run_comparative_experiment()` to run MILP exact, Classical GA, and Hybrid GA solvers sequentially, displaying speedup factors and saving comparative plots to `docs/cap41_hybrid_convergence.png`.



---

## Technical Specifications

### Raw Files Moved
- `data/raw/cap41.txt` (Size: 10,212 bytes, 16 facilities, 50 customers, fixed cost: 7500.0)
- `data/raw/cap42.txt` (Size: 10,227 bytes, 16 facilities, 50 customers, fixed cost: 12500.0)
- `data/raw/cap43.txt` (Size: 10,227 bytes, 16 facilities, 50 customers, fixed cost: 17500.0)
- `data/raw/cap44.txt` (Size: 10,227 bytes, 16 facilities, 50 customers, fixed cost: 22500.0)

### Mathematical formulation of PuLP Exact model in `baseline.py`
```python
# Variables
y = pulp.LpVariable.dicts("y", range(16), cat=pulp.LpBinary)
x = pulp.LpVariable.dicts("x", ((j, i) for j in range(50) for i in range(16)), lowBound=0)

# Objective
prob += pulp.lpSum(fixed_costs[i] * y[i]) + pulp.lpSum(transport_costs[j,i] * x[j,i])

# Constraints
# Customer Demands Met
for j in range(50):
    prob += pulp.lpSum(x[j, i] for i in range(16)) == demands[j]
# Facility capacity boundaries
for i in range(16):
    prob += pulp.lpSum(x[j, i] for j in range(50)) <= capacities[i] * y[i]
```

### In-Memory SciPy linprog Formulation in `ga_solver.py`
We flatten decision variables to $x_{j, k}$ where $j \in [0, n-1]$ is the customer index, and $k \in [0, \text{num\_open}-1]$ is the open facility index.
```python
# Objective: Flattened transport costs
c = [transport_costs[j, i] for j in range(50) for i in open_indices]

# Equality Constraints: Sum_{k} x_{jk} == demands[j]
A_eq = np.zeros((50, 50 * num_open))
for j in range(50):
    A_eq[j, j * num_open : (j + 1) * num_open] = 1.0
b_eq = demands

# Inequality Constraints: Sum_{j} x_{jk} <= capacities[open_indices[k]]
A_ub = np.zeros((num_open, 50 * num_open))
for k in range(num_open):
    for j in range(50):
        A_ub[k, j * num_open + k] = 1.0
b_ub = capacities[open_indices]

# Solve in-memory
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0.0, None)] * len(c), method='highs')
```

---

### Step 5: Generalization and cap51 Integration (2026-05-25)
- **Dynamic File Discovery**: Upgraded `src/verify_parser.py` and `src/baseline.py` to scan `data/raw` dynamically using `glob.glob(os.path.join(raw_dir, "cap*.txt"))`. This decouples the modules from hardcoded lists of datasets.
- **Regex-Based Numerical Sorting**: Implemented `numerical_sort_key` employing regular expressions `re.search(r'\d+', filename)` to extract the benchmark index. This guarantees mathematically correct sorting (e.g. `cap41` < `cap51`) rather than lexical ASCII sorting (which breaks when digits differ in length).
- **Robust Field Boundaries**: Updated parser metadata rendering block to safely index `fixed_costs` with bounds checking (`len(dataset.fixed_costs) > 10`) to prevent index out of bounds exceptions on smaller instances.
- **Verification Execution**: Ran dynamic solvers successfully, benchmarked all 5 files, and captured objective values, computational CPU times, and active facility sets.

---

### Step 6: Problem Set VI Benchmarks & Dynamic Automated Execution (2026-05-25)
- **Zero-Modification Execution**: Successfully executed the dynamic pipeline scripts `src/verify_parser.py` and `src/baseline.py` to automatically capture the parameters and benchmark performance of the four new Problem Set VI datasets (`cap61`, `cap62`, `cap63`, `cap64`) with absolutely zero changes to the underlying source code. This confirms the perfect generalization of the file scanner.
- **Parametric Dissection**: Dissected facility capacity limits at $15,000$ (representing 3x capacity scaling compared to Problem Set IV). Captured the exact mathematical capacity/demand ratio of **4.1189** and a minimum active facility requirement of **4**.
- **Solvers Logging**: Logged output results row-by-row for the MILP Exact solver and Greedy Heuristic, showing that MILP maintains $16/16$ opened facilities while Greedy opens exactly $4/16$, resulting in a massive **36.39% optimality gap** ($>\$1.45$ billion in inefficient routing overhead).

---

### Step 7: Problem Set VII Dynamic Execution and UFLP Boundary Benchmarks (2026-05-25)
- **Zero-Modification dynamic Verification**: Successfully ran `src/verify_parser.py` and `src/baseline.py` in batch mode with zero manual script modifications. The file discovery loop automatically discovered all 13 datasets (`cap41` to `cap74`) in `data/raw/` and parsed/solved them.
- **Parametric Dissection**: Dissected UFLP capacity boundary scaling, logging facility capacities at $58,268$ (100% of total system demand). Captured the overall capacity/demand ratio of **16.0000** and a minimum feasibility facility threshold of **1**.
- **Solvers Metrics Logging**: Captured and logged Greedy vs. MILP solver comparisons. Verified that exact MILP solutions are mathematically identical to the penny to PS VI, opening $16/16$ facilities to route all customers to their closest supply source. Greedy opened exactly $1/16$ (the free facility 11) to zero out standard opening costs, incurring a massive **42.35% optimality gap** ($>\$1.69$ billion in wasted transport cost).

---

### Step 8: Problem Set VIII Dynamic Scaling to 25 Facilities (2026-05-25)
- **Zero-Modification dynamic Verification**: Successfully ran `src/verify_parser.py` and `src/baseline.py` in batch mode with zero manual script modifications. The file discovery loop automatically discovered all 17 datasets (`cap41` to `cap84`) in `data/raw/` and parsed/solved them. This confirms the flawless scalability of our dynamically tokenized parser code when scaling the facility dimension from $16$ to $25$.
- **Parametric Dissection**: Dissected facility location scaling, logging $m = 25$ facilities and $n = 50$ customers. Captured the overall capacity/demand ratio of **2.1453** and a minimum feasibility facility threshold of **12** (each facility has a capacity of $5,000$).
- **Solvers Metrics Logging**: Captured and logged Greedy vs. MILP solver comparisons. Verified that MILP solved `cap81-84` to mathematical optimality in under 370 ms, choosing to open all 25 facilities to minimize variable transportation costs (reducing total system cost by over $\$1.22$ billion compared to `cap41`). Greedy opened exactly $12/25$ facilities, incurring a massive **63.39% optimality gap** ($>\$1.99$ billion in wasted transport cost).

---

### Step 9: Problem Set IX Integration and Scaling under Loose Constraints (2026-05-25)
- **Zero-Modification dynamic Verification**: Successfully ran `src/verify_parser.py` and `src/baseline.py` in batch mode with zero manual script modifications. The file discovery loop automatically scanned, sorted, parsed, and solved all 21 raw datasets inside `data/raw/` in a single run. This verifies that our dynamic file scanning architecture functions with complete modularity and generalizability across all 21 OR-Library benchmark files.
- **Parametric Dissection**: Dissected facility location scaling, logging $m = 25$ facilities and $n = 50$ customers. Captured the overall capacity/demand ratio of **6.4358** (with $s_i = 15,000$ per facility) and a minimum feasibility facility threshold of **4**.
- **Solvers Metrics Logging**: Captured and logged Greedy vs. MILP solver comparisons. Verified that MILP solved `cap91-94` to mathematical optimality in under 260 ms, choosing to open all 25 facilities to minimize variable transportation costs (reducing total system cost to **\$2,860,332,101.90**, saving over **\$1.50 billion** compared to `cap41`). Greedy opened exactly **4/25** facilities, incurring a catastrophic **90.93% optimality gap** ($>\$2.60$ billion in wasted transport cost).

---

### Step 10: Problem Set X Integration and Scaled Uncapacitated Boundary Complexity Studies (2026-05-25)
- **Zero-Modification dynamic Verification**: Successfully ran `src/verify_parser.py` and `src/baseline.py` in batch mode with zero manual script modifications. The file discovery loop automatically scanned, sorted, parsed, and solved all 25 raw datasets inside `data/raw/` in a single run. This verifies that our dynamic file scanning architecture functions with complete modularity and generalizability across all 25 OR-Library benchmark files.
- **Parametric Dissection**: Dissected facility location scaling, logging $m = 25$ facilities and $n = 50$ customers. Captured the overall capacity/demand ratio of **25.0000** (with $s_i = 58,268$ per facility, matching the uncapacitated boundary) and a minimum feasibility facility threshold of **1**.
- **Solvers Metrics Logging**: Captured and logged Greedy vs. MILP solver comparisons. Verified that MILP solved `cap101-104` to mathematical optimality in under 270 ms, choosing to open all 25 facilities to minimize variable transportation costs (reducing total system cost to **\$2,860,332,101.90**, saving over **\$1.50 billion** compared to `cap41`). Greedy opened exactly **1/25** facilities (facility 11), incurring a catastrophic **99.27% optimality gap** ($>\$2.839$ billion in wasted transport cost).

---

### Step 11: Problem Set XI Integration and High-Dimensional Complexity Studies (2026-05-25)
- **Zero-Modification dynamic Verification**: Successfully ran `src/verify_parser.py` and `src/baseline.py` in batch mode with zero manual script modifications. The file discovery loop automatically scanned, sorted, parsed, and solved all 29 raw datasets inside `data/raw/` in a single run. This verifies that our dynamic file scanning architecture functions with complete modularity and generalizability across all 29 OR-Library benchmark files when candidate facilities double from 25 to 50.
- **Parametric Dissection**: Dissected facility location scaling, logging $m = 50$ facilities and $n = 50$ customers. Captured the overall capacity/demand ratio of **4.2905** (with $s_i = 5,000$ per facility) and a minimum feasibility facility threshold of **12**.
- **Solvers Metrics Logging**: Captured and logged Greedy vs. MILP solver comparisons. Verified that MILP solved `cap111-114` to mathematical optimality in under 605 ms, choosing to open between 45 and 47 facilities to minimize total costs (achieving **\$3,079,471,950.08** for `cap111`). Greedy opened exactly **12/50** facilities, incurring a catastrophic **114.27% optimality gap** ($>\$3.519$ billion in wasted transport cost).

---

### Step 12: Problem Set XII Integration and Scaled Loose Capacity Complexity Studies (2026-05-25)
- **Zero-Modification dynamic Verification**: Successfully ran `src/verify_parser.py` and `src/baseline.py` in batch mode with zero manual script modifications. The file discovery loop automatically scanned, sorted, parsed, and solved all 33 raw datasets inside `data/raw/` in a single run. This verifies that our dynamic file scanning architecture functions with complete modularity and generalizability across all 33 OR-Library benchmark files when candidate facilities are doubled to 50 under loose constraints.
- **Parametric Dissection**: Dissected facility location scaling, logging $m = 50$ facilities and $n = 50$ customers. Captured the overall capacity/demand ratio of **12.8716** (with $s_i = 15,000$ per facility) and a minimum feasibility facility threshold of **4**.
- **Solvers Metrics Logging**: Captured and logged Greedy vs. MILP solver comparisons. Verified that MILP solved `cap121-124` to mathematical optimality in under 605 ms, choosing to open between 45 and 47 facilities to minimize total costs (achieving **\$2,850,307,905.40** for `cap121`). Greedy opened exactly **4/50** facilities, incurring a catastrophic **249.94% optimality gap** ($>\$7.12$ billion in wasted transport cost).

---

### Step 13: Problem Set XIII Integration and Scaled Uncapacitated Boundary Complexity Studies (2026-05-25)
- **Zero-Modification dynamic Verification**: Successfully ran `src/verify_parser.py` and `src/baseline.py` in batch mode with zero manual script modifications. The file discovery loop automatically scanned, sorted, parsed, and solved all 37 raw datasets inside `data/raw/` in a single run. This verifies that our dynamic file scanning architecture functions with complete modularity and generalizability across all 37 OR-Library benchmark files when candidate facilities are doubled to 50 under uncapacitated bounds.
- **Parametric Dissection**: Dissected facility location scaling, logging $m = 50$ facilities and $n = 50$ customers. Captured the overall capacity/demand ratio of **50.0000** (with $s_i = 58,268$ per facility) and a minimum feasibility facility threshold of **1**. Documented the shift of the free facility to index 22.
- **Solvers Metrics Logging**: Captured and logged Greedy vs. MILP solver comparisons. Verified that MILP solved `cap131-134` to mathematical optimality in under 576 ms, choosing to open between 45 and 47 facilities to minimize total costs (achieving **\$2,850,307,905.40** for `cap131`). Greedy opened exactly **1/50** facilities, incurring a catastrophic **99.98% optimality gap** ($>\$2.849$ billion in wasted transport cost).

---

### Step 14: Phase 2 Code Modules Implementation (2026-05-25)
- **Structured solution mapping (`src/solution_representation.py`)**: Implemented the structured `CFLPSolution` class, shape and type validation checking, and continuous-to-discrete conversion utility `convert_flow_to_allocations` based on row argmax coordinate scans.
- **Hadamard-product cost calculations (`src/cost_calculator.py`)**: Vectorized overhead calculations (`np.sum(y * fixed_costs)`) and variable shipping costs (`np.sum(x * transport_costs)`).
- **Defensive constraints checking (`src/constraint_checker.py`)**: Vectorized demand matching, warehouse storage limitations, and closed facility flow origin checks. Bound comparisons under high-precision double-tolerance threshold bounds ($\epsilon = 10^{-7}$).
- **Nearest Feasible baseline solver (`src/baseline_solver.py`)**: Developed `GreedyBaselineSolver` utilizing the efficiency-sorted sequential warehouse opening and Nearest Feasible Facility greedy demand assignment algorithm, safe-guarded by forced-open fallback re-solves.

---

### Step 15: Phase 2 Integration Verification Execution (2026-05-25)
- **Verification script (`src/verify_phase2.py`)**: Created a comprehensive testing runner to run end-to-end baseline solvers, cost calculations, constraint checks, and perturbed demand under-satisfaction and closed facility shipping violations.
- **Verification execution**: Executed python test runner, verifying 100% feasibility and cost alignment on `cap41.txt` ($82.5$k opening cost, $5.132$B shipping cost, total cost **\$5,132,128,742.76** with 12 open warehouses) and flawless perturbation boundary captures.

---

### Step 16: Phase 3 Modular Genetic Algorithm — Core Module Implementation (2026-05-25)
- **`src/chromosome.py`**: Implemented `CFLPChromosome` class wrapping binary facility vectors with strict binary validation, `active_count()`, and `hamming_distance()` for Hamming-based population diversity tracking.
- **`src/population.py`**: Implemented `CFLPPopulationGenerator` with `generate_random_individual()` (purely random binary), `generate_heuristic_seeded_individual()` (guaranteed minimum feasibility), and `create_population()` with configurable `heuristic_ratio` blending.
- **`src/fitness.py`**: Implemented `CFLPFitnessEvaluator` formulating the $n \times \text{num\_open}$ continuous LP sub-problem, solving in-memory via SciPy HiGHS, reconstructing the dense $n \times m$ flow matrix from flattened LP variables, and feeding the result through `CFLPSolution`, `is_feasible()`, and `calculate_total_cost()`.
- **`src/repair.py`**: Implemented `CFLPFeasibilityRepairer` — the Lamarckian repair operator. Precomputes efficiency ratios ($f_i / s_i$) at init time. On `repair()` call, identifies capacity deficit, sorts closed facilities by efficiency, and greedily opens them in-place until $\sum s_i y_i \ge \sum d_j$.
- **`src/selection.py`**: Implemented `tournament_select()` (default, tournsize=3), minimization-mapped `roulette_select()`, and `apply_elitism()` (replaces worst offspring slots with elite clones from prior generation).
- **`src/crossover.py`**: Implemented `single_point_crossover()`, `two_point_crossover()` (default), and `uniform_crossover()`.
- **`src/mutation.py`**: Implemented `bit_flip_mutation()` inverting genes at each index with probability `indpb = 1/m`.
- **`src/genetic_algorithm.py`**: Implemented `ModularCFLPGASolver` supporting "penalty" and "repair" modes. Implemented `plot_experiments()` for comparative convergence plots and `run_experiments()` for automated benchmark execution.

---

### Step 17: Phase 3 — GA Comparative Experiment Execution on cap41.txt (2026-05-25)
- **Execution**: Ran `src/genetic_algorithm.py` on `cap41.txt`. Completed 4-stage experiment: MILP → Greedy → GA Pure Penalty (100 gens × 50 pop) → GA Lamarckian Repair (100 gens × 50 pop).
- **Key Results**:
  - MILP Optimal: **$4,368,647,185.19** (16/16 open)
  - Greedy Baseline: **$5,132,128,742.76** (12/16 open, 17.4764% gap)
  - GA Pure Penalty: **$4,368,647,185.19** (16/16 open, **0.0000%** gap, 61.67 s)
  - GA Lamarckian Repair: **$4,368,647,185.19** (16/16 open, **0.0000%** gap, 61.22 s)
- **Feasibility at Gen 0**: Penalty = 88% | Repair = 100% — validates Lamarckian operator.
- **Diversity Collapse**: Hamming distance collapsed to ≈ 0.08 by Gen 10.
- **Convergence Plot**: Saved to `docs/cap41_ga_convergence.png`.

---

### Step 18: High-Performance Regression Metrics & Latency Profiling (2026-05-25)
- **Unified Metric Suite (`src/evaluation_metrics.py`)**: Implemented publication-grade evaluation routines computing MAE, RMSE, R² score, and MAPE (critical for Cost Optimization models).
- **Latency Diagnostics**: Coded `compute_latency_speedup` timing predictions in microseconds and calculating speedup factors compared to Exact continuous SciPy LP times.

### Step 19: Unified Feature Engineering Design (2026-05-25)
- **Aggregation Enriched Chromosomes (`src/feature_engineering.py`)**: Designed `CFLPFeatureEngineer` mapping binary chromosomes into enriched feature representations. Incorporates raw binary facility choices and four engineered scalar parameters: Active facility count, Total active capacity, Capacity slack ratio, and Weighted average facility fixed cost.

### Step 20: Unified Multi-Regressor Surrogate Module & Training Pipeline (2026-05-25)
- **Multi-Architecture Support (`src/surrogate_model.py`)**: Designed wrapper encapsulating Random Forest, Gradient Boosting, XGBoost, and MLP. RF includes inter-tree variance calculation for uncertainty scoring.
- **Training Orchestration (`src/training_pipeline.py`)**: Structured end-to-end training pipeline implementing 80/20 train/test splitting, metric summaries, and pickle serialization. XGBoost achieved a stunning **R² of 0.9922** and **MAPE of 0.2758%** with a **2,810x latency speedup**.

### Step 21: Confidence-Aware Hybrid GA Solver (2026-05-25)
- **Modular Integration (`src/hybrid_ga.py`)**: Rewrote `hybrid_ga.py` to seamlessly integrate with our Phase 3 modular evolutionary framework.
- **Confidence-Aware & Warmup Schemes**: Enabled warmup periods (using exact LP for early generations to build elite chromosomes) and uncertainty-based fallbacks (exact LP triggered if RF standard deviation exceeds 5%).
- **Verification Benchmarking**: Executed a controlled comparative study on `cap41.txt`. Hybrid RF confidence-aware solver matched the global MILP optimum exactly to the penny (**$4,368,647,185.19**) at a **4.0x speedup** (22.62s vs 90.15s).

### Step 22: Active Learning Loop & Iterative Surrogate Refinement (2026-05-25)
- **Active Refinement Loop (`src/active_learning.py`)**: Developed an active learning loop collecting GA-explored chromosomes, appending them to the training dataset, and retraining models.
- **Empirical Execution**: Completed 3 active learning rounds on `cap41.txt`, augmenting the initial corpus from 2,517 to 2,936 samples. R² score rose monotonically from **0.936342** to **0.999974** by Round 3, representing an extremely high-accuracy surrogate mapping.












