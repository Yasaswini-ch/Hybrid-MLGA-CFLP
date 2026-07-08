# Hybrid ML-GA Solver for Capacitated Facility Location Problem (CFLP)

An academic and engineering project aimed at building a high-performance, hybrid solver integrating **Machine Learning (ML)** and **Genetic Algorithms (GA)** to solve Beasley's OR-Library benchmarks for the Capacitated Facility Location Problem (CFLP) and the Uncapacitated Facility Location Problem (UFLP).

---

## 📖 Table of Contents
1. [📌 Important: Read These Guides First](#-important-read-these-guides-first)
2. [Overview](#-overview)
3. [Scientific Background](#-scientific-background)
4. [Project Directory Layout & File Uses](#-project-directory-layout--file-uses)
5. [Environment Setup](#-environment-setup)
6. [First-Time User Execution Guide (Running Order)](#-first-time-user-execution-guide-running-order)
7. [Detailed Execution Flow (What Happens Under the Hood)](#-detailed-execution-flow-what-happens-under-the-hood)
8. [Hybrid ML-GA Workflow (Latest)](#-hybrid-ml-ga-workflow-latest)
9. [Latest Verified Benchmark Results](#-latest-verified-benchmark-results)

---

## 📌 Important: Read These Guides First

This project has been comprehensively audited, debugged, and documented. **Start here**:

### For Everyone:
- **[QUICK_START.md](docs/QUICK_START.md)** — 5-minute guide to run benchmarks and understand the code
- **[CFLP_Complete_Project_Guide.md — Chapter 16](docs/CFLP_Complete_Project_Guide.md)** — Plain-language walkthrough of the latest Hybrid ML-GA fixes and results (start here if you only read one thing)

### If You Want Details:
- **[BUG_FIXES_AND_CORRECTIONS.md](docs/BUG_FIXES_AND_CORRECTIONS.md)** — What bugs existed, how they were fixed, and how to verify
- **[IMPLEMENTATION_ARCHITECTURE.md](docs/IMPLEMENTATION_ARCHITECTURE.md)** — Complete guide to all modules, algorithms, and how to run each solver
- **[REPRODUCIBILITY_AND_VERIFICATION.md](docs/REPRODUCIBILITY_AND_VERIFICATION.md)** — How to reproduce results and run verification tests
- **[PHASE_4_HYBRID_BENCHMARK_REPORT.md](docs/PHASE_4_HYBRID_BENCHMARK_REPORT.md)** — Full Hybrid ML-GA re-benchmark results on all 15 OR-Library instances, including an honest analysis of where it currently underperforms and why

### Project Status:
- ✅ **All critical bugs fixed** (6 bugs identified and corrected)
- ✅ **Fully reproducible** (random seeds managed, caching fixed)
- ✅ **Research-ready** (defensive documentation, verification tests)
- ✅ **Hybrid ML-GA can now bootstrap its own training data** — no pre-trained model required to start (see [§8](#-hybrid-ml-ga-workflow-latest))
- ✅ **Predicted-cost decision logic corrected** to match the intended design: exact verification only when a prediction suggests a new best solution
- ✅ **Adaptive retraining is quality-gated** — a retraining round that produces a worse model is automatically rejected, never silently adopted
- ✅ **Re-benchmarked on all 15 OR-Library instances** with the corrected implementation (see [§9](#-latest-verified-benchmark-results))

---

---

## 🎯 Overview

The **Capacitated Facility Location Problem (CFLP)** is a critical supply-chain optimization problem. Given a set of potential facilities with opening costs and capacity bounds, and a set of customers with demands and variable transportation costs, we seek to determine which facilities to open and how to allocate customer demands to minimize total operational costs.

This project implements:
1. A robust Python parser for standard Beasley OR-Library datasets.
2. Baseline solvers including Greedy Heuristics and exact Mixed-Integer Linear Programming (MILP).
3. A classical Genetic Algorithm (GA) powered by the `DEAP` framework.
4. A **Hybrid ML-GA** model using a machine learning model as a **surrogate fitness function** to accelerate search iterations.

---

## 🔬 Scientific Background

### Objective Function
We aim to minimize:
$$Z = \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$$

Subject to:
- Customer Demand satisfaction: $\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$
- Facility Capacity constraint: $\sum_{j \in J} x_{ij} \le s_i y_i \quad \forall i \in I$
- Variable domains: $y_i \in \{0, 1\}, \quad x_{ij} \ge 0$

---

## 📂 Project Directory Layout & File Uses

```text
CAPL/
├── data/
│   ├── raw/                  # Beasley OR-Library text files (cap71 ... cap134, capa ... capc)
│   └── processed/            # Serialized pre-trained Machine Learning surrogate models (.pkl)
├── src/                      # Modular Python modules
│   ├── parser.py             # OR-Library Dataset Parser
│   ├── baseline.py           # Greedy & PuLP MILP Solvers
│   ├── baseline_solver.py    # Baseline solver wrapper classes
│   ├── chromosome.py         # Binary solution representation
│   ├── population.py         # Population initialization & heuristic seeding
│   ├── selection.py          # Selection operators (Tournament, Elitism)
│   ├── crossover.py          # Mating operators (Two-Point, Uniform)
│   ├── mutation.py           # Mutation operator (Bit-Flip)
│   ├── repair.py             # Constraint repairer (restores empty/invalid status configurations)
│   ├── fitness.py            # Exact cost and constraint penalization calculator
│   ├── cost_calculator.py    # Vectorized transportation cost helper
│   ├── constraint_checker.py # Facility capacity checking helper
│   ├── genetic_algorithm.py  # Modular Classical GA Solver class
│   ├── surrogate_model.py    # ML surrogate models for fitness approximation
│   ├── feature_engineering.py# Encodes binary configurations into numerical features for ML
│   ├── training_pipeline.py  # Generates training data and fits models (supports a fixed external validation set)
│   ├── hybrid_ga.py          # HybridMLGASolver — can bootstrap its own training data (surrogate=None) and uses predicted-cost-vs-best-solution decision logic
│   ├── active_learning.py    # Quality-gated adaptive retraining loop (rejects worse models, never adopts them)
│   ├── dataset_generator.py  # Synthesizes datasets for testing; de-duplicates GA-derived training samples
│   ├── evaluation_metrics.py # Evaluates ML surrogate MAPE accuracy
│   ├── preprocess_orlib.py   # One-time utility that split capa/capb/capc into the capa1-4/capb1-4/capc1-4 variants used by benchmark_large.py
│   ├── benchmark_statistical.py # Classical GA benchmark across all 15 OR-Library instances (30 runs each) — Table 2 results
│   ├── benchmark_hybrid_ga.py   # Hybrid ML-GA benchmark across all 15 OR-Library instances (bootstrap + confidence-aware, 10 runs each)
│   └── benchmark_large.py    # MILP vs. Greedy vs. Classical GA on the split large-scale variants (capa1-4, capb1-4, capc1-4)
├── docs/                     # Scientific reports, results spreadsheet (.csv), and graphs (.png)
│   ├── statistical_benchmark_results.csv  # Classical GA results (Optimal vs GA stats, 15 instances)
│   ├── hybrid_benchmark_results.csv       # Hybrid ML-GA results on the same 15 instances (fresh, corrected implementation)
│   ├── PHASE_4_HYBRID_BENCHMARK_REPORT.md # Full write-up comparing Hybrid ML-GA vs. Classical GA, with root-cause analysis
│   ├── large_benchmark_results.csv # Large-scale comparison output results
│   ├── hybrid_ga_comparison.png    # Performance comparison plot
│   ├── cap41_ga_convergence.png    # Classical GA fitness improvement graph
│   └── cap41_hybrid_convergence.png# Hybrid GA fitness improvement graph
└── requirements.txt          # Python library dependencies
```

### Detailed Output File Explanations:
1. **Pre-trained ML Models (`data/processed/*.pkl`)**: 
   * Serialized Python objects containing trained models (Random Forest, Gradient Boosting, XGBoost). Once trained, they are saved here so the GA can load them and immediately predict facility transportation costs in microseconds.
2. **Spreadsheet Logs (`docs/*.csv`)**:
   * Output files automatically generated by running the benchmark scripts. They log execution stats (costs, standard deviations, gaps) for grading and analysis.
3. **Convergence Graphs (`docs/*.png`)**:
   * Visual plots displaying how the total cost drops over generations, proving that the crossover, mutation, and selection logic successfully guide the search toward the optimal value.

---

## ⚙️ Environment Setup

### Prerequisites
- Python 3.10.x or higher installed.

### Virtual Environment Setup

1. **Create a Virtual Environment:**
   ```powershell
   python -m venv .venv
   ```
2. **Activate the Environment:**
   - **On Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **On Linux / macOS (bash):**
     ```bash
     source .venv/bin/activate
     ```
3. **Install Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 First-Time User Execution Guide (Running Order)

A first-time user can run the benchmark scripts out-of-the-box. The scripts automatically handle loading raw data, performing the optimizations, and generating the outputs.

### Step 1: Run the Statistical Classical GA Benchmark (Produces Table 2 Statistics)
This script runs the classical GA on all 15 instances (`cap71`-`cap134`, and `capa`-`capc`) across 30 random seeds, reproducing the exact publication table:
```bash
python src/benchmark_statistical.py
```
* **What it outputs**: Prints a formatted statistical markdown table directly to the console and generates:
  * `docs/statistical_benchmark_results.csv` (complete spreadsheet logs)
  * `docs/statistical_benchmark_results.png` (gap performance bar chart)
  * `docs/computational_table.png` (high-quality dark-themed table rendering)
* **Execution Time**: ~7.7 minutes (using parallelized solves and UFLP evaluation shortcuts).

### Step 2: Run the Large-Scale Capacitated Benchmark
This script runs a comparative analysis of the exact MILP solver, the greedy heuristic, and the classical GA solver on high-dimensional capacitated Beasley benchmarks (`capa1-4` to `capc1-4`):
```bash
python src/benchmark_large.py
```
* **What it outputs**: Prints details of cost gaps, facility counts, and runtimes, and generates [docs/large_benchmark_results.csv](file:///c:/Opensource/CAPL/docs/large_benchmark_results.csv).

### Step 3 (Optional): Train the Surrogate Models
If you wish to retrain and regenerate the machine learning models (.pkl files) from scratch:
```bash
python src/training_pipeline.py
```

### Step 4: Run the Hybrid ML-GA Benchmark (No Pre-Trained Model Needed)
This script runs the full Hybrid ML-GA pipeline — bootstrap its own training data, train a surrogate, then solve with the corrected predicted-cost decision logic — on all 15 OR-Library instances, directly comparable to Step 1's Classical GA results:
```bash
python src/benchmark_hybrid_ga.py
```
* **What it outputs**: `docs/hybrid_benchmark_results.csv`, directly comparable column-for-column with `docs/statistical_benchmark_results.csv`.
* **What it does NOT need**: any pre-existing `.pkl` model file — everything is generated from scratch for each instance.
* See **[docs/PHASE_4_HYBRID_BENCHMARK_REPORT.md](docs/PHASE_4_HYBRID_BENCHMARK_REPORT.md)** for the full results table and an honest analysis of where the hybrid approach currently wins and where it currently falls behind.

---

## 🔬 Detailed Execution Flow (What Happens Under the Hood)

When you run `python src/benchmark_statistical.py`, the following sequence occurs for each benchmark instance:

1. **Data Load & Reference Calculation**:
   * The raw text file is read by `parser.py` into NumPy arrays.
   * The published OR-Library optimal cost is loaded as the reference for gap calculations.
2. **Genetic Algorithm Search (30 Runs)**:
   * The GA solver is run 30 times with deterministic seeds (base seed 42 + run index) for full reproducibility.
   * Each run initializes the population using **Smart Individual Generation** with capacity-aware heuristic seeding.
   * Evolutionary generations proceed, selecting parent solutions via **Tournament Selection**, breeding them via **Two-Point Crossover**, and introducing variations via **Bit-Flip Mutations**.
   * Each individual is evaluated by solving the continuous transportation LP sub-problem using SciPy's HiGHS solver.
3. **Statistical Aggregation**:
   * The 30 actual GA costs are aggregated to compute Best, Average, Worst, Median, and Standard Deviation.
   * Optimality gaps are computed relative to the published OR-Library optimal cost.
4. **Report Generation**:
   * The final statistics are printed in a clean markdown table and saved to `docs/statistical_benchmark_results.csv` and `docs/statistical_benchmark_results.png`.

---

## 🤖 Hybrid ML-GA Workflow (Latest)

The Hybrid ML-GA (`src/hybrid_ga.py`) combines a Genetic Algorithm with a Machine Learning
surrogate model that predicts solution cost instead of solving the expensive LP
sub-problem every time. The workflow now runs in three clean stages, and — importantly —
**no pre-trained model is required to get started**.

### Stage 1 — Bootstrap: The GA Generates Its Own Training Data

```python
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga

bootstrap_ga = HybridMLGASolver(dataset=dataset, surrogate=None)
result = bootstrap_ga.solve()                       # runs a normal exact-LP GA search
X, y = extract_training_data_from_ga(result, dataset=dataset)  # ready-to-train dataset
```

Passing `surrogate=None` runs the GA exactly like a classical GA — every candidate is
evaluated with the real LP solver — but every `(chromosome, cost)` pair evaluated is
logged and later converted into a training dataset, with duplicate chromosomes
automatically removed.

### Stage 2 — Train a Surrogate Model

```python
from training_pipeline import SurrogateTrainingPipeline

pipeline = SurrogateTrainingPipeline(dataset=dataset, corpus_path=corpus_path, model_save_dir=model_dir)
trained = pipeline.run(model_types=("random_forest",))
surrogate = trained["random_forest"]["surrogate"]
```

### Stage 3 — Run the Hybrid GA With the Trained Surrogate

```python
informed_ga = HybridMLGASolver(
    dataset=dataset, surrogate=surrogate,
    mode="confidence_aware", warmup_fraction=0.15
)
result = informed_ga.solve()
```

**Decision rule (matches the original design intent exactly):** the surrogate predicts a
cost for each candidate; the real LP solver is only invoked to double-check that
prediction if it suggests a **new best solution** (`predicted_cost < best_cost_so_far`).
Otherwise, the prediction is trusted directly. This means most of the population is
evaluated in microseconds instead of milliseconds, while every genuinely promising
candidate still gets verified with the exact method before being trusted.

### Optional Stage 4 — Adaptive Retraining (Quality-Gated)

```python
from active_learning import SurrogateActiveLearner

learner = SurrogateActiveLearner(dataset_path=dataset_path, corpus_path=corpus_path, model_save_dir=model_dir)
history = learner.run_active_learning(n_rounds=5, mode="confidence_aware")
```

Each round collects new exact evaluations from a fresh Hybrid GA run, retrains the
surrogate, and only **adopts** the retrained model if it performs at least as well as the
best model seen in any earlier round — evaluated on a fixed validation set that never
changes, so the comparison is always fair. A round that produces a worse model is
rejected automatically, and the file every other part of the project loads is guaranteed
to always be the accepted (best-known) model, never a rejected one.

---

## 📊 Latest Verified Benchmark Results

Both the Classical GA and the (corrected) Hybrid ML-GA were run fresh on all 15
OR-Library CFLP instances. Full numbers: [docs/statistical_benchmark_results.csv](docs/statistical_benchmark_results.csv) and [docs/hybrid_benchmark_results.csv](docs/hybrid_benchmark_results.csv). Full analysis: [docs/PHASE_4_HYBRID_BENCHMARK_REPORT.md](docs/PHASE_4_HYBRID_BENCHMARK_REPORT.md).

**Best-run optimality gap versus the published OR-Library optimal cost:**

| Instance group | Facilities | Classical GA | Hybrid ML-GA |
|:---|:---:|:---:|:---:|
| cap71 – cap74   | 16  | 0.00% | 0.00% |
| cap101 – cap104 | 25  | 0.00% | 0.00% |
| cap131 – cap134 | 50  | 0.00% | 0.00% – 0.58% |
| capa, capb, capc | 100 | 1.7% – 2.2% | 8.5% – 13.3% |

**Summary:** On small and medium instances, the Hybrid ML-GA matches the Classical GA
almost exactly. On the three largest instances (100 facilities, 1000 customers), it
currently trails behind. This was investigated directly rather than assumed: the best
training example the surrogate ever saw for the `capa` instance was itself 15.6% away
from optimal — the model was simply never shown data good enough to guide the search
further, a data-coverage limitation rather than a defect in the surrogate model or the
decision logic. Full root-cause analysis in [docs/PHASE_4_HYBRID_BENCHMARK_REPORT.md](docs/PHASE_4_HYBRID_BENCHMARK_REPORT.md).
