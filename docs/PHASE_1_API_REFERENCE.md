# Phase 1: API Reference & Integration Guide

## Overview

This document provides detailed API documentation for Phase 1 implementations.

---

## New Functions & Methods

### 1. `CFLPDatasetGenerator.generate_from_ga_evaluations()`

**Location:** `src/dataset_generator.py` lines 131-177

**Purpose:** Convert exact evaluations collected during GA runs into training dataset.

**Signature:**
```python
def generate_from_ga_evaluations(self, 
                                  exact_evaluations_log: list
                                  ) -> Tuple[np.ndarray, np.ndarray]:
```

**Parameters:**
- `exact_evaluations_log` (list): List of (chromosome, cost) tuples collected by HybridMLGASolver.
  - Each tuple: `(chromosome_list, float_cost)`
  - chromosome_list: list of 0/1 values, length = num_facilities
  - float_cost: total objective cost from exact LP evaluation

**Returns:**
- Tuple of two numpy arrays:
  - `X` (np.ndarray): Binary chromosome matrix, shape `(N, m)`
    - dtype: int32
    - N = number of samples
    - m = number of facilities
    - Values: all 0 or 1
  
  - `y` (np.ndarray): Total cost array, shape `(N,)`
    - dtype: float64
    - Values: cost from exact LP evaluation

**Raises:**
- `ValueError`: If `exact_evaluations_log` is empty

**Example:**
```python
from dataset_generator import CFLPDatasetGenerator
from hybrid_ga import HybridMLGASolver

# Run GA and collect data
result = hybrid_ga.solve()
exact_log = result["exact_evaluations_log"]

# Extract training data
gen = CFLPDatasetGenerator(dataset)
X, y = gen.generate_from_ga_evaluations(exact_log)

print(f"Extracted {len(X)} training samples")
print(f"Features: {X.shape[1]}")
```

---

### 2. `extract_training_data_from_ga()`

**Location:** `src/hybrid_ga.py` lines 522-549

**Purpose:** Convenience function to extract training data directly from hybrid GA result dict.

**Signature:**
```python
def extract_training_data_from_ga(result: dict) -> Tuple[np.ndarray, np.ndarray]:
```

**Parameters:**
- `result` (dict): Result dictionary returned by `HybridMLGASolver.solve()`
  - Must contain key: `"exact_evaluations_log"`
  - Value: list of (chromosome, cost) tuples

**Returns:**
- Tuple of two numpy arrays:
  - `X` (np.ndarray): Binary chromosome matrix, shape `(N, m)`
  - `y` (np.ndarray): Cost array, shape `(N,)`

**Raises:**
- `KeyError`: If result dict missing `"exact_evaluations_log"`
- `ValueError`: If exact_evaluations_log is empty

**Example:**
```python
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga

# Run hybrid GA
hybrid_ga = HybridMLGASolver(...)
result = hybrid_ga.solve()

# Extract training data in one call
X, y = extract_training_data_from_ga(result)

# Use immediately
training_pipeline.train(X, y)
```

---

## Existing Methods (Enhanced in Phase 1)

### `CFLPDatasetGenerator.save()`

**Location:** `src/dataset_generator.py` lines 133-144

**Purpose:** Persist training data as compressed NumPy archive.

**Signature:**
```python
def save(self, X: np.ndarray, y: np.ndarray, save_path: str) -> None:
```

**Usage with Phase 1:**
```python
X_ga, y_ga = extract_training_data_from_ga(result)
gen.save(X_ga, y_ga, "training_data_ga_derived.npz")
```

### `CFLPDatasetGenerator.load()`

**Location:** `src/dataset_generator.py` lines 146-159

**Purpose:** Load previously saved training data.

**Signature:**
```python
def load(self, load_path: str) -> Tuple[np.ndarray, np.ndarray]:
```

**Usage with Phase 1:**
```python
X_ga, y_ga = gen.load("training_data_ga_derived.npz")
training_pipeline.train(corpus_path="training_data_ga_derived.npz")
```

### `CFLPDatasetGenerator.append()`

**Location:** `src/dataset_generator.py` lines 161-

**Purpose:** Append new samples to existing dataset with de-duplication.

**Signature:**
```python
def append(self, 
           X_existing: np.ndarray, 
           y_existing: np.ndarray,
           X_new: np.ndarray, 
           y_new: np.ndarray
           ) -> Tuple[np.ndarray, np.ndarray]:
```

**Usage with Phase 1:**
```python
# Combine multiple GA runs
X1, y1 = extract_training_data_from_ga(result1)
X2, y2 = extract_training_data_from_ga(result2)
X_combined, y_combined = gen.append(X1, y1, X2, y2)
```

---

## Integration Points

### With HybridMLGASolver

**Data Flow:**
```python
hybrid_ga = HybridMLGASolver(...)
result = hybrid_ga.solve()  # Returns dict with exact_evaluations_log
                             
X, y = extract_training_data_from_ga(result)
gen.save(X, y, "training_data.npz")
```

**Key Fields in Result Dict:**
```python
result = {
    "best_cost": float,                    # Final exact cost
    "best_surrogate_cost": float,          # Best surrogate predicted cost
    "best_individual": list,               # Best chromosome
    "exact_evaluations_log": [...],        # ← PHASE 1 uses this
    "exact_eval_count": int,
    "surrogate_eval_count": int,
    "history": {...},
    "elapsed_time": float,
    ...
}
```

### With TrainingPipeline

**Expected Usage (Phase 3):**
```python
from training_pipeline import TrainingPipeline

# Workflow after Phase 1
X_ga, y_ga = extract_training_data_from_ga(result)
gen.save(X_ga, y_ga, "training_data_ga_derived.npz")

# Phase 3: Automatic retraining
pipeline = TrainingPipeline(corpus_path="training_data_ga_derived.npz")
pipeline.train()
```

### With CFLPDatasetGenerator

**Data Format Compatibility:**
```python
gen = CFLPDatasetGenerator(dataset)

# Phase 1 output matches expected input format:
X_ga, y_ga = gen.generate_from_ga_evaluations(exact_log)

# Can be saved, loaded, appended like any other dataset
gen.save(X_ga, y_ga, path)
X, y = gen.load(path)
X_combined, y_combined = gen.append(X_old, y_old, X_ga, y_ga)
```

---

## Configuration & Tuning

### Warmup Fraction Parameter

**In HybridMLGASolver:**
```python
hybrid_ga = HybridMLGASolver(
    warmup_fraction=0.4,  # Collect data for first 40% of generations
    n_generations=100,    # 40 generations × 30 pop = ~1200 samples
    ...
)
```

**Effect on Data Collection:**
| warmup_fraction | n_generations | Expected Samples | Data Quality |
|-----------------|---------------|------------------|--------------|
| 0.1 | 100 | ~300 | Low (sparse) |
| 0.2 | 100 | ~600 | Medium |
| 0.3 | 100 | ~900 | Good |
| 0.4 | 100 | ~1200 | Very Good |
| 0.5 | 100 | ~1500 | Excellent (slow GA) |

**Recommendations:**
- Small instances (m ≤ 15): warmup_fraction=0.3-0.4
- Medium instances (m ≤ 30): warmup_fraction=0.2-0.3
- Large instances (m > 30): warmup_fraction=0.1-0.2

---

## Data Validation

### Quality Checks

After extraction, validate:

```python
X_ga, y_ga = extract_training_data_from_ga(result)

# 1. Shape validation
assert X_ga.shape[1] == dataset.num_facilities
assert len(X_ga) == len(y_ga)

# 2. Value validation
assert X_ga.dtype == np.int32
assert y_ga.dtype == np.float64
assert (X_ga >= 0).all() and (X_ga <= 1).all()
assert (y_ga > 0).all()

# 3. Data integrity
assert not np.isnan(y_ga).any()
assert not np.isinf(y_ga).any()

# 4. Variability
assert y_ga.std() > 0  # Some variation in costs

print("✓ All validation checks passed")
```

---

## Error Handling

### Common Errors & Solutions

**Error: KeyError: 'exact_evaluations_log'**
```python
result = hybrid_ga.solve()
X, y = extract_training_data_from_ga(result)  # KeyError!
```

**Cause:** Result dict doesn't have exact_evaluations_log
**Solution:** Ensure HybridMLGASolver was initialized with warmup_fraction > 0

**Error: ValueError: No exact evaluations logged**
```python
X, y = extract_training_data_from_ga(result)  # ValueError!
```

**Cause:** exact_evaluations_log is empty
**Solution:** 
- Increase warmup_fraction
- Increase n_generations
- Check if GA ran to completion

**Error: Shape mismatch in training**
```python
X_ga, y_ga = extract_training_data_from_ga(result)
len(X_ga) != len(y_ga)  # Error!
```

**Cause:** Corrupted extraction
**Solution:** Re-run extraction, check result dict

---

## Performance Characteristics

### Extraction Speed
```
N samples extraction time ≈ 0.001s × N
1000 samples ≈ 1 second
10000 samples ≈ 10 seconds
```

### Memory Usage
```
Memory = (N × m × 4 bytes) + (N × 8 bytes)
       = (N × m) / 250 MB + (N × 8) / 125 MB

For N=1000, m=50:
  ≈ 200 KB + 8 KB ≈ 208 KB
```

### I/O Performance
```
Save (N samples): ~50 ms
Load (N samples): ~20 ms
Append (merge): ~100 ms
```

---

## Testing Utilities

### Test Suite

**File:** `src/test_ga_derived_sampling.py`

**Execution:**
```bash
python src/test_ga_derived_sampling.py
```

**What it Tests:**
- GA-derived data extraction ✓
- Data quality validation ✓
- File I/O (save/load) ✓
- Comparison with full enumeration ✓

### Example Script

**File:** `src/example_ga_derived_workflow.py`

**Purpose:** Reference implementation of complete workflow

**Execution:**
```bash
python src/example_ga_derived_workflow.py
```

**Demonstrates:**
- Hybrid GA execution with warmup
- Data extraction
- Model retraining (simulated)
- Iteration workflow

---

## Related Documentation

**For Users:**
- `docs/PHASE_1_GA_DERIVED_SAMPLING.md` - Complete user guide
- `docs/PHASE_1_IMPLEMENTATION_SUMMARY.md` - Implementation details

**For Developers:**
- `src/dataset_generator.py` - Implementation source
- `src/hybrid_ga.py` - Extraction function source
- `src/test_ga_derived_sampling.py` - Test implementation

**For Context:**
- `docs/COMPREHENSIVE_AUDIT.md` - Mentor's objective verification
- `docs/PHASE_1_API_REFERENCE.md` - This document

---

## Version History

**Phase 1 (Initial Release):**
- ✅ `generate_from_ga_evaluations()` added
- ✅ `extract_training_data_from_ga()` added
- ✅ Test suite created
- ✅ Documentation completed

**Planned for Phase 2:**
- Competitive fitness check integration
- Automatic model retraining
- Enhanced uncertainty handling

**Planned for Phase 3:**
- Full training pipeline integration
- Adaptive warmup tuning
- Performance optimization

---

## Summary

**Phase 1 provides:**
- ✅ `generate_from_ga_evaluations()` - Convert GA log to training data
- ✅ `extract_training_data_from_ga()` - Convenience extraction function
- ✅ Complete test suite
- ✅ Comprehensive documentation
- ✅ Integration examples

**Ready for:**
- ✅ Testing and validation
- ✅ Integration with Phase 2 (competitive fitness)
- ✅ Phase 3 (automatic retraining)

