"""
dataset_generator.py
====================
Generates ML training datasets for surrogate model training by collecting
(chromosome, exact_cost) pairs from two sources:

  1. FULL ENUMERATION (small m only): Exhaustively evaluates all feasible
     binary configurations. Exact but only practical for m ≤ 20.

  2. GA-DERIVED SAMPLING (scalable): Collects training samples from exact LP
     evaluations performed during Classical GA runs. Scales to any m.
     Produces samples concentrated in promising search regions.

Datasets are serialized as compressed NumPy archives (.npz):
  - X: Feature matrix of shape (N, m) — raw binary chromosomes
  - y: Target cost array of shape (N,) — exact LP-solved total transport costs
"""

import os
import time
import itertools
from typing import Tuple
import numpy as np
from scipy.optimize import linprog

from parser import CFLPDataset


class CFLPDatasetGenerator:
    """
    Generates and manages the ML training corpus for CFLP surrogate models.
    """

    def __init__(self, dataset: CFLPDataset):
        """
        Args:
            dataset (CFLPDataset): Parsed CFLP problem instance.
        """
        self.dataset = dataset
        self.m = dataset.num_facilities
        self.n = dataset.num_customers
        self.total_demand = float(np.sum(dataset.demands))

        # Minimum open facilities for physical feasibility
        max_cap = float(np.max(dataset.capacities))
        self.min_open = int(np.ceil(self.total_demand / max_cap))

    # ------------------------------------------------------------------
    # INTERNAL: LP solver (transport cost only, not total cost)
    # ------------------------------------------------------------------
    def _solve_transport_lp(self, y_val: np.ndarray) -> float:
        """
        Solves the continuous transportation sub-problem for a given facility
        opening vector and returns the optimal transport cost.

        Returns float('inf') if the LP fails (infeasible configuration).
        """
        open_indices = np.where(y_val == 1)[0]
        num_open = len(open_indices)
        if num_open == 0:
            return float("inf")

        # Objective: flattened unit transport costs [customer-major order]
        c = []
        for j in range(self.n):
            for i in open_indices:
                c.append(self.dataset.transport_costs[j, i])
        c = np.array(c)

        # Equality: demand satisfaction for each customer
        A_eq = np.zeros((self.n, self.n * num_open))
        for j in range(self.n):
            A_eq[j, j * num_open : (j + 1) * num_open] = 1.0
        b_eq = self.dataset.demands

        # Inequality: capacity bounds for each open facility
        A_ub = np.zeros((num_open, self.n * num_open))
        for k in range(num_open):
            for j in range(self.n):
                A_ub[k, j * num_open + k] = 1.0
        b_ub = self.dataset.capacities[open_indices]

        bounds = [(0.0, None)] * len(c)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")

        return res.fun if res.success else float("inf")

    # ------------------------------------------------------------------
    # PUBLIC: Full Combinatorial Enumeration
    # ------------------------------------------------------------------
    def generate_full_enumeration(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enumerates ALL feasible binary configurations and solves each LP.
        Practical only for m ≤ 20 due to exponential growth.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y_transport)
                X         — binary chromosome matrix, shape (N, m)
                y_transport — LP-optimal transport costs, shape (N,)
        """
        print("[DatasetGenerator] Enumerating all feasible configurations...")
        configs = []
        for num_open in range(self.min_open, self.m + 1):
            for indices in itertools.combinations(range(self.m), num_open):
                vec = [0] * self.m
                for idx in indices:
                    vec[idx] = 1
                configs.append(vec)

        N = len(configs)
        print(f"  Found {N:,} feasible configurations.")

        X = np.array(configs, dtype=np.int32)
        y = np.zeros(N, dtype=np.float64)

        print(f"  Solving {N:,} LP sub-problems...")
        t0 = time.time()
        for i, row in enumerate(X):
            y[i] = self._solve_transport_lp(row)
            if (i + 1) % 500 == 0 or i == N - 1:
                elapsed = time.time() - t0
                rate_ms = (elapsed / (i + 1)) * 1000
                print(f"  [{i+1:5d}/{N}]  {rate_ms:.2f} ms/sample  |  elapsed: {elapsed:.1f}s")

        total_time = time.time() - t0
        print(f"  Done. Total: {total_time:.1f}s  ({total_time/N*1000:.2f} ms/sample)")
        return X, y

    # ------------------------------------------------------------------
    # PUBLIC: GA-Derived Sampling (from GA run evaluations)
    # ------------------------------------------------------------------
    def generate_from_ga_evaluations(self,
                                     exact_evaluations_log: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts exact evaluations collected during GA runs into training dataset.

        CRITICAL FIX: GA collects TOTAL COST (fixed + transport), but this method
        corrects it to TRANSPORT COST ONLY (the format expected by TrainingPipeline).

        Intended workflow:
          1. Run hybrid GA with warmup phase using exact LP evaluations
          2. Collect exact evaluations from initial generations via exact_evaluations_log
          3. Convert collected samples into training dataset via this method
          4. Use this dataset to train ML surrogate for subsequent GA runs

        Args:
            exact_evaluations_log (list): List of (chromosome_list, exact_cost) tuples
                                         collected during GA evolution. Chromosome is a list
                                         of binary values [0/1, 0/1, ...].

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y_transport_cost)
                X               — binary chromosome matrix, shape (N, m)
                y_transport_cost — transport costs only (fixed costs removed), shape (N,)
                                   This format is compatible with TrainingPipeline.
        """
        if not exact_evaluations_log:
            raise ValueError("No evaluations logged. Ensure GA collected exact evaluations.")

        N = len(exact_evaluations_log)
        print(f"[DatasetGenerator] Converting {N:,} GA-collected evaluations to training data...")
        print(f"  Correcting format: GA returns TOTAL COST, converting to TRANSPORT COST...")

        chromosomes = []
        costs_transport = []

        for i, (chromosome_list, cost_total) in enumerate(exact_evaluations_log):
            chromosome_array = np.array(chromosome_list, dtype=np.int32)

            # CRITICAL FIX: Subtract fixed costs to convert from total cost to transport cost
            # GA evaluator returns: total_cost = fixed_cost + transport_cost
            # TrainingPipeline expects: y = transport_cost (then adds fixed costs itself)
            fixed_cost = np.dot(chromosome_array, self.dataset.fixed_costs)
            cost_transport = cost_total - fixed_cost

            chromosomes.append(chromosome_list)
            costs_transport.append(cost_transport)

            if (i + 1) % 500 == 0 or i == N - 1:
                print(f"  Processed {i+1:,}/{N} evaluations")

        X = np.array(chromosomes, dtype=np.int32)
        y = np.array(costs_transport, dtype=np.float64)

        # De-duplicate by chromosome: elitism and population convergence naturally
        # cause the GA to re-evaluate the same chromosome across generations (see
        # diagnose_duplicate_source.py). Duplicate rows add no training signal and
        # risk train/test leakage, so collapse them here using the same row-uniqueness
        # rule already used by append().
        n_before = X.shape[0]
        _, unique_indices = np.unique(X, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        X, y = X[unique_indices], y[unique_indices]
        n_dupes = n_before - X.shape[0]

        print(f"  Converted to training data: {n_before:,} raw evaluations -> "
              f"{X.shape[0]:,} unique samples ({n_dupes:,} duplicate chromosomes removed)")
        print(f"  [OK] Format corrected: Y values are TRANSPORT COSTS (fixed costs removed)")
        print(f"       This data is now compatible with TrainingPipeline")
        return X, y

    # ------------------------------------------------------------------
    # PUBLIC: Save & Load Dataset
    # ------------------------------------------------------------------
    def save(self, X: np.ndarray, y: np.ndarray, save_path: str) -> None:
        """
        Saves (X, y) as a compressed NumPy archive.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target transport cost array.
            save_path (str): Full path to the output .npz file.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, X=X, y=y)
        print(f"  Dataset saved to: {save_path}  ({X.shape[0]:,} samples, {X.shape[1]} features)")

    def load(self, load_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads a previously saved (X, y) dataset from a compressed NumPy archive.

        Args:
            load_path (str): Path to the .npz file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y)
        """
        data = np.load(load_path)
        X, y = data["X"], data["y"]
        print(f"  Loaded dataset: {X.shape[0]:,} samples × {X.shape[1]} features  from: {load_path}")
        return X, y

    def append(self, X_existing: np.ndarray, y_existing: np.ndarray,
               X_new: np.ndarray, y_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Appends new (chromosome, cost) samples to an existing dataset,
        automatically de-duplicating rows using unique row checking.

        Args:
            X_existing, y_existing: Existing dataset.
            X_new, y_new: New samples to add.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Deduplicated combined (X, y).
        """
        X_combined = np.vstack([X_existing, X_new])
        y_combined = np.concatenate([y_existing, y_new])

        # De-duplicate by treating rows as unique keys
        _, unique_indices = np.unique(X_combined, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)

        X_out = X_combined[unique_indices]
        y_out = y_combined[unique_indices]

        added = X_out.shape[0] - X_existing.shape[0]
        print(f"  Appended {len(X_new):,} samples -> {added:,} unique new samples added.")
        print(f"  Combined dataset size: {X_out.shape[0]:,} samples.")
        return X_out, y_out
