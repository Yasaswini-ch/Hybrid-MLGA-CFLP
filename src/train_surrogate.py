import os
import itertools
import pickle
import time
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from parser import CFLPDataset

class SurrogateTrainer:
    """
    Generates the complete feasible dataset universe for a CFLP problem,
    solves each configuration exactly in-memory, and trains a Scikit-Learn
    Random Forest surrogate model to predict transportation costs.
    """
    
    def __init__(self, dataset: CFLPDataset):
        self.dataset = dataset
        self.num_facilities = dataset.num_facilities
        self.num_customers = dataset.num_customers
        self.total_demand = np.sum(dataset.demands)
        
        # Determine minimum facilities required for physical capacity feasibility
        self.min_facilities_needed = int(np.ceil(self.total_demand / dataset.capacities[0]))
        
    def generate_all_feasible_configs(self) -> List[List[int]]:
        """
        Mathematically generates all unique binary facility configurations that
        satisfy the capacity feasibility condition (at least min_facilities_needed open).
        
        Returns:
            List[List[int]]: List of all feasible binary configurations.
        """
        configs = []
        # Generate combinations for each count of open facilities from min_facilities_needed to m
        for num_open in range(self.min_facilities_needed, self.num_facilities + 1):
            for indices in itertools.combinations(range(self.num_facilities), num_open):
                vector = [0] * self.num_facilities
                for idx in indices:
                    vector[idx] = 1
                configs.append(vector)
        return configs

    def solve_transport_lp(self, y_val: np.ndarray) -> float:
        """
        Solves the continuous transportation sub-problem in-memory using SciPy highs solver.
        """
        open_indices = np.where(y_val == 1)[0]
        num_open = len(open_indices)
        
        # 1. Objective: Flattened unit transport costs
        c = []
        for j in range(self.num_customers):
            for i in open_indices:
                c.append(self.dataset.transport_costs[j, i])
        c = np.array(c)
        
        # 2. Equality Constraints: Demands satisfied
        A_eq = np.zeros((self.num_customers, self.num_customers * num_open))
        for j in range(self.num_customers):
            A_eq[j, j * num_open : (j + 1) * num_open] = 1.0
        b_eq = self.dataset.demands
        
        # 3. Inequality Constraints: Capacity respected
        A_ub = np.zeros((num_open, self.num_customers * num_open))
        for k in range(num_open):
            for j in range(self.num_customers):
                A_ub[k, j * num_open + k] = 1.0
        b_ub = self.dataset.capacities[open_indices]
        
        bounds = [(0.0, None)] * len(c)
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if not res.success:
            raise ValueError("LP Solver failed to find optimal routing for a feasible configuration.")
            
        return res.fun

    def build_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates and solves the complete feasible configuration space.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature matrix X and target costs array y.
        """
        print("[Step 1] Mathematically identifying all feasible configurations...")
        configs = self.generate_all_feasible_configs()
        total_configs = len(configs)
        print(f"  Found exactly {total_configs} feasible configurations in the entire search space universe.")
        
        X = np.array(configs)
        y = np.zeros(total_configs)
        
        print(f"[Step 2] Solving continuous transportation LP for all {total_configs} configurations...")
        start_time = time.time()
        for idx in range(total_configs):
            y[idx] = self.solve_transport_lp(X[idx])
            if (idx + 1) % 500 == 0 or idx == total_configs - 1:
                print(f"  Processed {idx + 1}/{total_configs} configurations...")
                
        elapsed = time.time() - start_time
        print(f"Dataset generated! Time elapsed: {elapsed:.2f} seconds ({elapsed/total_configs*1000:.2f} ms per sample).")
        
        return X, y

    def train_model(self, X: np.ndarray, y: np.ndarray, model_save_path: str) -> RandomForestRegressor:
        """
        Splits the dataset, trains the Random Forest regressor, evaluates accuracy,
        and serializes the model.
        """
        print("\n[Step 3] Splitting dataset and training surrogate model...")
        # 80/20 train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"  Training set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        start_time = time.time()
        rf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100.0
        
        print("=" * 60)
        print(f"{'SURROGATE MODEL ACCURACY METRICS':^60}")
        print("=" * 60)
        print(f"  R2 Score (Variance Explained)  : {r2:.6f}")
        print(f"  Mean Absolute Error (MAE)      : ${mae:,.2f}")
        print(f"  Mean Absolute Percentage (MAPE): {mape:.4f}%")
        print(f"  Model Training Time            : {train_time:.2f} seconds")
        print("=" * 60)
        
        # Save model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, 'wb') as file:
            pickle.dump(rf, file)
        print(f"Surrogate model pickled successfully to: {model_save_path}")
        
        return rf


def main():
    base_dir = os.path.dirname(__file__)
    raw_path = os.path.join(base_dir, "..", "data", "raw", "cap41.txt")
    model_path = os.path.join(base_dir, "..", "data", "processed", "surrogate_rf.pkl")
    dataset_npy_path = os.path.join(base_dir, "..", "data", "processed", "cflp_dataset.npz")
    
    print("=" * 80)
    print("SURROGATE AI MODEL DATA GENERATION & TRAINING")
    print("=" * 80)
    
    # Load dataset
    dataset = CFLPDataset(raw_path)
    trainer = SurrogateTrainer(dataset)
    
    # Build complete dataset universe
    X, y = trainer.build_dataset()
    
    # Save raw arrays
    np.savez(dataset_npy_path, X=X, y=y)
    print(f"Complete dataset universe saved to: {dataset_npy_path}")
    
    # Train and serialize the model
    trainer.train_model(X, y, model_path)

if __name__ == "__main__":
    main()
