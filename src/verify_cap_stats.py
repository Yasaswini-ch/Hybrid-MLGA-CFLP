import os
from parser import CFLPDataset
import numpy as np

def verify_and_collect_stats():
    raw_dir = "data/raw"
    names = [
        "capa1", "capa2", "capa3", "capa4",
        "capb1", "capb2", "capb3", "capb4",
        "capc1", "capc2", "capc3", "capc4"
    ]
    
    print("| Dataset | Facilities (m) | Customers (n) | Capacity per Facility | Total Demand | Total Capacity | Cap/Dem Ratio | Min Fixed Cost | Max Fixed Cost |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for name in names:
        file_path = os.path.join(raw_dir, f"{name}.txt")
        if not os.path.exists(file_path):
            print(f"| {name} | File Not Found | | | | | | | |")
            continue
            
        dataset = CFLPDataset(file_path)
        summary = dataset.get_summary()
        cap_val = dataset.capacities[0] # All facilities in a set have uniform capacity
        min_fixed = np.min(dataset.fixed_costs)
        max_fixed = np.max(dataset.fixed_costs)
        
        print(f"| {name} | {summary['facilities']} | {summary['customers']} | {cap_val:,.1f} | {summary['total_demand']:,.1f} | {summary['total_capacity']:,.1f} | {summary['capacity_demand_ratio']:.4f} | ${min_fixed:,.1f} | ${max_fixed:,.1f} |")

if __name__ == "__main__":
    verify_and_collect_stats()
