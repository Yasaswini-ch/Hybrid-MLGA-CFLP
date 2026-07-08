import os
import glob
import re
from parser import CFLPDataset

def main():
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    
    # Dynamically find all cap*.txt files in raw_dir
    file_pattern = os.path.join(raw_dir, "cap*.txt")
    raw_files = glob.glob(file_pattern)
    
    # Helper to sort files numerically (e.g. cap41 before cap100)
    def numerical_sort_key(file_path):
        filename = os.path.basename(file_path)
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0
        
    raw_files.sort(key=numerical_sort_key)
    files = [os.path.basename(f) for f in raw_files]
    
    print("=" * 60)
    print("CFLP PARSER BATCH VERIFICATION RUN")
    print("=" * 60)
    
    summaries = []
    
    for filename in files:
        file_path = os.path.join(raw_dir, filename)
        if not os.path.exists(file_path):
            print(f"[Error] File not found: {file_path}")
            continue
            
        try:
            dataset = CFLPDataset(file_path)
            summary = dataset.get_summary()
            summaries.append(summary)
            print(f"Successfully parsed: {filename}")
            print(f"  - Facilities: {dataset.num_facilities}, Customers: {dataset.num_customers}")
            print(f"  - Total Demand: {summary['total_demand']:.2f}, Total Capacity: {summary['total_capacity']:.2f}")
            print(f"  - Capacity/Demand Ratio: {summary['capacity_demand_ratio']:.4f}")
            print(f"  - Fixed Cost Range: ${summary['min_fixed_cost']:.2f} - ${summary['max_fixed_cost']:.2f}")
            print("-" * 60)
        except Exception as e:
            print(f"[Error] Failed parsing {filename}: {e}")
            
    # Print a markdown table for easy inclusion in research notes
    print("\nBatch Summary Markdown Table (for project journals):")
    print("| Dataset | Facilities (m) | Customers (n) | Total Demand | Total Capacity | Cap/Dem Ratio | Fixed Cost (Standard) | Special Fixed Cost (idx=10) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in summaries:
        # Note: Beasley fixed costs are uniform except facility 11 (index 10) which is 0.0
        # Let's extract standard fixed cost (e.g., from first facility)
        file_path = os.path.join(raw_dir, s['name'] + ".txt")
        dataset = CFLPDataset(file_path)
        std_cost = dataset.fixed_costs[0] if len(dataset.fixed_costs) > 0 else 0.0
        sp_cost = dataset.fixed_costs[10] if len(dataset.fixed_costs) > 10 else 0.0
        
        print(f"| {s['name']} | {s['facilities']} | {s['customers']} | {s['total_demand']:.0f} | {s['total_capacity']:.0f} | {s['capacity_demand_ratio']:.4f} | ${std_cost:,.1f} | ${sp_cost:,.1f} |")
        
if __name__ == "__main__":
    main()
