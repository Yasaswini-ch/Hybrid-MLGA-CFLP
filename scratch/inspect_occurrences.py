with open("scratch/check_formulation.py", "r") as f:
    code_check = f.read()

with open("scratch/compare_datasets.py", "r") as f:
    code_compare = f.read()

print("=== check_formulation.py ===")
print(code_check)
print("\n=== compare_datasets.py ===")
print(code_compare)
