import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('docs/CFLP_Complete_Project_Guide.md', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):
        if any(term in line.lower() for term in ['capa', 'capb', 'capc', 'cap71']):
            if '|' in line or 'table' in line.lower() or 'optimal' in line.lower():
                print(f"Line {idx}: {line.strip()}")
