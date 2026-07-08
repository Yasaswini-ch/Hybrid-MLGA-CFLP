import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('docs/project_journal.md', 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line_lower = line.lower()
        if any(term in line_lower for term in ['capa', 'capb', 'capc']):
            if any(k in line_lower for k in ['optimal', 'table', '|', 'beasley', 'large']):
                print(f"Line {line_num}: {line.strip()}")
