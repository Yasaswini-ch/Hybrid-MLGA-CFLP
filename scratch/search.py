import os
import glob

print("Starting search...")
files = glob.glob('docs/*.md') + glob.glob('docs/*.csv') + glob.glob('src/*.py')
print(f"Found {len(files)} files to check.")
for f in files:
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            if 'cap71' in content.lower():
                print(f"Found 'cap71' in: {f}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
print("Search done.")
