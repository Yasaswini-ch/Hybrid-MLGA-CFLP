# CFLP Explained (frontend)

Interactive, lesson-style walkthrough of the Hybrid ML-GA research. Static site (Vite + React), no backend.

```
cd frontend
npm install
npm run dev      # http://localhost:5173
npm run build    # static output in dist/
```

## Data
Regenerate JSON from the benchmark CSVs and `data/raw/cap71.txt` (run from repo root):

```
python frontend/scripts/export_data.py
```

## Notes
- The facility map is schematic: OR-Library instances have no coordinates.
- The GA demo in lesson 3 runs in the browser on cap71 using a fast greedy assignment as fitness; the Python/DEAP solver solves an LP per individual.
- Benchmark tables come straight from `docs/*.csv`.
