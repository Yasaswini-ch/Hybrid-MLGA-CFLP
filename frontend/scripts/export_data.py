"""Export benchmark CSVs + the cap71 instance to JSON for the static frontend.
Run from repo root:  python frontend/scripts/export_data.py
"""
import csv, json, os
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "frontend", "src", "data")

def rows(p):
    with open(os.path.join(ROOT, p), newline="") as f:
        return list(csv.DictReader(f))

def num(v):
    try: return float(v)
    except Exception: return v

hyb = {r["Instance"]: {k: num(v) for k, v in r.items()} for r in rows("docs/hybrid_benchmark_results.csv")}
cla = {r["Instance"]: {k: num(v) for k, v in r.items()} for r in rows("docs/statistical_benchmark_results.csv")}
large = [{k: num(v) for k, v in r.items()} for r in rows("docs/large_benchmark_results.csv")]

instances = []
for name in hyb:
    h, c = hyb[name], cla.get(name, {})
    instances.append({
        "name": name, "optimal": h["Optimal"],
        "classical": {"best": c.get("Best"), "avg": c.get("Average"), "avgGap": c.get("Avg Gap (%)"), "bestGap": c.get("Best Gap (%)"), "time": c.get("Total Time (s)")},
        "hybrid": {"best": h["Best"], "avg": h["Average"], "avgGap": h["Avg Gap (%)"], "bestGap": h["Best Gap (%)"], "time": h["Total Solve Time (s)"],
                   "r2": h["Surrogate R2"], "exact": h["Avg Exact Evals"], "surrogate": h["Avg Surrogate Evals"]},
    })

# cap71 instance for the in-browser demo
tok = open(os.path.join(ROOT, "data", "raw", "cap71.txt")).read().split()
m, n = int(tok[0]), int(tok[1]); p = 2
cap, fixed = [], []
for _ in range(m):
    cap.append(float(tok[p])); fixed.append(float(tok[p + 1])); p += 2
dem, cost = [], []
for _ in range(n):
    dem.append(float(tok[p])); p += 1
    cost.append([float(x) for x in tok[p:p + m]]); p += m
cap71 = {"name": "cap71", "m": m, "n": n, "cap": cap, "fixed": fixed, "dem": dem, "cost": cost,
         "optimal": hyb["cap71"]["Optimal"]}

json.dump({"instances": instances, "large": large}, open(os.path.join(OUT, "results.json"), "w"))
json.dump(cap71, open(os.path.join(OUT, "cap71.json"), "w"))
print(len(instances), "instances,", len(large), "large rows")
