"""Re-run the Hybrid ML-GA bootstrap + surrogate training for cap71 (same steps and seeds as
src/benchmark_hybrid_ga.py) and export real predicted-vs-exact points and feature importances.
Run from repo root:  python frontend/scripts/export_surrogate.py
"""
import json, os, random, sys, tempfile
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from parser import CFLPDataset
from hybrid_ga import HybridMLGASolver, extract_training_data_from_ga
from dataset_generator import CFLPDatasetGenerator
from training_pipeline import SurrogateTrainingPipeline

SEED = 42
ds = CFLPDataset(os.path.join(ROOT, "data", "raw", "cap71.txt"))
random.seed(SEED); np.random.seed(SEED)
boot = HybridMLGASolver(dataset=ds, surrogate=None, pop_size=30, n_generations=15, random_seed=SEED).solve()
X, y = extract_training_data_from_ga(boot, dataset=ds)

tmp = tempfile.mkdtemp()
corpus = os.path.join(tmp, "corpus.npz")
CFLPDatasetGenerator(ds).save(X, y, corpus)
res = SurrogateTrainingPipeline(dataset=ds, corpus_path=corpus, model_save_dir=tmp).run(model_types=("random_forest",))
rf = res["random_forest"]
Xt, yt = res["X_test"], res["y_test"]
pred = rf["surrogate"].predict(Xt)
imp = rf["surrogate"].model.feature_importances_
m = ds.num_facilities
names = [f"F{i+1}" for i in range(m)] + ["open count", "total capacity", "slack ratio", "avg fixed cost"]
idx = np.random.RandomState(0).permutation(len(yt))[:250]
out = {
    "samples": int(len(X)), "train": int(len(res["y_train"])), "test": int(len(yt)),
    "r2": float(rf["metrics"]["r2"]), "mape": float(rf["metrics"]["mape_pct"]), "mae": float(rf["metrics"]["mae"]),
    "trees": 200, "features": m + 4,
    "points": [[float(yt[i]), float(pred[i])] for i in idx],
    "importance": sorted([{"name": n, "v": float(v)} for n, v in zip(names, imp)], key=lambda d: -d["v"]),
}
json.dump(out, open(os.path.join(ROOT, "frontend", "src", "data", "surrogate.json"), "w"))
print({k: out[k] for k in ("samples", "train", "test", "r2", "mape", "mae")}, out["importance"][:5])
