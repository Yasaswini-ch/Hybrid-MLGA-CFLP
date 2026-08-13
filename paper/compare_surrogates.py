"""
compare_surrogates.py
======================
Reuses the already-cached bootstrap corpora (data/processed/convplot_corpus_*.npz,
produced by generate_all_convergence.py) to train and compare all four surrogate
architectures (Random Forest, Gradient Boosting, XGBoost, MLP) on every one of the
40 OR-Library instances, without re-running any GA search. Only the training_pipeline
train/test split + fit + evaluate step is repeated per architecture.
"""
import os
import sys
import csv
import time

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from parser import CFLPDataset
from training_pipeline import SurrogateTrainingPipeline

_ROOT_DIR = os.path.normpath(os.path.join(_SRC_DIR, ".."))
RAW_DIR = os.path.join(_ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(_ROOT_DIR, "data", "processed")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

INSTANCES = [
    "cap41", "cap42", "cap43", "cap44",
    "cap51",
    "cap61", "cap62", "cap63", "cap64",
    "cap71", "cap72", "cap73", "cap74",
    "cap81", "cap82", "cap83", "cap84",
    "cap91", "cap92", "cap93", "cap94",
    "cap101", "cap102", "cap103", "cap104",
    "cap111", "cap112", "cap113", "cap114",
    "cap121", "cap122", "cap123", "cap124",
    "cap131", "cap132", "cap133", "cap134",
    "capa4", "capb4", "capc4",
]

MODEL_TYPES = ("random_forest", "gradient_boosting", "xgboost", "mlp")

rows = []
_instances_to_run = sys.argv[1:] if len(sys.argv) > 1 else INSTANCES
for idx, name in enumerate(_instances_to_run):
    corpus_path = os.path.join(PROCESSED_DIR, f"convplot_corpus_{name}.npz")
    if not os.path.exists(corpus_path):
        print(f"[SKIP] {name}: no cached corpus at {corpus_path}")
        continue

    dataset = CFLPDataset(os.path.join(RAW_DIR, f"{name}.txt"))
    print(f"\n[{idx+1}/{len(INSTANCES)}] === {name} ===", flush=True)

    pipeline = SurrogateTrainingPipeline(dataset=dataset, corpus_path=corpus_path, model_save_dir=PROCESSED_DIR)
    t0 = time.time()
    try:
        results = pipeline.run(model_types=MODEL_TYPES)
    except Exception as e:
        print(f"[ERROR] {name}: {e}", flush=True)
        continue
    elapsed = time.time() - t0

    for mtype in MODEL_TYPES:
        if mtype not in results:
            continue
        m = results[mtype]["metrics"]
        lat = results[mtype]["latency"]
        rows.append({
            "instance": name,
            "model_type": mtype,
            "r2": m["r2"],
            "mape_pct": m["mape_pct"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "speedup_factor": lat["speedup_factor"],
            "train_time_sec": results[mtype]["train_time_sec"],
        })
    print(f"  done in {elapsed:.1f}s", flush=True)

    # incremental save
    csv_path = os.path.join(OUT_DIR, "surrogate_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

print(f"\nALL DONE. Saved {len(rows)} rows to surrogate_comparison.csv", flush=True)
