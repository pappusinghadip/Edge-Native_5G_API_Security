"""Round-5 review, Table 4.1: deployment cost of every detector in the comparison.

The four single-run baselines were never saved, so they are retrained here exactly as in
scripts/best_ensemble.py (same settings, seed 42) and must reproduce the Table 4.1 ROC-AUC before
anything is measured. Then, on the Mac, for every row of Table 4.1:
  size     serialized bytes: pickle for scikit-learn and XGBoost, the TFLite file for the 1D-CNN;
           an ensemble's size is the sum of the models it must run
  latency  single-flow inference, 1,000 warm-up and 10,000 timed calls with perf_counter_ns, as in
           Section 4.6; TFLite uses two threads, the other models predict one row on one thread
  memory   resident memory added by loading the model and running those calls, measured in a fresh
           process in which every library is already imported
Run: PYTHONPATH=. .venv/bin/python scripts/deployment_costs.py
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

M, OUT = Path("results/metrics"), Path("results/models/baselines")
WARMUP, RUNS = 1_000, 10_000
RATE_IDX, SYN_IDX = 3, 5                     # Rate and syn_count in the processed feature order
BASE = ["XGBoost", "Gradient Boosting", "Random Forest", "MLP"]
FILES = {"XGBoost": "xgboost.pkl", "Gradient Boosting": "hist_gb.pkl", "Random Forest": "random_forest.pkl",
         "MLP": "mlp.pkl", "meta": "stacking_meta.pkl", "rate": "rate_rule.pkl",
         "cnn_cen": "../centralized_best.tflite", "cnn_fed": "federated_cnn.tflite"}


def flat(x):
    return x.reshape(len(x), -1).astype(np.float32)


def load(split):
    d = np.load(f"data/processed/{split}.npz")
    return d["X"], d["y"].reshape(-1) if d["y"].ndim == 1 else d["y"].argmax(1)


# ------------------------------------------------------------------ training (parent process)
def train() -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    import tensorflow as tf
    from src.models.io import load_processed_split
    from src.utils.seed import set_global_seed

    set_global_seed(42)
    X_tr, y_tr, _ = load_processed_split(Path("data/processed/train.npz"))
    X_va, y_va, _ = load_processed_split(Path("data/processed/val.npz"))
    X_te, y_te, _ = load_processed_split(Path("data/processed/test.npz"))
    Ftr, Fva, Fte = flat(X_tr), flat(X_va), flat(X_te)
    OUT.mkdir(parents=True, exist_ok=True)

    cnn = tf.keras.models.load_model("results/models/centralized_best.h5", compile=False)
    val_p = {"1D-CNN": cnn.predict(X_va, batch_size=1024, verbose=0)[:, 1]}
    ext = json.loads((M / "extended_metrics.json").read_text())
    check = {}
    for name, clf in {
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9,
                                 n_jobs=-1, eval_metric="logloss", random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True, random_state=42),
    }.items():
        t0 = time.time(); clf.fit(Ftr, y_tr)
        val_p[name] = clf.predict_proba(Fva)[:, 1]
        auc = float(roc_auc_score(y_te, clf.predict_proba(Fte)[:, 1]))
        check[name] = {"roc_auc": round(auc, 4), "table_4_1": round(ext[name]["auc_roc"], 4),
                       "train_seconds": round(time.time() - t0, 1)}
        print(name, check[name], flush=True)
        (OUT / FILES[name]).write_bytes(pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL))
        del clf
    names = ["1D-CNN"] + BASE                    # column order of scripts/best_ensemble.py
    meta = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        np.column_stack([val_p[n] for n in names]), y_va)
    (OUT / FILES["meta"]).write_bytes(pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
    rate = json.loads((M / "baseline_ratelimiter.json").read_text())
    (OUT / FILES["rate"]).write_bytes(pickle.dumps({"features": [RATE_IDX, SYN_IDX],
                                                    "threshold": rate["threshold"]}))

    # the federated global model, converted exactly as src/models/export.py converts the centralized one
    fed = tf.keras.models.load_model("results/models/fl_fedavg_iid_global.h5", compile=False)
    conv = tf.lite.TFLiteConverter.from_keras_model(fed)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    (OUT / FILES["cnn_fed"]).write_bytes(conv.convert())
    return check


# ------------------------------------------------------------------ measurement (child process)
def measure(row: str) -> dict:
    import psutil
    import sklearn.ensemble, sklearn.linear_model, sklearn.neural_network  # noqa: E401,F401
    import tensorflow as tf
    import xgboost  # noqa: F401

    X = flat(load("test")[0][: WARMUP + RUNS]).copy()
    rows = [X[i:i + 1] for i in range(len(X))]
    times = np.empty(RUNS, dtype=np.int64)
    weights = next(e["weights"] for e in json.loads((M / "best_ensemble.json").read_text())["ensembles"]
                   if e.get("weights"))
    proc = psutil.Process()
    gc.collect()
    rss0 = proc.memory_info().rss

    def pk(name):
        with open(OUT / FILES[name], "rb") as fh:          # streamed, so no file-sized buffer stays resident
            m = pickle.load(fh)
        if hasattr(m, "n_jobs"):
            m.n_jobs = 1
        return m

    def tfl(key):
        it = tf.lite.Interpreter(model_path=str(OUT / FILES[key]), num_threads=2); it.allocate_tensors()
        i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
        def f(x):
            it.set_tensor(i, x.reshape(1, -1, 1)); it.invoke(); return float(it.get_tensor(o)[0, 1])
        return f

    def base(name):
        if name == "1D-CNN":
            return tfl("cnn_cen")
        m = pk(name)
        if name == "XGBoost":
            b = m.get_booster(); b.set_param({"nthread": 1})
            return lambda x: float(b.inplace_predict(x)[0])
        return lambda x: float(m.predict_proba(x)[0, 1])

    if row in ("cnn_cen", "cnn_fed"):
        f, parts = tfl(row), [row]
    elif row == "rate":
        with open(OUT / FILES["rate"], "rb") as fh:
            r = pickle.load(fh)
        a, s, t = *r["features"], r["threshold"]
        f, parts = (lambda x: float(max(x[0, a], x[0, s]) > t)), ["rate"]
    elif row == "Weighted":
        used = [(n, w) for n, w in weights.items() if w > 0]
        fs = [(base(n), w) for n, w in used]
        f, parts = (lambda x: sum(w * g(x) for g, w in fs)), [n for n, _ in used]
    elif row == "Stacked":
        names = ["1D-CNN"] + BASE
        fs, meta = [base(n) for n in names], pk("meta")
        f = lambda x: float(meta.predict_proba(np.array([[g(x) for g in fs]]))[0, 1])  # noqa: E731
        parts = ["cnn_cen"] + BASE + ["meta"]
    else:
        f, parts = base(row), [row]

    for x in rows[:WARMUP]:
        f(x)
    for k, x in enumerate(rows[WARMUP:]):
        t0 = time.perf_counter_ns(); f(x); times[k] = time.perf_counter_ns() - t0
    gc.collect()
    rss1 = proc.memory_info().rss
    ms = times / 1e6
    size = sum((OUT / FILES[p]).stat().st_size for p in parts)
    return {"parts": parts, "size_bytes": int(size), "mean_ms": float(ms.mean()),
            "median_ms": float(np.median(ms)), "p99_ms": float(np.percentile(ms, 99)),
            "max_ms": float(ms.max()), "memory_mb": round((rss1 - rss0) / 1e6, 2)}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--measure":
        print("RESULT " + json.dumps(measure(sys.argv[2])), flush=True)
        sys.exit(0)
    args = sys.argv[1:]
    sessions = int(args[args.index("--sessions") + 1]) if "--sessions" in args else 1
    prev = json.loads((M / "deployment_costs.json").read_text()) if (M / "deployment_costs.json").exists() else {}
    if "--measure-only" in args:                  # reuse the saved models and their reproduction check
        check = prev["reproduction"]
    else:
        check = train()
    bad = {n: c for n, c in check.items() if abs(c["roc_auc"] - c["table_4_1"]) > 5e-4}
    if bad:
        sys.exit(f"retrained baselines do not reproduce Table 4.1: {bad}")
    out = {"protocol": {"warmup": WARMUP, "runs": RUNS, "timer": "perf_counter_ns", "host": "Apple M4 Pro, macOS",
                        "threads": "TFLite 2; other models one thread per row", "input": "one test-set flow per call",
                        "memory": "RSS after load and timed calls minus RSS before load, fresh process, "
                                  "libraries already imported",
                        "size": "pickle (scikit-learn, XGBoost) or TFLite file; ensembles sum their parts",
                        "sessions": sessions,
                        "aggregation": "each row reports the median over sessions; *_range gives min and max"},
           "reproduction": check, "rows": {}}
    for k in ("first_session_p99_ms", "first_session_note"):      # provenance of the earlier sessions
        if k in prev:
            out[k] = prev[k]
    ROWS = ["Weighted", "XGBoost", "Stacked", "Gradient Boosting", "Random Forest", "cnn_cen", "cnn_fed", "MLP", "rate"]
    per = {row: [] for row in ROWS}
    for sess in range(sessions):
        for row in ROWS:
            r = subprocess.run([sys.executable, __file__, "--measure", row], capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": "."})
            line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line is None:
                sys.exit(f"{row} failed:\n{r.stderr[-2000:]}")
            per[row].append(json.loads(line[7:]))
            print(f"session {sess + 1} {row} p99 {per[row][-1]['p99_ms']:.4f} ms", flush=True)
    med = lambda v: float(np.median(v))
    for row, runs in per.items():
        out["rows"][row] = {"parts": runs[0]["parts"], "size_bytes": runs[0]["size_bytes"],
                            **{k: med([x[k] for x in runs]) for k in ("mean_ms", "median_ms", "p99_ms", "max_ms", "memory_mb")},
                            "p99_ms_range": [min(x["p99_ms"] for x in runs), max(x["p99_ms"] for x in runs)],
                            "memory_mb_range": [min(x["memory_mb"] for x in runs), max(x["memory_mb"] for x in runs)],
                            "sessions": runs}
    (M / "deployment_costs.json").write_text(json.dumps(out, indent=1))
    print("wrote", M / "deployment_costs.json")
