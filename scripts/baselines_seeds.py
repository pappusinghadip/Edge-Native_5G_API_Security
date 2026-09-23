"""Round-6 review, Table 4.1: repeat the learned baselines over the five seeds the 1D-CNN uses.

Same settings as scripts/best_ensemble.py, with random_state set to each seed (101-105) instead of 42.
Each model's threshold is tuned on the validation split (maximum F1) and applied once to test.
--data clean runs on data/processed_clean (the round-6 cleaned feature set); --permutation adds the
permutation importance of XGBoost (seed 101) on the test split, scored by ROC-AUC.
Threads follow OMP_NUM_THREADS (default 4) so the run stays cool.

Run: PYTHONPATH=. .venv/bin/python scripts/baselines_seeds.py --data original --permutation
     PYTHONPATH=. .venv/bin/python scripts/baselines_seeds.py --data clean --models XGBoost MLP
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, matthews_corrcoef,
                             roc_auc_score)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.models.ensemble import best_threshold
from src.models.io import load_processed_split
from src.utils.config import load_yaml

M = Path("results/metrics")
NJ = int(os.environ.get("OMP_NUM_THREADS", "4"))
MAKERS = {
    "XGBoost": lambda s: XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9,
                                       n_jobs=NJ, eval_metric="logloss", random_state=s),
    "Gradient Boosting": lambda s: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=s),
    "Random Forest": lambda s: RandomForestClassifier(n_estimators=300, n_jobs=NJ, random_state=s),
    "MLP": lambda s: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True, random_state=s),
}


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    return {"auc_roc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "mcc": float(matthews_corrcoef(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "false_positive_rate": fp / (fp + tn), "threshold": float(t)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["original", "clean"], default="original")
    ap.add_argument("--models", nargs="+", default=list(MAKERS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    ap.add_argument("--permutation", action="store_true")
    a = ap.parse_args()
    cfg = "configs" if a.data == "original" else "configs_clean"
    proc = Path(load_yaml(f"{cfg}/paths.yaml")["data"]["processed"])
    feats = load_yaml(f"{cfg}/model.yaml")["dataset"]["features"]
    flat = lambda X: X.reshape(len(X), -1)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = (load_processed_split(proc / f"{s}.npz") for s in ("train", "val", "test"))
    Xtr, Xva, Xte = flat(Xtr), flat(Xva), flat(Xte)

    out_path = M / f"baselines_seeds_{a.data}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {"data": a.data, "features": feats, "models": {}}
    for name in a.models:
        per = out["models"].get(name, {}).get("per_seed", {})
        for s in a.seeds:
            if str(s) in per:
                continue                                       # resumable: finished seeds are kept
            t0 = time.time()
            clf = MAKERS[name](s).fit(Xtr, ytr)
            thr = best_threshold(yva, clf.predict_proba(Xva)[:, 1])
            per[str(s)] = {**metrics(yte, clf.predict_proba(Xte)[:, 1], thr), "train_seconds": round(time.time() - t0, 1)}
            print(f"{a.data} {name} seed {s}: AUC {per[str(s)]['auc_roc']:.4f} ({per[str(s)]['train_seconds']} s)", flush=True)
            if a.permutation and name == "XGBoost" and s == 101 and "permutation_importance" not in out:
                pi = permutation_importance(clf, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=1)
                out["permutation_importance"] = {"model": "XGBoost seed 101", "split": "test", "scoring": "roc_auc",
                                                 "n_repeats": 5,
                                                 "features": {f: {"mean_drop": float(m), "sd": float(sd)} for f, m, sd
                                                              in zip(feats, pi.importances_mean, pi.importances_std)}}
            summ = {k: {"mean": statistics.mean(v[k] for v in per.values()),
                        "std": statistics.stdev(v[k] for v in per.values()) if len(per) > 1 else 0.0}
                    for k in per[str(s)] if k != "train_seconds"}
            out["models"][name] = {"per_seed": per, "summary": summ, "seeds": sorted(int(x) for x in per)}
            out_path.write_text(json.dumps(out, indent=1))        # saved after every seed
    print("wrote", out_path)


if __name__ == "__main__":
    main()
