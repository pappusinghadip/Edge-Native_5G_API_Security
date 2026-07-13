"""Standard IDS baselines requested in review — XGBoost, Random Forest, MLP.

Trained on the same 10 flow features and evaluated on the same held-out test set,
with the decision threshold tuned on the validation split (identical protocol to the
1D-CNN / GBM / ensemble comparison). Results append to results/metrics/extra_baselines.json.

Run: python -m scripts.extra_baselines
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.models.ensemble import _flat, best_threshold, scored_metrics
from src.models.io import load_processed_split
from src.utils.config import load_yaml, resolve_path
from src.utils.seed import set_global_seed


def main() -> None:
    set_global_seed(42)
    paths_cfg = load_yaml("configs/paths.yaml")
    processed = resolve_path(paths_cfg["data"]["processed"])
    Xtr, ytr, _ = load_processed_split(processed / "train.npz")
    Xva, yva, _ = load_processed_split(processed / "val.npz")
    Xte, yte, _ = load_processed_split(processed / "test.npz")
    Xtr, Xva, Xte = _flat(Xtr), _flat(Xva), _flat(Xte)

    models = {
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 subsample=0.9, n_jobs=-1, eval_metric="logloss", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=40, early_stopping=True, random_state=42),
    }

    results = []
    for name, clf in models.items():
        t0 = time.perf_counter()
        clf.fit(Xtr, ytr)
        train_s = time.perf_counter() - t0
        val_score = clf.predict_proba(Xva)[:, 1]
        test_score = clf.predict_proba(Xte)[:, 1]
        m = scored_metrics(name, yte, test_score, best_threshold(yva, val_score))
        m["train_seconds"] = round(train_s, 1)
        results.append(m)
        print(f"{name:14s} AUC={m['auc_roc']:.4f} F1={m['f1_binary']:.4f} FPR={m['false_positive_rate']*100:.2f}%  ({train_s:.0f}s)")

    out = resolve_path(paths_cfg["results"]["metrics"]) / "extra_baselines.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
