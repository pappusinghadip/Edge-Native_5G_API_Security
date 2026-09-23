"""Best-configuration hybrid detector: a held-out stacked ensemble.

Base models (CNN, XGBoost, Gradient Boosting, Random Forest, MLP) are trained on the
training split; their class probabilities on the validation split train a logistic
meta-learner (stacking); everything is evaluated on the untouched test split. A tuned
weighted average is computed as a simpler alternative. This is the strongest honest
detector under the current single-attack subset and forms the culmination of the
baseline comparison requested in review.

Run: python -m scripts.best_ensemble
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.models.ensemble import _flat, best_threshold, scored_metrics
from src.models.io import load_processed_split
from src.utils.config import load_yaml, resolve_path
from src.utils.seed import set_global_seed


def main() -> None:
    set_global_seed(42)
    paths_cfg = load_yaml("configs/paths.yaml")
    model_cfg = load_yaml("configs/model.yaml")
    processed = resolve_path(paths_cfg["data"]["processed"])
    models_dir = resolve_path(paths_cfg["results"]["models"])

    X_tr, y_tr, _ = load_processed_split(processed / "train.npz")
    X_va, y_va, _ = load_processed_split(processed / "val.npz")
    X_te, y_te, _ = load_processed_split(processed / "test.npz")
    Ftr, Fva, Fte = _flat(X_tr), _flat(X_va), _flat(X_te)

    # --- Base models: validation + test probabilities ---
    val_p, test_p = {}, {}

    import tensorflow as tf
    from src.models.cnn import focal_loss
    cnn = tf.keras.models.load_model(models_dir / model_cfg["training"]["checkpoint_name"],
                                     custom_objects={"focal_loss": focal_loss()})
    val_p["1D-CNN"] = cnn.predict(X_va, batch_size=1024, verbose=0)[:, 1]
    test_p["1D-CNN"] = cnn.predict(X_te, batch_size=1024, verbose=0)[:, 1]

    for name, clf in {
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9,
                                 n_jobs=-1, eval_metric="logloss", random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True, random_state=42),
    }.items():
        clf.fit(Ftr, y_tr)
        val_p[name] = clf.predict_proba(Fva)[:, 1]
        test_p[name] = clf.predict_proba(Fte)[:, 1]

    names = list(val_p.keys())
    Vp = np.column_stack([val_p[n] for n in names])
    Tp = np.column_stack([test_p[n] for n in names])

    # --- Base model metrics ---
    results = {"base_models": [], "ensembles": []}
    for n in names:
        results["base_models"].append(scored_metrics(n, y_te, test_p[n], best_threshold(y_va, val_p[n])))

    # --- Stacked ensemble: logistic meta-learner trained on validation probs ---
    meta = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta.fit(Vp, y_va)
    stack_val = meta.predict_proba(Vp)[:, 1]
    stack_test = meta.predict_proba(Tp)[:, 1]
    results["ensembles"].append(scored_metrics("Stacked (logistic meta)", y_te, stack_test, best_threshold(y_va, stack_val)))

    # --- Tuned weighted average (weights optimized for validation AUC) ---
    def neg_val_auc(w):
        w = np.clip(w, 0, None); s = w.sum()
        if s == 0:
            return 0.0
        return -roc_auc_score(y_va, Vp @ (w / s))
    w0 = np.ones(len(names)) / len(names)
    res = minimize(neg_val_auc, w0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-3})
    w = np.clip(res.x, 0, None); w = w / w.sum()
    wavg_test = Tp @ w
    wavg_val = Vp @ w
    m = scored_metrics("Weighted average", y_te, wavg_test, best_threshold(y_va, wavg_val))
    m["weights"] = {n: round(float(wi), 3) for n, wi in zip(names, w)}
    results["ensembles"].append(m)

    out = resolve_path(paths_cfg["results"]["metrics"]) / "best_ensemble.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Keep every model's validation/test scores so the ROC, precision-recall and
    # threshold analyses in Chapter 4 can be rebuilt without retraining.
    np.savez_compressed(
        resolve_path(paths_cfg["results"]["metrics"]) / "detector_probs.npz",
        y_val=y_va, y_test=y_te,
        **{f"val__{n}": val_p[n] for n in names},
        **{f"test__{n}": test_p[n] for n in names},
        val__Stacked=stack_val, test__Stacked=stack_test,
        val__Weighted=wavg_val, test__Weighted=wavg_test,
    )

    print("=== base models (test) ===")
    for r in results["base_models"]:
        print(f"  {r['model']:20s} AUC={r['auc_roc']:.4f} F1={r['f1_binary']:.4f} FPR={r['false_positive_rate']*100:.2f}%")
    print("=== ensembles (test) ===")
    for r in results["ensembles"]:
        extra = f"  weights={r['weights']}" if "weights" in r else ""
        print(f"  {r['model']:24s} AUC={r['auc_roc']:.4f} F1={r['f1_binary']:.4f} FPR={r['false_positive_rate']*100:.2f}%{extra}")
    print("saved", out)


if __name__ == "__main__":
    main()
