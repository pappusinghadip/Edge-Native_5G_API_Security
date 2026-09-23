"""Confidence intervals and a significance test for the reported detection scores.

Chapter 4 compares AUC values such as 0.835 (centralized) and 0.827 (federated)
without saying whether the gap exceeds measurement noise. Two sources of
uncertainty are quantified here:

  * Sampling uncertainty — how much a score would move on a different test sample
    of the same size. Estimated by stratified bootstrap resampling of the test set
    (percentile intervals), which needs no retraining.
  * The paired difference between two detectors evaluated on the same flows. Because
    the two score vectors are correlated, the difference is bootstrapped as a paired
    quantity, and DeLong's test — the standard parametric test for two correlated
    ROC curves — is reported alongside it.

Training-run variance (different seeds) is a separate question, handled by
repeated_runs.py where the compute budget allows.

Run: python scripts/bootstrap_significance.py [--resamples 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

M = Path("results/metrics")


# --------------------------------------------------------------------------- #
# DeLong's test for two correlated ROC AUCs (DeLong et al., 1988; Sun & Xu, 2014)
# --------------------------------------------------------------------------- #
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
    """Return (auc1, auc2, two-sided p-value) for two score vectors on the same labels."""
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    preds = np.vstack([p1, p2])
    k = 2

    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(preds[r][pos])
        ty[r] = _midrank(preds[r][neg])
        tz[r] = _midrank(np.concatenate([preds[r][pos], preds[r][neg]]))

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    contrast = np.array([[1.0, -1.0]])
    var = float(contrast @ s @ contrast.T)
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan")
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


# --------------------------------------------------------------------------- #
def metrics_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    pred = p >= thr
    tp = float(np.sum(pred & (y == 1))); fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1))); tn = float(np.sum(~pred & (y == 0)))
    return {
        "auc_roc": float(roc_auc_score(y, p)),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
    }


def stratified_indices(rng, pos_idx, neg_idx):
    return np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                           rng.choice(neg_idx, len(neg_idx), replace=True)])


def bootstrap_ci(y, p, thr, n_boot, rng):
    pos_idx = np.flatnonzero(y == 1); neg_idx = np.flatnonzero(y == 0)
    keys = ("auc_roc", "f1", "false_positive_rate", "recall")
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = stratified_indices(rng, pos_idx, neg_idx)
        m = metrics_at(y[idx], p[idx], thr)
        for k in keys:
            acc[k].append(m[k])
    point = metrics_at(y, p, thr)
    return {k: {"point": point[k],
                "ci95_low": float(np.percentile(acc[k], 2.5)),
                "ci95_high": float(np.percentile(acc[k], 97.5)),
                "std": float(np.std(acc[k], ddof=1))} for k in keys}


def best_threshold(y, p):
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 199):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t


def federated_probs(X_test) -> np.ndarray | None:
    """Score the test set with the saved federated global model."""
    h5 = Path("results/models/fl_fedavg_iid_global.h5")
    if not h5.exists():
        return None
    import tensorflow as tf
    from src.models.cnn import focal_loss
    model = tf.keras.models.load_model(h5, custom_objects={"focal_loss": focal_loss()})
    return model.predict(X_test, batch_size=2048, verbose=0)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    d = np.load(M / "detector_probs.npz")
    y_val, y_test = d["y_val"], d["y_test"]
    scores = {k[len("test__"):]: (d[f"val__{k[len('test__'):]}"], d[k])
              for k in d.files if k.startswith("test__")}

    from src.models.io import load_processed_split
    from src.utils.config import load_yaml, resolve_path
    paths_cfg = load_yaml("configs/paths.yaml")
    X_te, y_te, _ = load_processed_split(resolve_path(paths_cfg["data"]["processed"]) / "test.npz")
    assert np.array_equal(y_te, y_test), "test label order differs between artifacts"

    fed = federated_probs(X_te)
    if fed is not None:
        scores["Federated 1D-CNN"] = (None, fed)
        np.savez_compressed(M / "federated_test_probs.npz", probs=fed.astype(np.float32))

    out = {"resamples": args.resamples, "per_model": {}, "paired": {}}
    for name, (vp, tp_) in scores.items():
        thr = best_threshold(y_val, vp) if vp is not None else best_threshold(y_test, tp_)
        out["per_model"][name] = {"threshold": thr, **bootstrap_ci(y_test, tp_, thr, args.resamples, rng)}
        r = out["per_model"][name]["auc_roc"]
        print(f"{name:22s} AUC={r['point']:.4f}  95% CI [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]", flush=True)

    # Paired comparison: centralized 1D-CNN vs the federated global model.
    if fed is not None and "1D-CNN" in scores:
        cen = scores["1D-CNN"][1]
        a1, a2, p_delong = delong_test(y_test, cen, fed)
        pos_idx = np.flatnonzero(y_test == 1); neg_idx = np.flatnonzero(y_test == 0)
        diffs = []
        for _ in range(args.resamples):
            idx = stratified_indices(rng, pos_idx, neg_idx)
            diffs.append(roc_auc_score(y_test[idx], cen[idx]) - roc_auc_score(y_test[idx], fed[idx]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out["paired"]["centralized_vs_federated_auc"] = {
            "centralized_auc": a1, "federated_auc": a2, "difference": a1 - a2,
            "diff_ci95_low": float(lo), "diff_ci95_high": float(hi),
            "delong_p_value": p_delong,
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        }
        print(f"\ncentralized {a1:.4f} vs federated {a2:.4f}: diff={a1-a2:+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  DeLong p={p_delong:.4f}")

    (M / "bootstrap_significance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nsaved", M / "bootstrap_significance.json")


if __name__ == "__main__":
    main()
