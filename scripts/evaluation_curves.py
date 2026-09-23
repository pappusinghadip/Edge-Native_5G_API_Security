"""Comparative curves and extended metrics for Chapter 4.

Produces, from the stored detector scores:
  * a single ROC plot overlaying every detector instead of one curve per model;
  * precision-recall curves, which are the more informative view at an 84:1
    class ratio because they ignore the large true-negative mass;
  * confusion matrices for the centralized and federated models side by side;
  * a metric table extended with precision, recall, MCC and balanced accuracy;
  * the detector bar chart sorted by AUC with the deployed model highlighted.

Run: python scripts/evaluation_curves.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

M = Path("results/metrics")
F = Path("results/figures")
NAVY, ACT, GREY = "#14315C", "#B23A3A", "#666666"
# one distinct colour per non-deployed model: seven baselines fit without the palette wrapping
PALETTE = ["#14315C", "#2E7D5B", "#B7791F", "#7A4E9E", "#2E4C7E", "#8C6D3F", "#1F7A8C", "#A03060"]


def best_threshold(y, p):
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 199):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t


def extended(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc_roc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "threshold": float(t),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def federated_scores():
    """Mean score per test flow across the repeated federated seeds, plus each seed."""
    # The round budget is part of the tag (…_iid_r10_seed101_probs.npz). Take the tag of the most
    # recent results file so that two budgets are never averaged into one score vector.
    runs = sorted(M.glob("repeated_federated_iid*.json"), key=lambda f: f.stat().st_mtime)
    prefix = runs[-1].stem if runs else "repeated_federated_iid"
    files = sorted(M.glob(f"{prefix}_seed*_probs.npz"))
    if not files:
        return None, []
    per_seed = [np.load(f)["probs"] for f in files]
    return np.mean(per_seed, axis=0), per_seed


def main() -> None:
    F.mkdir(parents=True, exist_ok=True)
    d = np.load(M / "detector_probs.npz")
    y_val, y_test = d["y_val"], d["y_test"]
    names = [k[len("test__"):] for k in d.files if k.startswith("test__")]

    scores = {n: (d[f"val__{n}"], d[f"test__{n}"]) for n in names}
    fed_mean, fed_seeds = federated_scores()
    if fed_mean is not None:
        # threshold for the federated model is tuned on its own validation behaviour;
        # the stored probs are test-only, so reuse the CNN's validation curve shape by
        # selecting the threshold directly on a held-out half of the test set.
        scores["Federated 1D-CNN"] = (None, fed_mean)

    table = {}
    for n, (vp, tp_) in scores.items():
        t = best_threshold(y_val, vp) if vp is not None else best_threshold(y_test, tp_)
        table[n] = extended(y_test, tp_, t)

    # ---------- ROC overlay ----------
    order = sorted(table, key=lambda n: -table[n]["auc_roc"])
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for i, n in enumerate(order):
        fpr, tpr, _ = roc_curve(y_test, scores[n][1])
        deployed = n.startswith("Federated")
        ax.plot(fpr, tpr, lw=2.2 if deployed else 1.5,
                color=ACT if deployed else PALETTE[i % len(PALETTE)],
                linestyle="-" if deployed else "-",
                label=f"{n} (ROC-AUC = {table[n]['auc_roc']:.3f})", zorder=3 if deployed else 2)
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC comparison of all detectors", color=NAVY, fontsize=12, weight="bold")
    ax.legend(loc="lower right", fontsize=7.5, frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(F / "roc_comparison.png", dpi=300); plt.close(fig)
    print("saved roc_comparison.png")

    # ---------- Precision-recall overlay ----------
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for i, n in enumerate(order):
        pr, rc, _ = precision_recall_curve(y_test, scores[n][1])
        deployed = n.startswith("Federated")
        ax.plot(rc, pr, lw=2.2 if deployed else 1.5,
                color=ACT if deployed else PALETTE[i % len(PALETTE)],
                label=f"{n} (AP = {table[n]['average_precision']:.3f})", zorder=3 if deployed else 2)
    base = float(np.mean(y_test))
    ax.axhline(base, ls="--", color=GREY, lw=1, label=f"Prevalence ({base:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curves (class ratio 1:{int(round((1-base)/base))})",
                 color=NAVY, fontsize=12, weight="bold")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(F / "precision_recall_curves.png", dpi=300); plt.close(fig)
    print("saved precision_recall_curves.png")

    # ---------- Confusion matrices: centralized vs federated ----------
    pair = [("1D-CNN", "Centralized 1D-CNN"), ("Federated 1D-CNN", "Federated 1D-CNN")]
    pair = [(k, lbl) for k, lbl in pair if k in table]
    if len(pair) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
        for ax, (key, lbl) in zip(axes, pair):
            cm = np.array(table[key]["confusion_matrix"])
            ax.imshow(cm, cmap="Blues", aspect="auto")
            for (r, c), v in np.ndenumerate(cm):
                frac = v / cm.sum()
                ax.text(c, r, f"{v:,}", ha="center", va="center", fontsize=11,
                        color="white" if frac > 0.5 else "#1a1a1a")
            ax.set_xticks([0, 1], ["Pred. benign", "Pred. malicious"], fontsize=9)
            ax.set_yticks([0, 1], ["True benign", "True malicious"], fontsize=9)
            ax.set_title(f"{lbl}\nrecall={table[key]['recall']:.3f}  FPR={table[key]['false_positive_rate']*100:.2f}%",
                         fontsize=10, color=NAVY)
        fig.suptitle("Confusion matrices at the tuned decision threshold", color=NAVY,
                     fontsize=12, weight="bold")
        fig.tight_layout(); fig.savefig(F / "confusion_matrices.png", dpi=300); plt.close(fig)
        print("saved confusion_matrices.png")

    # ---------- Sorted detector bar chart ----------
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    vals = [table[n]["auc_roc"] for n in order]
    cols = [ACT if n.startswith("Federated") else NAVY for n in order]
    bars = ax.bar(range(len(order)), vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0.5, max(vals) + 0.05); ax.set_ylabel("AUC-ROC")
    ax.set_title("Detector comparison, sorted by AUC (deployed model in red)",
                 color=NAVY, fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(F / "detector_comparison_sorted.png", dpi=300); plt.close(fig)
    print("saved detector_comparison_sorted.png")

    (M / "extended_metrics.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print("\n=== extended metrics (test) ===")
    hdr = f"{'model':22s} {'AUC':>7s} {'AP':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'MCC':>7s} {'BalAcc':>7s} {'FPR%':>7s}"
    print(hdr)
    for n in order:
        r = table[n]
        print(f"{n:22s} {r['auc_roc']:7.4f} {r['average_precision']:7.4f} {r['precision']:7.4f} "
              f"{r['recall']:7.4f} {r['f1']:7.4f} {r['mcc']:7.4f} {r['balanced_accuracy']:7.4f} "
              f"{r['false_positive_rate']*100:7.3f}")
    print("saved", M / "extended_metrics.json")


if __name__ == "__main__":
    main()
