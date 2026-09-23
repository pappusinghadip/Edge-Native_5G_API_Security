"""Figures 4.2 and 4.3 redrawn for print (round-5 review, Section 11): distinct line styles and
markers so no curve depends on colour alone, larger labels, the legend outside the plot, and the same
model order in both figures. Draws from the stored Mac scores only; writes no metrics file.

Run: PYTHONPATH=. .venv/bin/python scripts/detector_curves_styled.py
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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

M, F = Path("results/metrics"), Path("results/figures")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9.5})
d = np.load(M / "detector_probs.npz")
y = d["y_test"]
scores = {k[len("test__"):]: d[k] for k in d.files if k.startswith("test__")}
# both 1D-CNN curves come from the corrected-loss Mac 60N arms: the median-ROC-AUC seed of each, the same
# runs Table 4.1 and the k-of-n policy use (named, not "latest"); the older detector_probs CNN is dropped
_fa = json.loads((M / "final_analysis.json").read_text())["arms_mac"]
for arm, key in (("centralized", "1D-CNN"), ("federated", "Federated 1D-CNN (deployed)")):
    a = _fa[arm]
    scores[key] = np.load(M / a["probs_files"][str(a["median_auc_seed"])])["probs"]
NAMES = {"Weighted": "Weighted ensemble", "Stacked": "Stacked ensemble", "1D-CNN": "1D-CNN (centralized)",
         "MLP": "Multilayer perceptron"}          # the names Table 4.1 uses
scores = {NAMES.get(k, k): v for k, v in scores.items()}
order = sorted(scores, key=lambda n: -roc_auc_score(y, scores[n]))
STYLE = [("-", "o"), ("--", "s"), ("-.", "^"), (":", "D"), ("-", "v"), ("--", "P"), ("-.", "X"), (":", "*")]
COL = ["#14315C", "#2E7D5B", "#B7791F", "#6B4C9A", "#3A7CA5", "#8C564B", "#555555", "#9E9E9E"]


def style(i, name):
    if name.startswith("Federated"):
        return dict(color="#B23A3A", linestyle="-", marker="o", lw=2.4, zorder=4)
    ls, mk = STYLE[i % len(STYLE)]
    return dict(color=COL[i % len(COL)], linestyle=ls, marker=mk, lw=1.6, zorder=3)


for kind in ("roc", "pr"):
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    for i, n in enumerate(order):
        s = scores[n]
        if kind == "roc":
            x, yy, _ = roc_curve(y, s); lab = f"{n} (ROC-AUC {roc_auc_score(y, s):.3f})"
        else:
            p, r, _ = precision_recall_curve(y, s); x, yy = r, p; lab = f"{n} (AP {average_precision_score(y, s):.3f})"
        ax.plot(x, yy, label=lab, markevery=0.08, markersize=5.5, markerfacecolor="white", **style(i, n))
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color="#999999", lw=1, linestyle=(0, (4, 4)), label="Random (ROC-AUC 0.500)")
        ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
    else:
        base = float(y.mean())
        ax.axhline(base, color="#999999", lw=1, linestyle=(0, (4, 4)), label=f"No skill (prevalence {base:.4f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout()
    name = "roc_comparison.png" if kind == "roc" else "precision_recall_curves.png"
    fig.savefig(F / name, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("saved", name)
