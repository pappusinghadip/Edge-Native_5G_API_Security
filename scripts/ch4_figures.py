"""Chapter 4 figures. Each figure draws on one training platform only.

  detector_ranking      Mac   (the detector comparison of Table 4.1)
  convergence           PC    (IID and non-IID bands over five seeds)
  confusion_median_runs PC    (median-ROC-AUC run of each headline arm)
  scalability           PC    (three seeds per client count)

Run: python scripts/ch4_figures.py
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

M = Path("results/metrics")
P = Path("results/metrics_PC_run_2026-09-07")
F = Path("results/figures")
NAVY, ACT, GREEN, ORANGE, GREY = "#14315C", "#B23A3A", "#2E7D5B", "#B7791F", "#666666"
FA = json.loads((M / "final_analysis.json").read_text())


def convergence() -> None:
    """Global-model ROC-AUC per round on the PC: IID and non-IID bands over five seeds."""
    rep = json.loads((P / "repeated_federated_iid_r20.json").read_text())
    iid = np.array([[h["auc_roc"] for h in r["history"]] for r in rep["runs"]])
    rounds = np.arange(1, iid.shape[1] + 1)
    cen = FA["arms"]["centralized"]["summary"]["roc_auc"]

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    ax.axhspan(cen["ci95_low"], cen["ci95_high"], color=GREY, alpha=0.15,
               label=f"Centralized 95% CI [{cen['ci95_low']:.4f}, {cen['ci95_high']:.4f}]")
    ax.axhline(cen["mean"], color=GREY, linestyle="--", linewidth=1.3,
               label=f"Centralized mean ({cen['mean']:.4f})")

    def band(curves, colour, label, marker):
        m, s = curves.mean(axis=0), curves.std(axis=0, ddof=1)
        ax.fill_between(rounds, m - s, m + s, color=colour, alpha=0.16)
        ax.plot(rounds, m, color=colour, linewidth=1.9, marker=marker, markersize=3.2,
                label=f"{label}, mean ± 1 s.d. (5 seeds)")

    band(iid, NAVY, "Federated, IID", "o")
    band(np.array(list(FA["noniid"]["0.5"]["curves"].values())), GREEN, "Non-IID α = 0.5", "s")

    collapsed = {str(s) for s in FA["noniid"]["0.1"]["collapsed_seeds"]}
    first = True
    for s, c in FA["noniid"]["0.1"]["curves"].items():
        dead = s in collapsed
        ax.plot(rounds, c, color=ORANGE, linewidth=1.6 if dead else 0.9, alpha=0.95 if dead else 0.55,
                linestyle="--" if dead else "-",
                label=(f"Non-IID α = 0.1, seed {s} (collapsed)" if dead else
                       ("Non-IID α = 0.1, other seeds" if first else None)))
        if not dead:
            first = False

    pts = [(i + 1, v) for c in FA["poisoning"]["noclip_curves"].values()
           for i, v in enumerate(c) if v is not None and np.isfinite(v)]
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=ACT, marker="X", s=50, zorder=5,
                   label="Poisoned, no filter (all 5 seeds collapse)")

    ax.set_ylim(min([0.30] + [p[1] - 0.02 for p in pts]), 0.88)
    ax.set_xlabel("Aggregation round"); ax.set_ylabel("Global-model ROC-AUC")
    ax.set_title("Federated convergence on the PC, with run-to-run variability", color=NAVY,
                 fontsize=12, weight="bold")
    ax.set_xticks(rounds[::2]); ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc="center right")
    fig.tight_layout(); fig.savefig(F / "fl_convergence_ci.png", dpi=300); plt.close(fig)
    print("saved fl_convergence_ci.png")


def confusion_median_runs() -> None:
    """One identified run per arm (the median-ROC-AUC seed), never a 'mean model'."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, (arm, label) in zip(axes, (("centralized", "Centralized 1D-CNN"),
                                       ("federated", "Federated 1D-CNN"))):
        seed = FA["arms"][arm]["median_auc_seed"]
        m = FA["arms"][arm]["per_seed"][str(seed)]
        cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                        color="white" if (i == 0 and j == 0) else "#1a1a1a")
        ax.set_xticks([0, 1], ["Pred. benign", "Pred. malicious"], fontsize=9)
        ax.set_yticks([0, 1], ["True benign", "True malicious"], fontsize=9)
        ax.set_title(f"{label} — seed {seed} (median ROC-AUC)\n"
                     f"ROC-AUC={m['roc_auc']:.4f}  recall={m['recall']:.3f}  "
                     f"FPR={100 * m['false_positive_rate']:.2f}%", color=NAVY, fontsize=10.5)
    fig.suptitle("Confusion matrices at the tuned decision threshold (PC)", color=NAVY,
                 fontsize=12.5, weight="bold")
    fig.tight_layout(); fig.savefig(F / "confusion_matrices_median.png", dpi=300); plt.close(fig)
    print("saved confusion_matrices_median.png")


def scalability() -> None:
    """Per-round compute, communication and final ROC-AUC on the PC, three seeds per K."""
    sc = FA["scalability"]
    ks = sorted(int(k) for k in sc)
    rows = [sc[str(k)] for k in ks]
    with plt.rc_context({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12}):
        fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))
        axes[0].errorbar(ks, [r["mean_round_seconds"]["mean"] for r in rows],
                         yerr=[r["mean_round_seconds"]["std"] for r in rows], color=NAVY, marker="o", capsize=4)
        axes[0].set_ylim(0, max(r["mean_round_seconds"]["mean"] for r in rows) * 1.4)
        axes[0].set_title("Compute: flat in K", color=NAVY, weight="bold")
        axes[0].set_ylabel("Mean seconds per round")
        axes[1].plot(ks, [r["total_comm_mb"] for r in rows], color=NAVY, marker="o")
        axes[1].set_yscale("log")
        axes[1].set_title("Communication: linear in K", color=NAVY, weight="bold")
        axes[1].set_ylabel("Total MB over 5 rounds")
        for k, r in zip(ks, rows):
            v = list(r["per_seed"].values())
            axes[2].scatter([k] * len(v), v, color=ACT, s=36, zorder=4, alpha=0.8)
        axes[2].plot(ks, [r["final_auc"]["mean"] for r in rows], color=NAVY, marker="o", label="mean of 3 seeds")
        axes[2].scatter([], [], color=ACT, s=36, label="individual seeds")
        axes[2].set_title("Accuracy at 5 rounds", color=NAVY, weight="bold")
        axes[2].set_ylabel("Final global ROC-AUC"); axes[2].legend(loc="lower left")
        for ax in axes:
            ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
            ax.set_xlabel("Clients (K)"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(F / "scalability_extended.png", dpi=300); plt.close(fig)
    print("saved scalability_extended.png")


def detector_ranking() -> None:
    """Figure 4.1 — the Mac detector comparison, using exactly the values of Table 4.1."""
    ext = json.loads((M / "extended_metrics.json").read_text())
    five = M / "baselines_seeds_original.json"          # round 6: the four single models over five seeds
    seeds5 = json.loads(five.read_text())["models"] if five.exists() else {}
    val = lambda k: seeds5[k]["summary"]["auc_roc"]["mean"] if k in seeds5 else ext[k]["auc_roc"]
    rows = [(k, val(k), False) for k in
            ("Weighted", "XGBoost", "Stacked", "Gradient Boosting", "Random Forest", "MLP")]
    rows.append(("1D-CNN\n(centralized)", FA["arms_mac"]["centralized"]["summary"]["roc_auc"]["mean"], False))
    rows.append(("1D-CNN\n(federated)", FA["arms_mac"]["federated"]["summary"]["roc_auc"]["mean"], True))
    rows.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows], color=[ACT if r[2] else NAVY for r in rows])
    for bar, (_, v, _) in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0.5, 0.93); ax.set_ylabel("ROC-AUC")
    ax.set_title("Detectors ranked by ROC-AUC (Mac; deployed federated model in red)",
                 color=NAVY, fontsize=12, weight="bold")
    ax.tick_params(axis="x", labelrotation=20, labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(F / "detector_ranking.png", dpi=300); plt.close(fig)
    print("saved detector_ranking.png")


if __name__ == "__main__":
    F.mkdir(parents=True, exist_ok=True)
    convergence()
    confusion_median_runs()
    scalability()
    detector_ranking()
