"""The two Chapter 4 figures whose data changed in the round-5 reruns (fixed loss, Asus).

  fl_convergence_ci_r5.png        IID, non-IID and undefended-poisoned convergence, five seeds
  confusion_matrices_median_r5.png  median-ROC-AUC run of the 60N centralized and federated arms

Run: PYTHONPATH=. .venv/bin/python scripts/round5_figures.py
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

from src.models.io import load_processed_split
from src.utils.config import load_yaml, resolve_path

M = Path("results/metrics")
T1 = Path("results/metrics_PC_run_2026-09-14_round5")
T2 = Path("results/metrics_PC_run_2026-09-16_round5b")
F = Path("results/figures")
NAVY, ACT, GREEN, ORANGE, GREY = "#14315C", "#B23A3A", "#2E7D5B", "#B7791F", "#666666"
SEEDS = [101, 102, 103, 104, 105]
R5 = json.loads((M / "round5_analysis.json").read_text())


def hist(path: Path) -> list[float]:
    """Per-round ROC-AUC, from either output shape (simulate writes history at the top level,
    repeated_runs nests it inside the single run record)."""
    d = json.loads(path.read_text())
    h = d["history"] if "history" in d else d["runs"][0]["history"]
    return [r["auc_roc"] for r in h]


def convergence() -> None:
    """Two panels (round-5 review, Section 11): (a) clean training under IID and three non-IID levels
    against the centralized interval, with sample exposure on a second axis; (b) the poisoning runs."""
    load = lambda base, pat: np.array([hist(base / pat.format(s=s)) for s in SEEDS])
    clean = [("federated IID", load(T1, "repeated_federated_iid_r5_le3_s{s}.json"), NAVY, "-", "o"),
             ("non-IID α = 1.0", load(T2, "fl_fedavg_dirichlet_r5_noniid_a1p0_s{s}.json"), "#3A7CA5", "--", "D"),
             ("non-IID α = 0.5", load(T2, "fl_fedavg_dirichlet_r5_noniid_a0p5_s{s}.json"), GREEN, "-.", "s"),
             ("non-IID α = 0.1", load(T2, "fl_fedavg_dirichlet_r5_noniid_a0p1_s{s}.json"), ORANGE, ":", "^")]
    attack = [("poisoned, no filter", np.nan_to_num(load(T2, "fl_poison1_noclip_iid_r5_poison_noclip_s{s}.json"), nan=0.5),
               ACT, ":", "x"),
              ("poisoned, norm filter", load(T2, "fl_poison1_clip_iid_r5_poison_clip_s{s}.json"), NAVY, "-", "o"),
              ("no attacker, norm filter", load(T2, "fl_fedavg_iid_r5_clean_clip_s{s}.json"), GREEN, "--", "s")]
    rounds = np.arange(1, clean[0][1].shape[1] + 1)
    cen = R5["arms"]["centralized_60N"]["summary"]["auc_roc"]
    plt.rcParams.update({"font.size": 10.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax in (a, b):
        ax.axhspan(cen["ci95_low"], cen["ci95_high"], color=GREY, alpha=0.18, label="centralized 60 epochs, 95% CI")
        ax.set_xlabel("aggregation round"); ax.grid(alpha=0.25); ax.set_xticks(rounds[1::2])
    for ax, series in ((a, clean), (b, attack)):
        for label, data, colour, ls, mk in series:
            m, sd = data.mean(axis=0), data.std(axis=0, ddof=1)
            ax.fill_between(rounds, m - sd, m + sd, color=colour, alpha=0.14)
            ax.plot(rounds, m, color=colour, linestyle=ls, marker=mk, markersize=4, markevery=2,
                    markerfacecolor="white", linewidth=1.8, label=label)
    a.set_ylim(0.70, 0.85); a.set_ylabel("global-model ROC-AUC")
    a.set_title("(a) clean training, five seeds per curve", color=NAVY, fontsize=11)
    b.set_ylim(0.0, 0.9)          # the undefended runs fall far below chance before settling at 0.5
    b.set_title("(b) one malicious client in five, five seeds per curve", color=NAVY, fontsize=11)
    top = a.secondary_xaxis("top", functions=(lambda r: 3 * r, lambda e: e / 3))
    top.set_xlabel("sample exposure (multiples of N)")
    a.legend(loc="lower right", fontsize=9); b.legend(loc="center right", fontsize=9)
    fig.tight_layout(); fig.savefig(F / "fl_convergence_ci_r5.png", dpi=300); plt.close(fig)
    print("wrote fl_convergence_ci_r5.png")


def confusion() -> None:
    paths = load_yaml("configs/paths.yaml")
    _, y, _ = load_processed_split(resolve_path(paths["data"]["processed"]) / "test.npz")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    picked = []
    for ax, (arm, pattern, title) in zip(axes, (
            ("centralized_60N", "repeated_centralized_r5_e60_s{s}", "Centralized, 60 epochs"),
            ("federated_60N", "repeated_federated_iid_r5_le3_s{s}", "Federated, 20 rounds × 3"))):
        per = R5["arms"][arm]["per_seed"]
        med = sorted(SEEDS, key=lambda s: per[str(s)]["auc_roc"])[len(SEEDS) // 2]
        run = json.loads((T1 / f"{pattern.format(s=med)}.json").read_text())["runs"][0]
        probs = np.load(T1 / f"{pattern.format(s=med)}_seed{med}_probs.npz")["probs"]
        pred = (probs >= run["threshold"]).astype(int)
        cm = np.array([[int(((pred == j) & (y == i)).sum()) for j in (0, 1)] for i in (0, 1)])
        picked.append((med, cm))
        ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm())
        for i in (0, 1):
            for j in (0, 1):
                ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center", fontsize=11,
                        color="white" if (i == 0 and j == 0) else "#1a1a1a")
        ax.set_xticks([0, 1], ["predicted benign", "predicted attack"], fontsize=8)
        ax.set_yticks([0, 1], ["benign", "attack"], fontsize=8)
        ax.set_title(f"{title}, seed {med}", color=NAVY, fontsize=10)
    fig.tight_layout(); fig.savefig(F / "confusion_matrices_median_r5.png", dpi=300); plt.close(fig)
    print("wrote confusion_matrices_median_r5.png", picked)


if __name__ == "__main__":
    convergence()
    confusion()
