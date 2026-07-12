"""Aggregate Phase 3 FL results into thesis figures and a summary table.

Reads results/metrics/fl_*.json (+ centralized_results.json) and produces:
  results/figures/fl_convergence.png     — global-model AUC vs FL round
  results/figures/fl_vs_centralized.png  — final AUC / F1 comparison bars
  results/metrics/phase3_summary.md      — comparison table

Run after scripts/run_fl_phase3.sh finishes:
  .venv/bin/python -m scripts.summarize_fl   (or: python scripts/summarize_fl.py)
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

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"


def load(name: str) -> dict | None:
    path = METRICS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def plot_convergence(runs: dict[str, dict]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, data in runs.items():
        hist = data.get("history", [])
        if not hist:
            continue
        ax.plot([r["round"] for r in hist], [r["auc_roc"] for r in hist], marker="o", ms=3, label=label)
    cen = load("centralized_results.json")
    if cen:
        ax.axhline(cen["auc_roc"], ls="--", color="gray", label=f"Centralized baseline (AUC={cen['auc_roc']:.3f})")
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Global-model AUC-ROC (test set)")
    ax.set_title("Federated Learning Convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "fl_convergence.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_comparison(bars: list[tuple[str, float, float]]) -> Path:
    labels = [b[0] for b in bars]
    aucs = [b[1] for b in bars]
    f1s = [b[2] for b in bars]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    w = 0.38
    ax.bar([i - w / 2 for i in x], aucs, w, label="AUC-ROC")
    ax.bar([i + w / 2 for i in x], f1s, w, label="F1 (malicious)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Centralized vs Federated vs Isolated (test set)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for i, (a, f) in enumerate(zip(aucs, f1s)):
        ax.text(i - w / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i + w / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    out = FIGURES / "fl_vs_centralized.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    conv_runs = {}
    for tag, label in [
        ("fl_fedavg_iid.json", "FedAvg (IID)"),
        ("fl_fedavg_dirichlet.json", "FedAvg (non-IID α=0.5)"),
        ("fl_poison1_noclip_iid.json", "Poisoned, no clip"),
        ("fl_poison1_clip_iid.json", "Poisoned, SafeFedAvg clip"),
    ]:
        d = load(tag)
        if d:
            conv_runs[label] = d

    if conv_runs:
        print("convergence ->", plot_convergence(conv_runs))

    # Comparison bars
    bars: list[tuple[str, float, float]] = []
    cen = load("centralized_results.json")
    if cen:
        bars.append(("Centralized", cen["auc_roc"], cen["f1_binary"]))
    for tag, label in [
        ("fl_fedavg_iid.json", "FL IID"),
        ("fl_fedavg_dirichlet.json", "FL non-IID"),
        ("fl_isolated_iid.json", "Isolated (mean)"),
    ]:
        d = load(tag)
        if d and d.get("final"):
            bars.append((label, d["final"]["auc_roc"], d["final"]["f1_binary"]))
    if bars:
        print("comparison ->", plot_comparison(bars))

    # Summary table
    lines = ["# Phase 3 — Federated Learning Results", "", "## Global-model performance (held-out test set)", "",
             "| Configuration | AUC-ROC | F1 (malicious) | Recall (mal.) | FPR |",
             "|---|---|---|---|---|"]

    def row(name: str, m: dict) -> str:
        return (f"| {name} | {m['auc_roc']:.4f} | {m['f1_binary']:.4f} | "
                f"{m.get('recall_binary', float('nan')):.4f} | {m['false_positive_rate']*100:.2f}% |")

    if cen:
        lines.append(f"| Centralized baseline | {cen['auc_roc']:.4f} | {cen['f1_binary']:.4f} | "
                     f"{cen['recall_binary']:.4f} | {cen['false_positive_rate']*100:.2f}% |")
    for tag, name in [
        ("fl_fedavg_iid.json", "FL FedAvg (IID)"),
        ("fl_fedavg_dirichlet.json", "FL FedAvg (non-IID α=0.5)"),
        ("fl_isolated_iid.json", "Isolated local training (mean)"),
        ("fl_poison1_noclip_iid.json", "Poisoned (1/5), no defense"),
        ("fl_poison1_clip_iid.json", "Poisoned (1/5), SafeFedAvg clip"),
    ]:
        d = load(tag)
        if d and d.get("final"):
            lines.append(row(name, d["final"]))

    # Poisoning conclusion
    pn = load("fl_poison1_noclip_iid.json")
    pc = load("fl_poison1_clip_iid.json")
    if pn and pc and pn.get("final") and pc.get("final"):
        lines += ["", "## Poisoning robustness", "",
                  f"A single sign-flipping client (1 of 5) drove the undefended global model to "
                  f"AUC {pn['final']['auc_roc']:.3f}. With the SafeFedAvg gradient-norm safeguard enabled, "
                  f"the global model held at AUC {pc['final']['auc_roc']:.3f} "
                  f"(rejecting {sum(r['rejected_updates'] for r in pc.get('history', []))} malicious updates across all rounds)."]

    out = METRICS / "phase3_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("summary ->", out)


if __name__ == "__main__":
    main()
