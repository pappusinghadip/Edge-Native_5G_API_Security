"""Round-7 review: show why the norm filter separates the evaluated attack (Section 4.5).

Reads the per-client, per-round decision logs of the defended poisoning runs on the PC (kappa = 2.5, seeds
101-105): every update's L2 norm, whether it came from the attacker, and the round's cutoff (kappa times the cohort
median). Plots honest and malicious norms per round with the cutoff, and writes the norm ratios to
results/metrics/poison_norms.json.

Run: PYTHONPATH=. .venv/bin/python scripts/poison_norm_figure.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.pyplot as plt
import numpy as np

PC = Path("results/metrics_PC_run_2026-09-16_round5b")
SEEDS = [101, 102, 103, 104, 105]
honest, bad, cut = [], [], []          # (round, norm) and (round, cutoff) over all seeds
ratio_h, ratio_b = [], []              # norm / cohort median
for s in SEEDS:
    for r in json.loads((PC / f"fl_poison1_clip_iid_r5_poison_clip_s{s}.json").read_text())["history"]:
        med = r["median_norm"]
        cut.append((r["round"], r["norm_cutoff"]))
        for c in r["clients"]:
            (bad if c["adversarial"] else honest).append((r["round"], c["update_norm"]))
            (ratio_b if c["adversarial"] else ratio_h).append(c["update_norm"] / med)

H, B, C = np.array(honest), np.array(bad), np.array(cut)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
j = lambda n: (np.random.default_rng(0).random(n) - 0.5) * 0.35      # small horizontal jitter
ax.scatter(H[:, 0] + j(len(H)), H[:, 1], s=14, marker="o", facecolor="#14315C", edgecolor="none", alpha=0.55,
           label="honest update (400)")
ax.scatter(B[:, 0] + j(len(B)), B[:, 1], s=26, marker="x", color="#B23A3A", linewidth=1.3,
           label="malicious update, λ = 5 (100)")
rounds = np.unique(C[:, 0])
ax.plot(rounds, [np.median(C[C[:, 0] == r, 1]) for r in rounds], color="#333333", linestyle="--", linewidth=1.3,
        label="cutoff κ × median, κ = 2.5 (median over seeds)")
ax.set_yscale("log")
ax.set_xlabel("aggregation round"); ax.set_ylabel("update L2 norm (log scale)")
ax.set_xticks(range(2, 21, 2)); ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=8.5, loc="upper right", frameon=True)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"results/figures/poison_norms.{ext}", dpi=300)
out = {"source": str(PC), "kappa": 2.5, "lambda": 5, "seeds": SEEDS,
       "honest_over_median": {"max": float(max(ratio_h)), "median": float(np.median(ratio_h))},
       "malicious_over_median": {"min": float(min(ratio_b)), "median": float(np.median(ratio_b))},
       "updates": {"honest": len(ratio_h), "malicious": len(ratio_b)}}
Path("results/metrics/poison_norms.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
