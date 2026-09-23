"""Ranked comparison of every detection model (Task 3.1 + review baselines)."""
import json, os, tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = "results/metrics/"
rows = []
rl = json.load(open(M + "baseline_ratelimiter.json"))
rows.append(("Static rate-limiter", rl["auc_roc"], rl["f1_binary"], False))
be = json.load(open(M + "best_ensemble.json"))
for r in be["base_models"]:
    rows.append((r["model"], r["auc_roc"], r["f1_binary"], False))
best = max(be["ensembles"], key=lambda e: e["auc_roc"])
rows.append(("Best ensemble", best["auc_roc"], best["f1_binary"], True))

rows.sort(key=lambda r: r[1])
labels = [r[0] for r in rows]; auc = [r[1] for r in rows]; f1 = [r[2] for r in rows]
colors = ["#B23A3A" if r[3] else "#14315C" for r in rows]

fig, ax = plt.subplots(figsize=(7.4, 4.6))
bars = ax.barh(range(len(rows)), auc, color=colors, height=0.62)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlim(0.45, 0.93); ax.set_xlabel("AUC-ROC (test set)")
ax.set_title("Detection model comparison (single-attack subset)")
for i, (a, f) in enumerate(zip(auc, f1)):
    ax.text(a + 0.003, i, f"{a:.3f}   (F1 {f:.3f})", va="center", fontsize=8.3, color="#1a1a1a")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig("results/figures/detector_comparison.png", dpi=300)
print("saved results/figures/detector_comparison.png")
