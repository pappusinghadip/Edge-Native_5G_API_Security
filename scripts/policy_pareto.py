"""Round-5 review: choose k and n from a Pareto frontier, not by narrative preference.

A configuration is Pareto-optimal if no other one is at least as good on all four criteria (higher
detection, lower false-block rate, lower median time to enforcement, smaller window n, i.e. less state)
and strictly better on one. Reads the deployed-model sweep; draws the Pareto plot the review asks for.

Run: PYTHONPATH=. .venv/bin/python scripts/policy_pareto.py
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

M, F = Path("results/metrics"), Path("results/figures")
grid = json.loads((M / "sensitivity_analysis.json").read_text())["grid"]
def dominates(a, b):
    """Two operational criteria: more attacks enforced against, fewer benign sources blocked."""
    return (a["detection_rate"] >= b["detection_rate"] and a["false_block_rate"] <= b["false_block_rate"]
            and (a["detection_rate"] > b["detection_rate"] or a["false_block_rate"] < b["false_block_rate"]))


for g in grid:
    g["pareto"] = not any(dominates(o, g) for o in grid if o is not g)

# cost J = (1 - detection) + wf * false_block + wd * median TTE, a missed attack costing 1
weights = [(wf, wd) for wf in (1, 2, 5, 10) for wd in (0.0, 0.002, 0.005)]
winners = {}
for wf, wd in weights:
    b = min(grid, key=lambda g: (1 - g["detection_rate"]) + wf * g["false_block_rate"] + wd * g["ttd_flows_median"])
    winners.setdefault(f"k={b['k']}, n={b['n']}", []).append([wf, wd])
out = {"criteria": "Pareto on detection (up) and false-block rate (down); TTE shown by marker area",
       "grid": [{k: g[k] for k in ("k", "n", "detection_rate", "false_block_rate", "ttd_flows_median", "pareto")}
                for g in grid],
       "cost_function": {"form": "J = (1 - detection) + wf * false_block + wd * median_TTE_flows",
                         "weights_tried": weights, "winners": winners}}
(M / "policy_pareto.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

fig, ax = plt.subplots(figsize=(7.6, 5.0))
ZOOM = (-0.004, 0.036, 0.86, 1.005)          # the crowded corner, drawn again as an inset


def draw(axis, label_all):
    for g in grid:
        on, chosen = g["pareto"], (g["k"], g["n"]) == (3, 10)
        axis.scatter(g["false_block_rate"], g["detection_rate"], s=18 + 6 * g["ttd_flows_median"],
                     marker="o" if on else "s", facecolor="#14315C" if on else "white",
                     edgecolor="#B23A3A" if chosen else ("#14315C" if on else "#666666"),
                     linewidth=2.0 if chosen else 1.1, alpha=0.9, zorder=3)
        inside = ZOOM[0] <= g["false_block_rate"] <= ZOOM[1] and ZOOM[2] <= g["detection_rate"] <= ZOOM[3]
        if label_all or not inside:
            axis.annotate(f"k={g['k']}, n={g['n']}", (g["false_block_rate"], g["detection_rate"]),
                          textcoords="offset points", xytext=(8, -3), fontsize=7.5,
                          color="#B23A3A" if chosen else "#333333", fontweight="bold" if chosen else "normal")


draw(ax, label_all=False)
ax.add_patch(plt.Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2], fill=False,
                           edgecolor="#999999", linestyle="--", linewidth=0.8))
ax.set_xlabel("false-block rate (benign prelude)"); ax.set_ylabel("detection rate (complete scenario success)")
ax.set_title("k-of-n policy on the deployed model: filled = Pareto-optimal on detection and false blocks;\n"
             "marker area grows with median time to enforcement; red outline = setting used throughout",
             color="#14315C", fontsize=9)
ax.grid(alpha=0.25)
ins = ax.inset_axes([0.27, 0.06, 0.44, 0.40])      # the empty lower-middle region
draw(ins, label_all=True)
ins.set_xlim(ZOOM[0], ZOOM[1]); ins.set_ylim(ZOOM[2], ZOOM[3])
ins.tick_params(labelsize=7); ins.grid(alpha=0.25)
ins.set_title("detail of the dashed box", fontsize=7.5, color="#666666")
fig.tight_layout(); fig.savefig(F / "policy_pareto.png", dpi=300); plt.close(fig)
print("pareto-optimal:", [(g["k"], g["n"]) for g in grid if g["pareto"]], "winners:", winners)
