"""Hybrid-architecture and API-attack-lifecycle diagrams (review additions)."""
import os, tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY = "#14315C"; STEP = "#2E4C7E"; ACT = "#B23A3A"; LIGHT = "#EAF0F8"; RED_L = "#FBEAEA"

def box(ax, cx, cy, w, h, text, edge=STEP, fill=LIGHT, fs=10):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.06",
                                linewidth=1.6, edgecolor=edge, facecolor=fill))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color="#1a1a1a")

def arrow(ax, x1, y1, x2, y2, color=NAVY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.5, color=color))

# ---------------- Hybrid architecture ----------------
fig, ax = plt.subplots(figsize=(6.4, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
box(ax, 5, 9.0, 4.6, 1.05, "Flow features  (10-D vector)", edge=NAVY, fill="#FFFFFF")
arrow(ax, 4.0, 8.47, 2.7, 7.35); arrow(ax, 6.0, 8.47, 7.3, 7.35)
box(ax, 2.6, 6.7, 3.0, 1.05, "1D-CNN\n(federated, low-latency)")
box(ax, 7.4, 6.7, 3.0, 1.05, "Gradient Boosting\n(strongest on tabular)")
arrow(ax, 2.6, 6.17, 4.0, 4.75); arrow(ax, 7.4, 6.17, 6.0, 4.75)
box(ax, 5, 4.1, 6.4, 1.15, "Weighted fusion\n$s = w\\,p_{CNN} + (1-w)\\,p_{GBM}$", edge=STEP)
arrow(ax, 5, 3.52, 5, 2.5)
box(ax, 5, 1.85, 4.2, 1.05, "Risk score  $P(\\mathrm{malicious})$", edge=ACT, fill=RED_L)
ax.set_title("Hybrid CNN–Gradient Boosting detector", fontsize=12, color=NAVY, weight="bold", pad=4)
fig.tight_layout(); fig.savefig("results/figures/hybrid_architecture.png", dpi=300, bbox_inches="tight")
plt.close(fig); print("saved hybrid_architecture.png")

# ---------------- API-attack lifecycle ----------------
fig, ax = plt.subplots(figsize=(5.6, 7.5)); ax.set_xlim(0, 10); ax.set_ylim(0, 17.5); ax.axis("off")
stages = [
    ("External API client  (legitimate consumer or attacker)", NAVY, "#FFFFFF"),
    ("Exposed 5G / MEC API endpoint\n(service exposure, MEC application, management)", NAVY, "#FFFFFF"),
    ("Repeated unauthorised or rejected requests", STEP, LIGHT),
    ("Automated client-credential abuse\nand low-rate request campaigns", STEP, LIGHT),
    ("Access-token replay and misuse", STEP, LIGHT),
    ("Behavioural signature: retry bursts,\nfailure-rate spikes, short inter-arrival times", STEP, LIGHT),
    ("Detection  →  k-of-n policy  →  mitigation", ACT, RED_L),
]
y = 16.3
for i, (txt, edge, fill) in enumerate(stages):
    box(ax, 5, y, 8.6, 1.35, txt, edge=edge, fill=fill, fs=9.5)
    if i < len(stages) - 1:
        arrow(ax, 5, y - 0.72, 5, y - 1.62)
    y -= 2.3
ax.set_title("API-layer attack lifecycle and observable signals", fontsize=11.5, color=NAVY, weight="bold", pad=4)
fig.tight_layout(); fig.savefig("results/figures/api_attack_flow.png", dpi=300, bbox_inches="tight")
plt.close(fig); print("saved api_attack_flow.png")

# ---------------- Global system architecture ----------------
fig, ax = plt.subplots(figsize=(7.2, 5.9)); ax.set_xlim(0, 10); ax.set_ylim(2.9, 14.1); ax.axis("off")

# Edge-node column (local detection and enforcement path)
box(ax, 3.4, 13.2, 4.4, 0.95, "External API clients\n(applications, enterprise consumers)",
    edge=NAVY, fill="#FFFFFF", fs=8.5)
edge_stages = [
    "5G / MEC API gateway\n(service-exposure and management APIs)",
    "Feature Extractor  (flow features)",
    "1D-CNN detector  (TFLite, on-node)",
    "Mitigation Agent  (k-of-n policy)",
    "Federated Client  (local training)",
]
y = 11.6
for txt in edge_stages:
    box(ax, 3.4, y, 4.4, 0.95, txt, fs=9)
    arrow(ax, 3.4, y + 1.18, 3.4, y + 0.52)
    y -= 1.7
# dashed boundary marking what runs on the MEC node
ax.add_patch(FancyBboxPatch((0.95, 4.15), 4.95, 8.05, boxstyle="round,pad=0.08",
                            linewidth=1.2, edgecolor=STEP, facecolor="none", linestyle=(0, (5, 4))))
ax.text(0.95, 12.42, "MEC edge node", fontsize=8, color=STEP, style="italic", ha="left")

# Aggregation side
arrow(ax, 5.68, 5.05, 6.5, 6.05)
box(ax, 8.2, 6.7, 3.2, 1.15, "Aggregation Server\n(FedAvg + norm filter)", edge=NAVY, fill="#FFFFFF", fs=8.5)
arrow(ax, 8.2, 7.32, 8.2, 8.45)
box(ax, 8.2, 9.1, 3.2, 0.95, "Global Model", edge=ACT, fill=RED_L, fs=9)

# Return path: global model redistributed to the edge detectors
ret = FancyArrowPatch((6.58, 9.2), (5.66, 8.35), arrowstyle="-|>", mutation_scale=14,
                      linewidth=1.5, color=ACT, linestyle=(0, (5, 3)),
                      connectionstyle="arc3,rad=0.3")
ax.add_patch(ret)
ax.text(7.05, 9.88, "updated edge models", fontsize=8, color=ACT, style="italic", ha="center")

ax.text(5.0, 3.42, "Weight updates only: raw traffic never leaves the node",
        fontsize=8.5, color=STEP, style="italic", ha="center")
ax.set_title("Edge-native adaptive security framework", fontsize=12, color=NAVY, weight="bold", pad=6)
fig.tight_layout(); fig.savefig("results/figures/system_architecture.png", dpi=300, bbox_inches="tight")
plt.close(fig); print("saved system_architecture.png")
