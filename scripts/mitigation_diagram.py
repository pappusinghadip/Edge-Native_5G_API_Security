"""Render the detection-to-mitigation flow diagram (review comment: weakness 2)."""
import os, tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY = "#14315C"; STEP = "#2E4C7E"; ACT = "#B23A3A"; OK = "#1E7B34"; LIGHT = "#EAF0F8"

fig, ax = plt.subplots(figsize=(5.4, 5.6))
ax.set_xlim(0, 10); ax.set_ylim(5.75, 14.75); ax.axis("off")
BW = 6.8   # box width; the figure prints at about 4.8 in, so text stays near body size

def box(y, text, color=STEP, fill=LIGHT, w=BW, h=1.15, x=5.0):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.08",
                                linewidth=1.4, edgecolor=color, facecolor=fill))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, color="#1a1a1a")

def arrow(y1, y2, x=5.0, color=NAVY, label=None, lx=None):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color=color))
    if label:
        ax.text((lx if lx is not None else x) + 0.25, (y1 + y2) / 2, label, fontsize=9,
                color=color, ha="left", va="center", style="italic")

# Vertical pipeline
box(14.0, "Incoming API flow\n(request rate, inter-arrival, retries, flags)", color=NAVY, fill="#FFFFFF")
arrow(13.36, 12.79)
box(12.15, "Detection: 1D-CNN (TFLite, on-node)\noutputs a malicious probability (risk score)", color=STEP)
arrow(11.51, 10.94)
box(10.3, "Policy engine: k-of-n window per identity\n(account, client ID or token); block when k\nof the last n flows are flagged (k = 3, n = 10)",
    color=STEP, h=1.35)
arrow(9.56, 9.09, label="sustained evidence")
box(8.45, "Enforcement: rate-limit or temporary block\non the offending source", color=ACT, fill="#FBEAEA")
arrow(7.81, 7.24)
box(6.6, "Recovery timer: auto-release after a\ncooldown window (no permanent lockout)", color=STEP)

# "allow" side branch from the policy engine
ax.add_patch(FancyArrowPatch((8.5, 10.3), (9.95, 10.3), arrowstyle="-|>", mutation_scale=12,
                             linewidth=1.4, color=OK))
ax.text(9.2, 10.55, "below k:\nallow", fontsize=9, color=OK, ha="center", va="bottom")

# Feedback loop: recovery -> back to monitoring
loop = FancyArrowPatch((1.5, 6.6), (1.5, 14.0), connectionstyle="arc3,rad=-0.35",
                       arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color="#888888", linestyle="--")
ax.add_patch(loop)
ax.text(0.95, 10.3, "resume monitoring", fontsize=9, color="#666666",
        ha="center", va="center", rotation=90)

ax.set_title("Detection-to-mitigation pipeline (edge node)", fontsize=11, color=NAVY, weight="bold", pad=4)
fig.tight_layout()
fig.savefig("results/figures/mitigation_flow.png", dpi=300, bbox_inches="tight")
print("saved results/figures/mitigation_flow.png")
