"""Render the detection-to-mitigation flow diagram (review comment: weakness 2)."""
import os, tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY = "#14315C"; STEP = "#2E4C7E"; ACT = "#B23A3A"; OK = "#1E7B34"; LIGHT = "#EAF0F8"

fig, ax = plt.subplots(figsize=(6.6, 6.0))
ax.set_xlim(0, 10); ax.set_ylim(5.6, 14.9); ax.axis("off")

def box(y, text, color=STEP, fill=LIGHT, w=5.2, h=1.15, x=5.0):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.08",
                                linewidth=1.6, edgecolor=color, facecolor=fill))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, color="#1a1a1a", wrap=True)

def arrow(y1, y2, x=5.0, color=NAVY, label=None, lx=None):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color=color))
    if label:
        ax.text((lx if lx is not None else x) + 0.25, (y1 + y2) / 2, label, fontsize=8.2,
                color=color, ha="left", va="center", style="italic")

# Vertical pipeline
box(14.0, "Incoming API flow\n(request rate, inter-arrival, retries, flags)", color=NAVY, fill="#FFFFFF")
arrow(13.42, 12.75)
box(12.15, "Detection  —  1D-CNN / CNN+GBM ensemble\noutputs a malicious probability (risk score)", color=STEP)
arrow(11.57, 10.9)
box(10.3, "Policy engine  —  k-of-n sliding window\n(block when ≥ 3 of the last 10 flows are flagged)", color=STEP)
arrow(9.72, 9.05, label="sustained\nevidence")
box(8.45, "Enforcement  —  rate-limit / temporary block\non the offending source", color=ACT, fill="#FBEAEA")
arrow(7.87, 7.2)
box(6.6, "Recovery timer  —  auto-release after a\ncooldown window (no permanent lockout)", color=STEP)

# "allow" side branch from the policy engine
ax.add_patch(FancyArrowPatch((7.6, 10.3), (9.0, 10.3), arrowstyle="-|>", mutation_scale=14,
                             linewidth=1.4, color=OK))
ax.text(9.05, 10.3, "below\nthreshold\n→ allow", fontsize=8.2, color=OK, ha="left", va="center")

# Feedback loop: recovery -> back to monitoring
loop = FancyArrowPatch((2.4, 6.6), (2.4, 14.0), connectionstyle="arc3,rad=-0.45",
                       arrowstyle="-|>", mutation_scale=15, linewidth=1.4, color="#888888", linestyle="--")
ax.add_patch(loop)
ax.text(0.75, 10.3, "resume\nmonitoring", fontsize=8.2, color="#888888", ha="center", va="center", rotation=90)

ax.set_title("Detection-to-mitigation pipeline (edge node)", fontsize=12, color=NAVY, weight="bold", pad=6)
fig.tight_layout()
fig.savefig("results/figures/mitigation_flow.png", dpi=300, bbox_inches="tight")
print("saved results/figures/mitigation_flow.png")
