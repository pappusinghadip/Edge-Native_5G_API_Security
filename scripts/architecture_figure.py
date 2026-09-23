"""Figure 3.1, round-5 version: deployment boundaries of the framework.

Adds what the round-5 review (Section 6, "Section 3.2 architecture") asks the figure to show: TLS
termination, feature extraction, identity-keyed policy state, the enforcement hook, model signing and
rollback, and the trust boundaries around the edge node and the aggregator. Solid boxes are implemented and
evaluated in Chapter 4; dashed boxes are specified by the design but not implemented, so the figure does not
claim more than the evaluation shows. The tree models appear only as an offline centralized reference.

Run: .venv/bin/python scripts/architecture_figure.py
"""
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

NAVY, STEP, RED, LIGHT, GREY = "#14315C", "#2E4C7E", "#B23A3A", "#EAF0F8", "#8A8A8A"
FS = 8.5
fig, ax = plt.subplots(figsize=(7.2, 6.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 90); ax.axis("off")


def box(cx, cy, w, h, text, built=True, fill=LIGHT, edge=STEP, fs=FS, color="#1a1a1a"):
    """built: True = implemented and evaluated (filled), "stub" = in-memory stub (white, solid outline),
    False = design only (white, dashed outline). Fill and dash both differ, so grayscale print keeps them apart."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.4",
                                linewidth=1.3, edgecolor=edge, linestyle=(0, (4, 2.5)) if built is False else "-",
                                facecolor=fill if built is True else "white"))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.25)


def arrow(p, q, color=NAVY, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, linewidth=1.3, color=color,
                                 linestyle=(0, (4, 2.5)) if dashed else "-", shrinkA=0, shrinkB=0))


def zone(x0, y0, x1, y1):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=RED, linewidth=1.2,
                           linestyle=(0, (1, 1.8))))


def note(x, y, text, color=STEP, ha="center", fs=FS - 1, rot=0):
    ax.text(x, y, text, fontsize=fs, color=color, style="italic", ha=ha, va="center", linespacing=1.2,
            rotation=rot)


L, R, A = 21, 50.25, 85.5                 # column centres: request path, learning path, aggregator
LW, RW, AW = 30, 23.5, 24
# trust boundaries
zone(1, 12, 63.5, 74); note(26, 76, "Trust boundary 1: MEC edge node (operator)", color=RED, ha="left")
zone(71.5, 23, 99.5, 74); note(72.5, 77.3, "Trust boundary 2:\naggregation domain", color=RED, ha="left")

# untrusted side
box(L, 83.5, 40, 6.4, "External API clients\n(applications, enterprise consumers)", fill="white", edge=NAVY)
arrow((L, 80.1), (L, 72.4)); note(L - 1.5, 76.5, "HTTPS", ha="right")
note(L + 21, 83.5, "untrusted\npublic network", color=RED, ha="left", fs=FS - 1.5)

# request and decision path
box(L, 66, LW, 10.4, "API gateway\nTLS termination,\nauthentication,\nflow-record export", built=False)
box(L, 54, LW, 6.4, "Feature selection, scaling\n(10 flow features)")
box(L, 42, LW, 6.4, "Federated 1D-CNN\n(TFLite, on node)")
box(L, 30, LW, 8.4, "Mitigation agent\nk-of-n state, one\ninstance per identity")
box(L, 18, LW, 8.4, "Enforcement hook\nblock recorded in memory;\ngateway call not built", built="stub")
for y0, y1 in ((60.4, 57.6), (50.4, 45.6), (38.4, 34.6), (25.4, 22.6)):
    arrow((L, y0), (L, y1))
ax.plot([L - LW / 2 - 0.4, 3.0, 3.0], [18, 18, 66], color=RED, lw=1.3)
arrow((3.0, 66), (L - LW / 2 - 0.4, 66), color=RED)
note(4.4, 42, "enforce", color=RED, rot=90)

# learning path on the node
box(R, 66, RW, 6.4, "Federated client\n(local training)")
box(R, 54, RW, 6.4, "Local flow records\n(raw data stays here)")
box(R, 42, RW, 8.4, "Model verifier\nsignature check,\nrollback to last good", built=False)
arrow((L + LW / 2 + 0.4, 54), (R - RW / 2 - 0.4, 54))
arrow((R, 57.6), (R, 62.4))
arrow((R - RW / 2 - 0.4, 42), (L + LW / 2 + 0.4, 42), color=RED, dashed=True)

# aggregation domain
box(A, 66, AW, 6.4, "Norm filter + FedAvg\n(κ = 2.5)")
box(A, 42, AW, 6.4, "Model registry\nversioning, signing", built=False)
arrow((A, 62.4), (A, 45.6))
note(A, 31, "Honest-but-curious server:\nsees each client's update\n(no secure aggregation)", fs=FS - 1.2)
arrow((R + RW / 2 + 0.4, 66), (A - AW / 2 - 0.4, 66), dashed=True)
note(67.5, 70.3, "updates,\nmutual TLS", fs=FS - 2.3)
arrow((A - AW / 2 - 0.4, 42), (R + RW / 2 + 0.4, 42), color=RED, dashed=True)
note(67.5, 46.3, "signed\nmodel", color=RED, fs=FS - 2)

# offline reference, outside the deployment
box(32.25, 5, 62.5, 6.4, "Centralized reference, offline and not deployed:\n"
    "tree models, weighted and stacked ensembles (Chapter 4)", fill="#F2F2F2", edge=GREY, color="#444444", fs=FS - 0.5)

ax.legend(handles=[Patch(facecolor=LIGHT, edgecolor=STEP, label="implemented and evaluated"),
                   Patch(facecolor="white", edgecolor=STEP, label="in-memory stub"),
                   Patch(facecolor="white", edgecolor=STEP, linestyle="--", label="design only, not implemented"),
                   Line2D([], [], color=RED, lw=1.2, ls=(0, (1, 1.8)), label="trust boundary"),
                   Line2D([], [], color=NAVY, lw=1.3, label="request path"),
                   Line2D([], [], color=NAVY, lw=1.3, ls=(0, (4, 2.5)), label="model exchange")],
          loc="lower right", bbox_to_anchor=(1.0, -0.01), fontsize=FS - 1.3, frameon=False, handlelength=2.2)

fig.savefig("results/figures/system_architecture_r5.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("results/figures/system_architecture_r5.svg", bbox_inches="tight", facecolor="white")
print("saved system_architecture_r5.png and .svg")
