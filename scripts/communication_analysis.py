"""Phase 3, Task 3.2/3.3 — communication overhead and privacy analysis.

Federated Learning transmits only model weight updates; a centralized design must
transmit raw traffic. This script quantifies both for the experimental setup and
extrapolates to operational 5G-edge scale, and frames the privacy argument.

Run: python -m scripts.communication_analysis   (or python scripts/communication_analysis.py)
"""

from __future__ import annotations

import json
import pathlib
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"

# Measured float32 weight size of the 1D-CNN. Taken from measured_communication.json when that
# has been produced, so this script and the measured figures cannot drift apart.
def _model_bytes() -> int:
    p = pathlib.Path("results/metrics/measured_communication.json")
    if p.exists():
        return int(json.loads(p.read_text())["model"]["serialized_float32_bytes"])
    return 697_864


MODEL_BYTES = _model_bytes()
TRAIN_ROWS = 777_833           # processed training rows
FEATURES = 10


def model_update_mb() -> float:
    return MODEL_BYTES / 1e6


def fl_total_mb(k_clients: int, rounds: int) -> float:
    # Each round: server sends the global model down and each client sends its update up.
    return 2.0 * model_update_mb() * k_clients * rounds


def raw_train_mb_on_disk() -> float:
    """Actual raw CSV volume for the training portion (what a centralized design ships)."""
    raw_dir = ROOT / "data" / "raw"
    total = sum(p.stat().st_size for p in raw_dir.glob("*.csv"))
    return (total / 1e6) * 0.70  # 70% train split


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    update_mb = model_update_mb()
    per_round_per_client = 2.0 * update_mb  # down + up

    # --- Experimental setup (K=5, R=20) ---
    k, r = 5, 20
    fl_mb = fl_total_mb(k, r)
    raw_mb = raw_train_mb_on_disk()

    # --- Operational extrapolation (per revised plan, Q1) ---
    # A MEC site sees ~50 GB/day of flow data; FL sends a fixed few MB per round.
    daily_traffic_gb = 50.0
    rounds_per_day = 96  # e.g., one aggregation round every 15 minutes
    fl_per_node_per_day_mb = per_round_per_client * rounds_per_day
    centralized_per_node_per_day_mb = daily_traffic_gb * 1000.0
    reduction_factor = centralized_per_node_per_day_mb / fl_per_node_per_day_mb

    results = {
        "model_update_mb": round(update_mb, 4),
        "per_round_per_client_mb": round(per_round_per_client, 4),
        "experimental": {
            "k_clients": k, "rounds": r,
            "fl_total_transfer_mb": round(fl_mb, 2),
            "centralized_raw_upload_mb": round(raw_mb, 2),
            "note": ("On this small dataset the two are comparable; FL's advantage is "
                     "operational, not toy-scale — see the extrapolation below and the "
                     "privacy argument."),
        },
        "operational_per_node_per_day": {
            "assumed_daily_flow_traffic_gb": daily_traffic_gb,
            "assumed_rounds_per_day": rounds_per_day,
            "fl_transfer_mb": round(fl_per_node_per_day_mb, 2),
            "centralized_transfer_mb": round(centralized_per_node_per_day_mb, 2),
            "reduction_factor_x": round(reduction_factor, 1),
        },
        "privacy": {
            "raw_records_leaving_node_fl": 0,
            "raw_records_leaving_node_centralized": TRAIN_ROWS,
            "argument": ("Under FL, no raw subscriber metadata (addresses, timing, session "
                         "content) leaves the edge node — only model parameters do, which "
                         "supports GDPR compliance. Weight sharing is not perfectly private "
                         "(gradient-inversion and membership-inference attacks exist), so the "
                         "thesis assumes an honest-but-curious model and identifies secure "
                         "aggregation as future work."),
        },
    }
    out = METRICS / "communication_analysis.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- Figure: transfer cost vs daily traffic volume (log-log), FL flat vs centralized linear ---
    traffic_mb = np.logspace(0, 7, 200)  # 1 MB .. 10 TB per day
    fl_line = np.full_like(traffic_mb, fl_per_node_per_day_mb)
    centralized_line = traffic_mb
    crossover = fl_per_node_per_day_mb

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.loglog(traffic_mb, centralized_line, label="Centralized (ship raw traffic)", color="#B23A3A")
    ax.loglog(traffic_mb, fl_line, label=f"Federated (weights only, {fl_per_node_per_day_mb:.0f} MB/day)", color="#14315C")
    ax.axvline(crossover, ls=":", color="gray")
    ax.annotate("crossover", (crossover, crossover), textcoords="offset points", xytext=(6, 6), fontsize=8, color="gray")
    ax.set_xlabel("Daily flow traffic per edge node (MB, log scale)")
    ax.set_ylabel("Data transferred per node per day (MB, log)")
    ax.set_title("Communication cost: Federated vs Centralized")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig_path = FIGURES / "communication_overhead.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"per-round per-client: {per_round_per_client:.2f} MB")
    print(f"experimental FL total (K=5,R=20): {fl_mb:.1f} MB  vs centralized raw: {raw_mb:.1f} MB")
    print(f"operational reduction: {reduction_factor:.0f}x  (FL {fl_per_node_per_day_mb:.0f} MB/day vs {centralized_per_node_per_day_mb:.0f} MB/day)")
    print("saved", out, "and", fig_path)


if __name__ == "__main__":
    main()
