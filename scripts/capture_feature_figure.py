"""Round-9 review: show the two capture-sensitive features by class.

Reads Time_To_Live and Header_Length from the raw CICIoT2023 CSVs of the thesis subset (DictionaryBruteForce
against the four benign captures; values are means over the dataset's ten-packet windows) and draws their
class-conditional distributions, plus a few quantiles in results/metrics/capture_features.json.

Run: PYTHONPATH=. .venv/bin/python scripts/capture_feature_figure.py
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
import pandas as pd

RAW = Path("data/raw")
COLS = ["Time_To_Live", "Header_Length"]
read = lambda f: pd.read_csv(RAW / f, usecols=COLS).replace([np.inf, -np.inf], np.nan).dropna()
attack = read("DictionaryBruteForce.pcap.csv")
benign = pd.concat([read(f) for f in ("BenignTraffic.pcap.csv", "BenignTraffic1.pcap.csv",
                                      "BenignTraffic2.pcap.csv", "BenignTraffic3.pcap.csv")])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
out = {"source": "raw CSVs, thesis subset", "rows": {"attack": len(attack), "benign": len(benign)}, "quantiles": {}}
for ax, col in zip(axes, COLS):
    a, b = attack[col].to_numpy(), benign[col].to_numpy()
    hi = np.percentile(np.concatenate([a, b]), 99)          # clip the long tail for readability
    bins = np.linspace(0, hi, 60)
    ax.hist(b, bins=bins, density=True, histtype="step", linewidth=1.6, color="#14315C", label="benign")
    ax.hist(a, bins=bins, density=True, histtype="step", linewidth=1.6, linestyle="--", color="#B23A3A",
            label="DictionaryBruteForce")
    ax.set_xlabel(f"{col} (window mean)", fontsize=9)
    ax.set_ylabel("density", fontsize=9); ax.tick_params(labelsize=8); ax.grid(alpha=0.25)
    out["quantiles"][col] = {cls: {q: float(np.percentile(v, q)) for q in (5, 25, 50, 75, 95)}
                             for cls, v in (("attack", a), ("benign", b))}
axes[0].legend(fontsize=8.5, frameon=True)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"results/figures/capture_features.{ext}", dpi=300)
Path("results/metrics/capture_features.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["rows"]), {c: {k: v[50] for k, v in out["quantiles"][c].items()} for c in COLS})
