"""Round-6 review: the principal comparison on the cleaned feature set, next to the original (both on the Mac).

Reads the ten *_r6clean_* runs (scripts/run_round6_clean.sh), the corrected-loss Mac 60N pair on the original
data (*_r5mac_*), and the five-seed baselines (scripts/baselines_seeds.py) on both data versions. The two data
versions have different test sets (duplicates removed), so cross-version differences are unpaired descriptions;
the centralized-federated gap is paired by seed within each version.

Run: PYTHONPATH=. .venv/bin/python scripts/round6_clean_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

M = Path("results/metrics")
SEEDS = [101, 102, 103, 104, 105]
DELTAS = [0.02, 0.01, 0.005]


def arm(stem: str) -> dict:
    runs = {s: json.loads((M / f"{stem}_s{s}.json").read_text())["runs"][0] for s in SEEDS}
    auc = np.array([runs[s]["auc_roc"] for s in SEEDS])
    return {"per_seed": {s: runs[s]["auc_roc"] for s in SEEDS}, "mean": float(auc.mean()), "std": float(auc.std(ddof=1)),
            "f1_mean": float(np.mean([runs[s]["f1_binary"] for s in SEEDS])),
            "fpr_mean": float(np.mean([runs[s]["false_positive_rate"] for s in SEEDS])),
            "wall_minutes_median": float(np.median([runs[s].get("wall_seconds", np.nan) for s in SEEDS]) / 60),
            "sample_exposure": runs[SEEDS[0]].get("sample_exposure")}


def paired(c: dict, f: dict) -> dict:
    d = np.array([c["per_seed"][s] - f["per_seed"][s] for s in SEEDS]); n, sd = len(d), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    upper1 = d.mean() + stats.t.ppf(0.95, n - 1) * sd / np.sqrt(n)
    return {"mean_difference": float(d.mean()), "diff_ci95": [float(d.mean() - half), float(d.mean() + half)],
            "one_sided_upper95": float(upper1), "cohens_dz": float(d.mean() / sd),
            "paired_t_p_two_sided": float(stats.ttest_rel([c["per_seed"][s] for s in SEEDS],
                                                          [f["per_seed"][s] for s in SEEDS]).pvalue),
            "non_inferior": {str(m): bool(d.mean() + half < m) for m in DELTAS}}


out = {"platform": "Mac (Apple M4 Pro), 4 threads per job", "exposure": "60N: centralized 60 epochs, federated 20 x 3"}
for tag, cstem, fstem in (("original", "repeated_centralized_r5mac_e60", "repeated_federated_iid_r5mac_le3"),
                          ("clean", "repeated_centralized_r6clean_e60", "repeated_federated_iid_r6clean_le3")):
    c, f = arm(cstem), arm(fstem)
    base = json.loads((M / f"baselines_seeds_{tag}.json").read_text())["models"]
    out[tag] = {"centralized": c, "federated": f, "paired": paired(c, f),
                "baselines": {m: {"mean": v["summary"]["auc_roc"]["mean"], "std": v["summary"]["auc_roc"]["std"],
                                  "seeds": len(v["seeds"])} for m, v in base.items()}}
meta = json.loads(Path("data/processed_clean/preprocessing_metadata.json").read_text())
out["clean_data"] = {k: meta[k] for k in ("exact_duplicates_removed", "duplicates_removed_by_class",
                                           "rows_with_conflicting_labels_kept", "split_sizes", "changes")}
out["test_flows"] = {"original": int(np.load("data/processed/test.npz")["y"].size),
                     "clean": meta["split_sizes"]["test"]["rows"]}
(M / "round6_clean.json").write_text(json.dumps(out, indent=1))
for tag in ("original", "clean"):
    o = out[tag]; p = o["paired"]
    print(f"{tag:8s} cen {o['centralized']['mean']:.4f} fed {o['federated']['mean']:.4f} diff {p['mean_difference']:.4f} "
          f"CI [{p['diff_ci95'][0]:.4f}, {p['diff_ci95'][1]:.4f}] upper1 {p['one_sided_upper95']:.4f} dz {p['cohens_dz']:.1f} "
          f"NI {p['non_inferior']}  XGB {o['baselines']['XGBoost']['mean']:.4f} MLP {o['baselines']['MLP']['mean']:.4f}")
