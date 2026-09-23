"""Round-6 step 5: summarise the CNN feature study and the FedBN sensitivity run.

Feature study (Mac, centralized 1D-CNN, 20N, seeds 101-103): original feature order, a fixed random order, and
the original order without Time_To_Live; each variant is paired with the original by seed. FedBN (Mac, Dirichlet
alpha 0.1, 20N): the FedAvg global model against the mean of the FedBN clients' models, paired by seed.
Output: results/metrics/step5_analysis.json.

Run: PYTHONPATH=. .venv/bin/python scripts/step5_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

M = Path("results/metrics")
SEEDS = [101, 102, 103]


def ci(d: np.ndarray) -> list[float]:
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - h), float(d.mean() + h)]


def run(v: str, s: int) -> dict:
    return json.loads((M / f"repeated_centralized_r6feat_{v}_e20_s{s}.json").read_text())["runs"][0]


out = {"feature_study": {"exposure": "20N (20 centralized epochs)", "seeds": SEEDS, "variants": {}}}
auc = {v: np.array([run(v, s)["auc_roc"] for s in SEEDS]) for v in ("orig", "perm", "nottl")}
for v in auc:
    out["feature_study"]["variants"][v] = {
        "auc_per_seed": dict(zip(map(str, SEEDS), auc[v].tolist())), "auc_mean": float(auc[v].mean()),
        "auc_std": float(auc[v].std(ddof=1)),
        "f1_mean": float(np.mean([run(v, s)["f1_binary"] for s in SEEDS])),
        "features": json.loads((M / f"repeated_centralized_r6feat_{v}_e20_s{SEEDS[0]}.json").read_text()).get("features")}
    if v != "orig":
        d = auc[v] - auc["orig"]
        out["feature_study"]["variants"][v]["minus_original"] = {"mean": float(d.mean()), "ci95": ci(d),
                                                                "per_seed": d.tolist()}
fb = json.loads((M / "fedbn_sensitivity.json").read_text())
seeds = sorted(fb["seeds"], key=int)
fa = np.array([fb["seeds"][s]["fedavg"]["auc_roc"] for s in seeds])
bn = np.array([fb["seeds"][s]["fedbn"]["mean_auc"] for s in seeds])
out["fedbn"] = {"alpha": fb["alpha"], "exposure": fb["exposure"], "seeds": [int(s) for s in seeds],
                "fedavg_auc_mean": float(fa.mean()), "fedavg_auc_std": float(fa.std(ddof=1)),
                "fedbn_client_mean_auc": float(bn.mean()), "fedbn_client_mean_auc_std": float(bn.std(ddof=1)),
                "fedbn_worst_client_auc": float(min(fb["seeds"][s]["fedbn"]["min_auc"] for s in seeds)),
                "fedbn_best_client_auc": float(max(fb["seeds"][s]["fedbn"]["max_auc"] for s in seeds)),
                "fedbn_minus_fedavg": {"mean": float((bn - fa).mean()), "ci95": ci(bn - fa)}}
(M / "step5_analysis.json").write_text(json.dumps(out, indent=1))
print(json.dumps({v: (round(x["auc_mean"], 4), x.get("minus_original")) for v, x in out["feature_study"]["variants"].items()}))
print(json.dumps({k: v for k, v in out["fedbn"].items() if k != "seeds"}))
