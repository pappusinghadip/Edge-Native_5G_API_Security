"""Round-6 review, Section 8.3: FedBN sensitivity on the most skewed partition (Dirichlet alpha = 0.1).

For each seed, plain FedAvg (every variable averaged, as in the thesis) and FedBN (batch-normalisation
variables kept local) are trained on the same saved partition at 20N (20 rounds x 1 local epoch). FedAvg is
scored as its global model; FedBN has no single global model, so each client's model (shared layers plus
its own BN) is scored on the pooled test set, each at a threshold tuned on the pooled validation split.
Output: results/metrics/fedbn_sensitivity.json (resumable per seed).

Run: PYTHONPATH=. .venv/bin/python scripts/fedbn_sensitivity.py --seeds 101 102 103
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.fl.simulate import best_threshold, evaluate_global, load_clients, load_global_splits, make_model, run_fedavg
from src.utils.config import load_yaml
from src.utils.seed import set_global_seed

OUT = Path("results/metrics/fedbn_sensitivity.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=20)
    a = ap.parse_args()
    fl, mdl, pth = (load_yaml(f"configs/{n}.yaml") for n in ("fl", "model", "paths"))
    fl["federation"]["local_epochs"] = 1
    gs = load_global_splits(pth)
    out = json.loads(OUT.read_text()) if OUT.exists() else {
        "alpha": a.alpha, "rounds": a.rounds, "local_epochs": 1, "exposure": "20N", "seeds": {}}
    for s in a.seeds:
        if str(s) in out["seeds"]:
            continue
        rec = {}
        for arm in ("fedavg", "fedbn"):
            set_global_seed(s); t0 = time.time()
            clients = load_clients(pth, "dirichlet", int(fl["federation"]["num_clients"]), alpha=a.alpha)
            res = run_fedavg(clients, gs, fl, mdl, f"dir_{a.alpha}", a.rounds, fedbn=(arm == "fedbn"))
            if arm == "fedavg":
                rec[arm] = {"auc_roc": res.final["auc_roc"], "f1": res.final["f1_binary"],
                            "wall_seconds": round(time.time() - t0, 1)}
            else:
                m = make_model(fl, mdl); per = {}
                for cid, w in sorted(res._client_weights.items()):
                    m.set_weights(w)
                    t = best_threshold(m, *gs["val"])
                    ev = evaluate_global(m, *gs["test"], threshold=t)
                    per[str(cid)] = {"auc_roc": ev["auc_roc"], "f1": ev["f1_binary"], "rows": int(len(clients[cid].y))}
                aucs = [v["auc_roc"] for v in per.values()]
                rec[arm] = {"per_client": per, "mean_auc": float(np.mean(aucs)), "min_auc": float(min(aucs)),
                            "max_auc": float(max(aucs)), "wall_seconds": round(time.time() - t0, 1)}
            print(f"seed {s} {arm}: {rec[arm].get('auc_roc', rec[arm].get('mean_auc')):.4f}", flush=True)
        out["seeds"][str(s)] = rec
        OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
