"""Round-5 review, critical item 5: define and sweep the static rate-limiter baseline.

For each candidate rule the script reports ROC-AUC, the operating point at a threshold tuned on the
validation split (maximum F1), the operating point at the fixed 99th benign percentile used so far,
and the best F1 reachable on the test split (an oracle upper bound, labelled as such). It also
reports the raw benign and attack distributions of the rate features, and runs the best rule through
the same k-of-n policy and scenario streams as the deployed model.

The processed flows carry no identity or timestamp field, so the identity key and time window of a
gateway rule cannot be varied here; the output says so.

Run: PYTHONPATH=. .venv/bin/python scripts/baseline_sweep.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.mitigation.agent import MitigationPolicy, run_real_scenarios
from src.models.io import load_processed_split
from src.utils.config import load_yaml, resolve_path

M = Path("results/metrics")
FEATS = load_yaml("configs/model.yaml")["data"]["features"] if "data" in load_yaml("configs/model.yaml") else None
proc = resolve_path(load_yaml("configs/paths.yaml")["data"]["processed"])
(Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = (load_processed_split(proc / f"{s}.npz") for s in ("train", "val", "test"))
flat = lambda X: X.reshape(X.shape[0], -1)
Xtr, Xva, Xte = flat(Xtr), flat(Xva), flat(Xte)
scaler = pickle.load(open(proc / "scaler.pkl", "rb"))
names = ["Header_Length", "Protocol Type", "Time_To_Live", "Rate", "ack_count", "syn_count",
         "Tot sum", "AVG", "IAT", "Number"]
IDX = {n: i for i, n in enumerate(names)}

RULES = {  # each rule maps scaled features to a score where higher means more suspicious
    "rate": lambda X: X[:, IDX["Rate"]],
    "syn_count": lambda X: X[:, IDX["syn_count"]],
    "max(rate, syn_count)": lambda X: np.maximum(X[:, IDX["Rate"]], X[:, IDX["syn_count"]]),
    "short inter-arrival (-IAT)": lambda X: -X[:, IDX["IAT"]],
}


def point(y, s, t):
    p = (s >= t).astype(int)
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": float(t), "precision": prec, "recall": rec, "f1": f1,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0}


def best_f1(y, s):
    grid = np.unique(np.quantile(s, np.linspace(0.0, 0.9999, 400)))
    return max((point(y, s, t) for t in grid), key=lambda r: r["f1"])


out = {"note": "no identity or timestamp field in the processed flows, so the rule's identity key and "
               "time window cannot be varied on this data", "rules": {}}
for name, rule in RULES.items():
    s_tr, s_va, s_te = rule(Xtr), rule(Xva), rule(Xte)
    tuned = best_f1(yva, s_va)
    fixed = float(np.percentile(s_tr[ytr == 0], 99.0))
    out["rules"][name] = {
        "roc_auc": float(roc_auc_score(yte, s_te)),
        "validation_tuned": point(yte, s_te, tuned["threshold"]),
        "fixed_99th_benign_percentile": point(yte, s_te, fixed),
        "oracle_best_test_f1": best_f1(yte, s_te),
    }

# raw distributions: invert the min-max scaling for the rate-type features
raw = scaler.inverse_transform(Xte)
q = [5, 25, 50, 75, 95, 99]
out["distributions"] = {}
for f in ("Rate", "syn_count", "IAT"):
    v = raw[:, IDX[f]]
    b, a = v[yte == 0], v[yte == 1]
    out["distributions"][f] = {"quantiles": q,
                               "benign": [float(x) for x in np.percentile(b, q)],
                               "attack": [float(x) for x in np.percentile(a, q)],
                               "attack_above_benign_p99": float((a > np.percentile(b, 99)).mean())}

# the best rule through the same k-of-n policy and scenario streams as the deployed model
best_name = max(out["rules"], key=lambda n: out["rules"][n]["roc_auc"])
s_te = RULES[best_name](Xte)
thr = out["rules"][best_name]["validation_tuned"]["threshold"]
pol = MitigationPolicy(threshold=thr, window=10, min_hits=3, action="block", cooldown_flows=100)
r = run_real_scenarios(s_te[yte == 0], s_te[yte == 1], pol, 0.0, n_scenarios=200)
out["policy_k3_n10"] = {"rule": best_name, "threshold": thr,
                        **{k: r[k] for k in ("detection_rate", "conditional_detection_rate",
                                             "false_block_rate", "ttd_flows_median")}}

(M / "baseline_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote", M / "baseline_sweep.json")
for n, v in out["rules"].items():
    t, f = v["validation_tuned"], v["fixed_99th_benign_percentile"]
    print(f"  {n:28} ROC-AUC {v['roc_auc']:.4f} | tuned: F1 {t['f1']:.3f} recall {t['recall']:.3f} FPR {t['false_positive_rate']:.4f}"
          f" | fixed p99: F1 {f['f1']:.3f} recall {f['recall']:.3f} | oracle F1 {v['oracle_best_test_f1']['f1']:.3f}")
for f, d in out["distributions"].items():
    print(f"  {f:10} median benign {d['benign'][2]:.3g} attack {d['attack'][2]:.3g}; attacks above benign p99: {d['attack_above_benign_p99']:.3f}")
print("  policy:", out["policy_k3_n10"])
