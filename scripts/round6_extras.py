"""Round-6 review: numbers the text needs that come from saved data, with no training.

  static_validation_selected  the static heuristic whose scoring feature and threshold are chosen on the
                              validation split only, then evaluated once on test
  training_cost               optimizer steps, sample exposure and wall-clock time of every principal PC arm
  partitions                  per-client size, attack count, prevalence and label entropy for the IID and
                              Dirichlet 0.1 / 0.5 / 1.0 partitions (seed 42, as src/data/partitioner.py draws them)
  cnn_layers                  layer-by-layer table of the 1D-CNN

Run: PYTHONPATH=. .venv/bin/python scripts/round6_extras.py
"""
from __future__ import annotations

import glob
import json
import math
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, matthews_corrcoef,
                             roc_auc_score)

from src.data.partitioner import partition_dirichlet_indices, partition_iid_indices
from src.models.io import load_processed_split
from src.utils.config import load_yaml

M, P = Path("results/metrics"), Path("results/metrics_PC_run_2026-09-14_round5")
flat = lambda X: X.reshape(X.shape[0], -1)
(Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = (load_processed_split(Path(f"data/processed/{s}.npz"))
                                                for s in ("train", "val", "test"))
Xtr, Xva, Xte = flat(Xtr), flat(Xva), flat(Xte)
NAMES = ["Header_Length", "Protocol Type", "Time_To_Live", "Rate", "ack_count", "syn_count", "Tot sum", "AVG",
         "IAT", "Number"]
I = {n: i for i, n in enumerate(NAMES)}
RULES = {"rate": lambda X: X[:, I["Rate"]], "syn_count": lambda X: X[:, I["syn_count"]],
         "max(rate, syn_count)": lambda X: np.maximum(X[:, I["Rate"]], X[:, I["syn_count"]]),
         "short inter-arrival (-IAT)": lambda X: -X[:, I["IAT"]]}


def full_metrics(y, s, t):
    p = (s >= t).astype(int)
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    return {"roc_auc": float(roc_auc_score(y, s)), "average_precision": float(average_precision_score(y, s)),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "mcc": float(matthews_corrcoef(y, p)), "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "false_positive_rate": fp / (fp + tn), "threshold": float(t)}


def best_f1_threshold(y, s):
    grid = np.unique(np.quantile(s, np.linspace(0.0, 0.9999, 400)))
    def f1(t):
        p = s >= t; tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = (~p & (y == 1)).sum()
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return float(max(grid, key=f1))


out = {}
val_auc = {n: float(roc_auc_score(yva, r(Xva))) for n, r in RULES.items()}
chosen = max(val_auc, key=val_auc.get)
thr = best_f1_threshold(yva, RULES[chosen](Xva))
out["static_validation_selected"] = {"selection": "feature with the highest validation ROC-AUC among the four "
                                     "pre-declared rules; threshold maximising validation F1; test used once",
                                     "validation_roc_auc": val_auc, "chosen": chosen,
                                     "test": full_metrics(yte, RULES[chosen](Xte), thr)}

# ---- training cost of the principal PC arms --------------------------------------------------------- #
iid = [len(np.load(f"data/partitions/client_{k}_iid.npz")["y"]) for k in range(5)]
N = len(ytr)
fl, mdl = load_yaml("configs/fl.yaml")["federation"], load_yaml("configs/model.yaml")["training"]
def steps(arm, r):
    if arm.startswith("centralized"):
        return r["epochs_completed"] * math.ceil(N / int(mdl["batch_size"]))
    if arm.startswith("federated"):
        return 20 * r["local_epochs"] * sum(math.ceil(n / int(fl["local_batch_size"])) for n in iid)
    return 60 * sum(math.ceil(n / int(fl["local_batch_size"])) for n in iid)       # isolated: 5 clients alone
cost = {}
for arm, pat in (("centralized_20N", "repeated_centralized_r5_e20_s*.json"),
                 ("centralized_60N", "repeated_centralized_r5_e60_s*.json"),
                 ("federated_20N", "repeated_federated_iid_r5_le1_s*.json"),
                 ("federated_60N", "repeated_federated_iid_r5_le3_s*.json"),
                 ("isolated_60N", "fl_isolated_iid_r5_isolated_s*.json")):
    runs = []
    for f in sorted(glob.glob(str(P / pat))):
        d = json.load(open(f)); runs.append(d["runs"][0] if "runs" in d else d)
    r0 = runs[0]
    cost[arm] = {"optimizer": ("Adam, lr 0.001, batch %s" % mdl["batch_size"]) if arm.startswith("centralized")
                 else "SGD with momentum 0.9, lr %s, batch %s" % (fl["learning_rate"], fl["local_batch_size"]),
                 "optimizer_steps": steps(arm, {"epochs_completed": r0.get("epochs_completed"),
                                                "local_epochs": r0.get("local_epochs", 1)}),
                 "sample_exposure": r0.get("sample_exposure"),
                 "wall_seconds_median": (statistics.median(r["wall_seconds"] for r in runs)
                                         if all("wall_seconds" in r for r in runs) else None),   # isolated: not recorded
                 "wall_seconds_range": ([min(r["wall_seconds"] for r in runs), max(r["wall_seconds"] for r in runs)]
                                        if all("wall_seconds" in r for r in runs) else None),
                 "seeds": len(runs)}
out["training_cost"] = {"platform": "PC (AMD Ryzen 7 laptop, WSL 2, CPU)", "train_rows": N, "iid_client_rows": iid,
                        "arms": cost,
                        "note": "federated steps count every client's local steps; isolated counts the five clients' "
                                "steps together although no model combines them"}

# ---- partition statistics --------------------------------------------------------------------------- #
def stats(parts):
    rows = []
    for k, idx in enumerate(parts):
        n, a = int(len(idx)), int(ytr[idx].sum()); p = a / n if n else 0.0
        h = 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        rows.append({"client": k, "rows": n, "attacks": a, "prevalence": p, "label_entropy_bits": h})
    return rows
parts = {"iid": partition_iid_indices(N, 5, seed=42)}
for a in (0.1, 0.5, 1.0):
    parts[f"dirichlet_{a}"] = partition_dirichlet_indices(ytr, num_clients=5, alpha=a, seed=42)
for a, tag in ((0.1, "0p1"), (0.5, "0p5")):             # the redrawn partitions must equal the saved files
    saved = [len(np.load(f"data/partitions/client_{k}_dir_{tag}.npz")["y"]) for k in range(5)]
    assert saved == [len(i) for i in parts[f"dirichlet_{a}"]], (a, saved)
out["partitions"] = {k: stats(v) for k, v in parts.items()}
out["partitions_note"] = "Dirichlet 0.1 and 0.5 redraws checked against the saved partition files; 1.0 uses the same function and seed"

# ---- CNN layer table ---------------------------------------------------------------------------------- #
from src.models.cnn import build_model_from_config  # noqa: E402
model = build_model_from_config(load_yaml("configs/model.yaml"))
layers = []
for l in model.layers:
    shp = l.output_shape if hasattr(l, "output_shape") else None
    layers.append({"layer": l.__class__.__name__, "name": l.name, "output_shape": str(shp),
                   "params": int(l.count_params()),
                   "config": {k: l.get_config().get(k) for k in ("filters", "kernel_size", "units", "activation",
                                                                 "rate", "pool_size", "padding") if k in l.get_config()}})
out["cnn_layers"] = {"layers": layers, "total_params": int(model.count_params()),
                     "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights))}
(M / "round6_extras.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({"static": {k: out["static_validation_selected"][k] for k in ("validation_roc_auc", "chosen")},
                  "static_test": {k: round(v, 4) for k, v in out["static_validation_selected"]["test"].items()},
                  "cost": {k: (v["optimizer_steps"], v["sample_exposure"], round(v["wall_seconds_median"] / 60, 1))
                           for k, v in cost.items()},
                  "cnn_params": (out["cnn_layers"]["total_params"], out["cnn_layers"]["trainable_params"])}, indent=1))
