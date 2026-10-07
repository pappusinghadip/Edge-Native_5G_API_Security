"""Round-7 review: analyses that need no new training, plus the Time_To_Live runs of scripts/run_round7.sh.

time_to_live   Mac, 20N, seeds 101-103. Centralized runs from step 5 (r6feat) and federated runs from round 7
               (r7fed20), each with and without Time_To_Live. Reports each arm's drop and the centralized-minus-
               federated gap with and without the feature, paired by seed, with t intervals and the one-sided 95%
               upper bound used for non-inferiority.
capture_sensitive  Mac, 20N, seeds 101-103 (round 8): the 1D-CNN without Time_To_Live and Header_Length,
               centralized and federated, paired by seed with the original and with the no-Time_To_Live runs.
per_client     PC, 60N, the saved test scores of the non-IID FedAvg arms (seeds 101-105). Each client gets a
               held-out set drawn from the test split with its own training label mix: for every class, the test
               records of that class are shuffled (seed 42) and divided among the clients in the proportions the
               training partition gave them. The global model is scored on each client's set at the run's own
               validation-tuned threshold. The partitions skew labels only, so ranking quality (ROC-AUC) should
               vary little across clients while precision and the false-positive count follow each client's
               prevalence; clients without attacks have no recall or ROC-AUC. Each client's seed-mean ROC-AUC and
               precision get a 95% bootstrap interval over its held-out set (200 resamples).
partition_draws Mac, federated 60N, training seed 101: the thesis draw (partition seed 42) and four more (142, 242,
               342, 442; spaced because the partitioner retries a failed draw with seed + 1) at alpha 0.1 and 0.5,
               with each draw's client sizes, attack counts and the share of attacks held by clients that carry
               under 5% of the training flows (and so under 5% of the FedAvg weight), and the FedAvg weight of
               the client holding each attack flow, averaged over the attack flows.
Output: results/metrics/round7_analysis.json.

Run: PYTHONPATH=. .venv/bin/python scripts/round7_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

M = Path("results/metrics")
PC = Path("results/metrics_PC_run_2026-09-16_round5b")
SEEDS3, SEEDS5 = [101, 102, 103], [101, 102, 103, 104, 105]


def ci(d: np.ndarray) -> list[float]:
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - h), float(d.mean() + h)]


def upper95(d: np.ndarray) -> float:
    return float(d.mean() + stats.t.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))


def auc_of(path: Path) -> float:
    return json.loads(path.read_text())["runs"][0]["auc_roc"]


out: dict = {}

# --- Time_To_Live: centralized (step 5) and federated (round 7), 20N, Mac ---
A = {("cen", v): np.array([auc_of(M / f"repeated_centralized_r6feat_{v}_e20_s{s}.json") for s in SEEDS3])
     for v in ("orig", "nottl")}
A.update({("fed", v): np.array([auc_of(M / f"repeated_federated_iid_r7fed20_{v}_s{s}.json") for s in SEEDS3])
          for v in ("orig", "nottl")})
ttl = {"platform": "Mac", "exposure": "20N", "seeds": SEEDS3, "arms": {}}
for (arm, v), a in A.items():
    ttl["arms"][f"{arm}_{v}"] = {"auc_per_seed": a.tolist(), "auc_mean": float(a.mean()), "auc_std": float(a.std(ddof=1))}
for arm in ("cen", "fed"):
    d = A[(arm, "nottl")] - A[(arm, "orig")]
    ttl[f"{arm}_drop"] = {"mean": float(d.mean()), "ci95": ci(d), "per_seed": d.tolist()}
for v in ("orig", "nottl"):
    g = A[("cen", v)] - A[("fed", v)]
    ttl[f"gap_{v}"] = {"mean": float(g.mean()), "ci95": ci(g), "upper95_one_sided": upper95(g),
                       "per_seed": g.tolist(), "non_inferior": {str(m): upper95(g) < m for m in (0.02, 0.01, 0.005)}}
dd = (A[("fed", "nottl")] - A[("fed", "orig")]) - (A[("cen", "nottl")] - A[("cen", "orig")])
ttl["drop_difference_fed_minus_cen"] = {"mean": float(dd.mean()), "ci95": ci(dd)}
out["time_to_live"] = ttl

# --- round 8: capture-sensitive features, Time_To_Live and Header_Length removed (Mac, 20N, seeds 101-103) ---
if all((M / f).exists() for s in SEEDS3 for f in (f"repeated_centralized_r8nocap_e20_s{s}.json",
                                                   f"repeated_federated_iid_r8nocap_s{s}.json")):
    cc = np.array([auc_of(M / f"repeated_centralized_r8nocap_e20_s{s}.json") for s in SEEDS3])
    ff = np.array([auc_of(M / f"repeated_federated_iid_r8nocap_s{s}.json") for s in SEEDS3])
    cap = {"platform": "Mac", "exposure": "20N", "seeds": SEEDS3, "removed": ["Time_To_Live", "Header_Length"],
           "arms": {"cen": {"auc_per_seed": cc.tolist(), "auc_mean": float(cc.mean()), "auc_std": float(cc.std(ddof=1))},
                    "fed": {"auc_per_seed": ff.tolist(), "auc_mean": float(ff.mean()), "auc_std": float(ff.std(ddof=1))}}}
    for arm, a_ in (("cen", cc), ("fed", ff)):
        d = a_ - A[(arm, "orig")]
        cap[f"{arm}_drop"] = {"mean": float(d.mean()), "ci95": ci(d), "per_seed": d.tolist()}
        e = a_ - A[(arm, "nottl")]
        cap[f"{arm}_minus_nottl"] = {"mean": float(e.mean()), "ci95": ci(e)}
    g = cc - ff
    cap["gap"] = {"mean": float(g.mean()), "ci95": ci(g), "upper95_one_sided": upper95(g),
                  "non_inferior": {str(m): upper95(g) < m for m in (0.02, 0.01, 0.005)}}
    out["capture_sensitive"] = cap

# --- per-client held-out evaluation of the non-IID global models (PC) ---
test = np.load("data/processed/test.npz")
y = test["y"].astype(int)
parts = json.loads((M / "round6_extras.json").read_text())["partitions"]
rng = np.random.default_rng(42)
pc: dict = {"platform": "PC", "exposure": "60N", "seeds": SEEDS5, "alphas": {}}
for alpha in ("0.1", "0.5", "1.0"):
    clients = parts[f"dirichlet_{alpha}"]
    share = {1: np.array([c["attacks"] for c in clients], float), 0: np.array([c["rows"] - c["attacks"] for c in clients], float)}
    held = [[] for _ in clients]
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        cuts = np.floor(np.cumsum(share[cls] / share[cls].sum())[:-1] * len(idx)).astype(int)
        for k, part in enumerate(np.split(idx, cuts)):
            held[k].extend(part.tolist())
    per_seed, P, T = [], {}, {}
    for s in SEEDS5:
        stem = PC / f"fl_fedavg_dirichlet_r5_noniid_a{alpha.replace('.', 'p')}_s{s}"
        p = np.load(f"{stem}_probs.npz")["probs"]
        t = json.loads(Path(f"{stem}.json").read_text())["final"]["threshold"]
        P[s], T[s] = p, t
        rows = []
        for k, h in enumerate(held):
            h = np.array(h, int); yk, pk = y[h], p[h]; pred = pk >= t
            pos, neg = int(yk.sum()), int((1 - yk).sum())
            tp, fp = int((pred & (yk == 1)).sum()), int((pred & (yk == 0)).sum())
            rows.append({"client": k, "n": len(h), "attacks": pos, "prevalence": pos / len(h),
                         "auc_roc": float(roc_auc_score(yk, pk)) if pos and neg else None,
                         "recall": tp / pos if pos else None, "fpr": fp / neg if neg else None,
                         "precision": tp / (tp + fp) if tp + fp else None})
        per_seed.append({"seed": s, "threshold": t, "clients": rows})
    def boot(h, B=200):
        """95% percentile intervals of the seed-mean ROC-AUC and precision, resampling the held-out set (seed 0)."""
        g = np.random.default_rng(0); auc_b, prec_b = [], []
        for _ in range(B):
            r = h[g.integers(0, len(h), len(h))]; yr = y[r]
            if 0 < yr.sum() < len(yr):
                auc_b.append(np.mean([roc_auc_score(yr, P[s][r]) for s in SEEDS5]))
            pr = []
            for s in SEEDS5:
                pred = P[s][r] >= T[s]
                if pred.sum():
                    pr.append((pred & (yr == 1)).sum() / pred.sum())
            if pr:
                prec_b.append(np.mean(pr))
        q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 20 else None
        return q(auc_b), q(prec_b)
    summ = []
    for k in range(len(clients)):
        vals = lambda key: [r["clients"][k][key] for r in per_seed if r["clients"][k][key] is not None]
        summ.append({"client": k, "n": per_seed[0]["clients"][k]["n"], "attacks": per_seed[0]["clients"][k]["attacks"],
                     "prevalence": per_seed[0]["clients"][k]["prevalence"],
                     **{f"{m}_mean": (float(np.mean(vals(m))) if vals(m) else None)
                        for m in ("auc_roc", "recall", "fpr", "precision")}})
        hk = np.array(held[k], int)
        summ[-1]["auc_roc_ci95"], summ[-1]["precision_ci95"] = boot(hk) if len(hk) >= 20 else (None, None)
    aucs = [c["auc_roc_mean"] for c in summ if c["auc_roc_mean"] is not None and c["attacks"] >= 20]
    pooled = float(np.mean([json.loads(Path(PC / f"fl_fedavg_dirichlet_r5_noniid_a{alpha.replace('.', 'p')}_s{s}.json").read_text())["final"]["auc_roc"] for s in SEEDS5]))
    pc["alphas"][alpha] = {"clients": summ, "per_seed": per_seed, "pooled_auc_mean": pooled,
                           "auc_clients_with_20plus_attacks": {"worst": min(aucs), "median": float(np.median(aucs)),
                                                               "best": max(aucs), "n": len(aucs)} if aucs else None}
out["per_client"] = pc

# --- partition draws: Mac, federated 60N, training seed 101, five Dirichlet draws per alpha ---
fa = json.loads((M / "final_analysis.json").read_text())
iid_mac = fa["arms_mac"]["federated"]["summary"]["roc_auc"]
DRAWS = [42, 142, 242, 342, 442]
pdr: dict = {"platform": "Mac", "exposure": "60N", "training_seed": 101, "draws": DRAWS,
             "small_client_weight": 0.05, "iid_reference": {"auc_mean": iid_mac["mean"], "auc_std": iid_mac["std"],
                                                             "source": "Mac federated 60N, seeds 101-105"},
             "alphas": {}}
for alpha in ("0.1", "0.5"):
    at = alpha.replace(".", "p"); rows = []
    for d in DRAWS:
        run = json.loads((M / f"repeated_federated_dirichlet_r7draw{d}_a{at}_s101.json").read_text())["runs"][0]
        part = Path("data/partitions" if d == 42 else f"data/partitions_draw{d}")
        ys = [np.load(part / f"client_{k}_dir_{at}.npz")["y"] for k in range(5)]
        n = np.array([len(y) for y in ys]); a = np.array([int(y.sum()) for y in ys])
        small = n / n.sum() < pdr["small_client_weight"]
        rows.append({"draw": d, "auc_roc": run["auc_roc"], "f1": run["f1_binary"], "recall": run["recall_binary"],
                     "fpr": run["false_positive_rate"], "client_rows": n.tolist(), "client_attacks": a.tolist(),
                     "attack_share_in_small_clients": float(a[small].sum() / a.sum()),
                     # FedAvg weight of the client holding each attack flow, averaged over attack flows
                     "attack_weighted_fedavg_weight": float(((a / a.sum()) * (n / n.sum())).sum()),
                     "collapsed": run["auc_roc"] < iid_mac["mean"] - 0.1})
    au = np.array([r["auc_roc"] for r in rows])
    pdr["alphas"][alpha] = {"draws": rows, "auc_mean": float(au.mean()), "auc_std": float(au.std(ddof=1)),
                            "auc_min": float(au.min()), "auc_max": float(au.max()),
                            "collapsed": [r["draw"] for r in rows if r["collapsed"]]}
out["partition_draws"] = pdr

(M / "round7_analysis.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in ttl.items() if k != "arms"}, indent=1))
for a, r in pdr["alphas"].items():
    print("draws", a, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "draws"})
    for x in r["draws"]:
        print("   ", x["draw"], round(x["auc_roc"], 4), "small share", round(x["attack_share_in_small_clients"], 3), "attack-weighted w", round(x["attack_weighted_fedavg_weight"], 4), x["client_rows"], x["client_attacks"])
for a, r in pc["alphas"].items():
    print(a, "pooled", round(r["pooled_auc_mean"], 4), r["auc_clients_with_20plus_attacks"])
    for c in r["clients"]:
        print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
